import os

import torch
from modules import hashes, shared, sd_models
from modules_forge.unet_patcher import UnetPatcher

from ldm_patched.modules.model_management import get_torch_device, unet_dtype, unet_manual_cast
from ldm_patched.modules.ops import manual_cast

from motion_module import MotionWrapper, MotionModuleType
from scripts.animatediff_logger import logger_animatediff as logger


class AnimateDiffMM:
    def __init__(self):
        self.mm: MotionWrapper = None
        self.script_dir = None
        self.ad_params = None


    def set_script_dir(self, script_dir):
        self.script_dir = script_dir


    def set_ad_params(self, ad_params):
        self.ad_params = ad_params


    def get_model_dir(self):
        model_dir = shared.opts.data.get("animatediff_model_path", os.path.join(self.script_dir, "model"))
        if not model_dir:
            model_dir = os.path.join(self.script_dir, "model")
        return model_dir


    def load(self, model_name: str):
        model_path = os.path.join(self.get_model_dir(), model_name)
        if not os.path.isfile(model_path):
            raise RuntimeError("Please download models manually.")
        if self.mm is None or self.mm.mm_name != model_name:
            logger.info(f"Loading motion module {model_name} from {model_path}")
            model_hash = hashes.sha256(model_path, f"AnimateDiff/{model_name}")
            mm_state_dict = sd_models.read_state_dict(model_path)
            model_type = MotionModuleType.get_mm_type(mm_state_dict)
            logger.info(f"Guessed {model_name} architecture: {model_type}")
            mm_config = dict(mm_name=model_name, mm_hash=model_hash, mm_type=model_type)
            if unet_manual_cast(unet_dtype(), get_torch_device()) is not None:
                mm_config["operations"] = manual_cast
            self.mm = MotionWrapper(**mm_config)
            self.mm.load_state_dict(mm_state_dict)
        self.set_layer_mapping(shared.sd_model)


    def inject(self, sd_model, model_name="mm_sd15_v3.safetensors"):
        unet: UnetPatcher = sd_model.forge_objects.unet.clone()
        sd_ver = "SDXL" if sd_model.is_sdxl else "SD1.5"
        assert sd_model.is_sdxl == self.mm.is_xl, f"Motion module incompatible with SD. You are using {sd_ver} with {self.mm.mm_type}."

        if self.mm.enable_gn_hack():
            try:
                from einops import rearrange
                def groupnorm32_mm_forward(gn32_original_forward, x, transformer_options={}):
                    x = rearrange(x, "(b f) c h w -> b c f h w", f=self.ad_params.batch_size)
                    x = gn32_original_forward(self, x)
                    x = rearrange(x, "b c f h w -> (b f) c h w", f=self.ad_params.batch_size)
                    return x
                unet.set_groupnorm_wrapper(groupnorm32_mm_forward)
            except:
                logger.warning(f"{sd_ver} GroupNorm32 forward function is NOT hacked. Performance will be degraded. Please use newer motion module")

        logger.info(f"Injecting motion module {model_name} into {sd_ver} UNet.")

        input_block_map = {block_idx: mm_idx for mm_idx, block_idx in enumerate([1, 2, 4, 5, 7, 8, 10, 11])}
        def mm_block_modifier(x, identifier, layer, layer_index, ts, transformer_options):
            block_type, block_idx  = transformer_options["block"]
            if layer_index == len(ts) - 1: # we only do motion module injection on the last layer
                if block_type == "middle" and getattr(self.mm, "mid_block", None) is not None and identifier == "before":
                    return self.mm.mid_block(x) # between SpatialTransformer and ResBlock
                elif block_type == "input" and block_idx in input_block_map and identifier == "after":
                    block_idx_ = input_block_map[block_idx]
                    mm_idx0, mm_idx1 = block_idx_ // 2, block_idx_ % 2
                    return self.mm.down_blocks[mm_idx0].motion_modules[mm_idx1](x)
                elif block_type == "output":
                    apply_before = block_idx % 3 == 2 and block_idx != (8 if sd_model.is_sdxl else 11)
                    if identifier == ("before" if apply_before else "after"):
                        mm_idx0, mm_idx1 = block_idx // 3, block_idx % 3
                        return self.mm.up_blocks[mm_idx0].motion_modules[mm_idx1](x)
            return x

        def mm_memory_estimator(input_shape):
            input_shape[0] = 2 * self.ad_params.batch_size
            return unet.memory_required(input_shape)

        def mm_cn_forward(model, inner_model, hint, **kwargs):
            controls = []
            control_batch_size = shared.opts.data.get("animatediff_control_batch_size", 0)
            if control_batch_size == 0:
                control_batch_size = 2 * self.ad_params.batch_size
            for i in range(0, hint.shape[0], control_batch_size):
                current_kwargs = {k: (v[i:i + control_batch_size].to(get_torch_device())
                                  if type(v) == torch.Tensor else v) for k, v in kwargs.items()}
                current_kwargs["hint"] = hint[i:i + control_batch_size].to(get_torch_device())
                current_ctrl = inner_model(**current_kwargs)
                if len(controls) == 0:
                    controls = [[c.cpu() if type(c) == torch.Tensor else c] for c in current_ctrl]
                else:
                    for i, c in enumerate(current_ctrl):
                        controls[i].append(c.cpu() if type(c) == torch.Tensor else c)

            for i in range(len(controls)):
                if type(controls[i][0]) == torch.Tensor:
                    controls[i] = torch.cat(controls[i], dim=0)
                else:
                    controls[i] = controls[i][0] # should be None for T2I Adapters

            model.transformer_options["cond_mark"] = model.transformer_options["cond_mark"].cpu()
            return controls

        from scripts.animatediff_infv2v import AnimateDiffInfV2V
        mm_patcher = unet.add_extra_torch_module_during_sampling(self.mm)
        unet.add_extra_preserved_memory_during_sampling(mm_patcher.model_size())
        unet.set_model_unet_function_wrapper(AnimateDiffInfV2V.mm_sd_forward)
        unet.add_block_inner_modifier(mm_block_modifier)
        unet.set_memory_peak_estimation_modifier(mm_memory_estimator)
        if shared.opts.data.get("animatediff_disable_control_wrapper", False):
            logger.warning("ControlNet wrapper is disabled. Be cautious that you may run out of VRAM.")
        else:
            unet.set_controlnet_model_function_wrapper(mm_cn_forward)
        sd_model.forge_objects.unet = unet


    def set_ddim_alpha(self, sd_model):
        logger.info(f"Setting DDIM alpha.")
        unet: UnetPatcher = sd_model.forge_objects.unet.clone()
        beta_start = 0.00085
        beta_end = 0.020 if self.mm.is_adxl else 0.012
        if self.mm.is_adxl:
            betas = torch.linspace(beta_start**0.5, beta_end**0.5, 1000, dtype=torch.float32, device=get_torch_device()) ** 2
        else:
            betas = torch.linspace(
                beta_start,
                beta_end,
                1000 if sd_model.is_sdxl else sd_model.num_timesteps,
                dtype=torch.float32,
                device=get_torch_device())
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        unet.add_alphas_cumprod_modifier(lambda _: alphas_cumprod)
        sd_model.forge_objects.unet = unet


    def set_layer_mapping(self, sd_model):
        if hasattr(sd_model, 'network_layer_mapping'):
            for name, module in self.mm.named_modules():
                network_name = name.replace(".", "_")
                sd_model.network_layer_mapping[network_name] = module
                module.network_layer_name = network_name


mm_animatediff = AnimateDiffMM()
