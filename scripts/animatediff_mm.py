import os

import torch
from modules import hashes, shared, sd_models
from modules_forge.unet_patcher import UnetPatcher

from motion_module import MotionWrapper, MotionModuleType
from scripts.animatediff_logger import logger_animatediff as logger
from ldm_patched.modules.model_management import get_torch_device


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
            self.mm = MotionWrapper(model_name, model_hash, model_type)
            missed_keys = self.mm.load_state_dict(mm_state_dict)
            logger.warn(f"Missing keys {missed_keys}")


    def inject(self, sd_model, model_name="mm_sd_v3.safetensors"):
        unet: UnetPatcher = sd_model.forge_objects.unet.clone()
        sd_ver = "SDXL" if sd_model.is_sdxl else "SD1.5"
        assert sd_model.is_sdxl == self.mm.is_xl, f"Motion module incompatible with SD. You are using {sd_ver} with {self.mm.mm_type}."
        input_block_map = {block_idx: mm_idx for mm_idx, block_idx in enumerate([1, 2, 4, 5, 7, 8, 10, 11])}

        # TODO: What's the best way to do GroupNorm32 forward function hack?
        if self.mm.enable_gn_hack():
            logger.warning(f"{sd_ver} GroupNorm32 forward function is NOT hacked. Performance will be degraded. Please use newer motion module")
            # from ldm_patched.ldm.modules.diffusionmodules import model as diffmodel
            # self.gn32_original_forward = diffmodel.Normalize
            # gn32_original_forward = self.gn32_original_forward

            # def groupnorm32_mm_forward(self, x):
            #     x = rearrange(x, "(b f) c h w -> b c f h w", b=2)
            #     x = gn32_original_forward(self, x)
            #     x = rearrange(x, "b c f h w -> (b f) c h w", b=2)
            #     return x

            # diffmodel.Normalize = groupnorm32_mm_forward

        logger.info(f"Injecting motion module {model_name} into {sd_ver} UNet.")

        def mm_block_modifier(x, identifier, layer, layer_index, ts, transformer_options):
            from ldm_patched.ldm.modules.attention import SpatialTransformer
            if identifier == "after" and isinstance(layer, SpatialTransformer):
                block_type, block_idx  = transformer_options["block"]
                if block_type == "middle":
                    if getattr(self.mm, "mid_block", None) is not None:
                        return self.mm.mid_block(x)
                elif block_type == "input":
                    block_idx = input_block_map[block_idx]
                    mm_idx0, mm_idx1 = block_idx // 2, block_idx % 2
                    return self.mm.down_blocks[mm_idx0].motion_modules[mm_idx1](x)
                elif block_type == "output":
                    mm_idx0, mm_idx1 = block_idx // 3, block_idx % 3
                    return self.mm.up_blocks[mm_idx0].motion_modules[mm_idx1](x)
            return x

        from scripts.animatediff_infv2v import AnimateDiffInfV2V
        mm_patcher = unet.add_extra_torch_module_during_sampling(self.mm)
        unet.add_extra_preserved_memory_during_sampling(mm_patcher.model_size())
        unet.set_model_unet_function_wrapper(AnimateDiffInfV2V.mm_sd_forward)
        unet.add_block_inner_modifier(mm_block_modifier)
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


mm_animatediff = AnimateDiffMM()
