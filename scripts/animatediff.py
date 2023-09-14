import os
import gc
import gradio as gr
import imageio
import torch
from einops import rearrange

from modules import scripts, images, shared, script_callbacks, hashes
from modules.devices import torch_gc, device, cpu
from modules.processing import StableDiffusionProcessing, Processed
from scripts.logging_animatediff import logger_animatediff
from motion_module import MotionWrapper, VanillaTemporalModule


from ldm.modules.diffusionmodules.openaimodel import TimestepBlock, TimestepEmbedSequential
from ldm.modules.diffusionmodules.util import GroupNorm32
from ldm.modules.attention import SpatialTransformer
def mm_tes_forward(self, x, emb, context=None):
    for layer in self:
        if isinstance(layer, TimestepBlock):
            x = layer(x, emb)
        elif isinstance(layer, (SpatialTransformer, VanillaTemporalModule)):
            x = layer(x, context)
        else:
            x = layer(x)
    return x
TimestepEmbedSequential.forward = mm_tes_forward
script_dir = scripts.basedir()
groupnorm32_original_forward = GroupNorm32.forward


class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)

    def get_block_name(self):
        return "button"


class AnimateDiffScript(scripts.Script):
    motion_module: MotionWrapper = None

    def __init__(self):
        self.logger = logger_animatediff
        self.prev_beta = None
        self.prev_alpha_cumprod = None
        self.prev_alpha_cumprod_prev = None

    def title(self):
        return "AnimateDiff"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def unload_motion_module(self):
        self.logger.info("Moving motion module to CPU")
        if AnimateDiffScript.motion_module is not None:
            AnimateDiffScript.motion_module.to(cpu)
        torch_gc()
        gc.collect()

    def remove_motion_module(self):
        self.logger.info("Removing motion module from any memory")
        del AnimateDiffScript.motion_module
        AnimateDiffScript.motion_module = None
        torch_gc()
        gc.collect()

    def ui(self, is_img2img):
        model_dir = shared.opts.data.get("animatediff_model_path", os.path.join(script_dir, "model"))
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        model_list = [f for f in os.listdir(model_dir) if f != ".gitkeep"]
        with gr.Accordion('AnimateDiff', open=False):
            with gr.Row():
                def refresh_models(*inputs):
                    new_model_list = [f for f in os.listdir(model_dir) if f != ".gitkeep"]
                    dd = inputs[0]
                    if dd in new_model_list:
                        selected = dd
                    elif len(new_model_list) > 0:
                        selected = new_model_list[0]
                    else:
                        selected = None
                    return gr.Dropdown.update(choices=new_model_list, value=selected)
                model = gr.Dropdown(choices=model_list, value=(model_list[0] if len(model_list) > 0 else None), label="Motion module", type="value")
                refresh_model = ToolButton(value='\U0001f504')
                refresh_model.click(refresh_models, model, model)
            with gr.Row():
                enable = gr.Checkbox(value=False, label='Enable AnimateDiff')
                video_length = gr.Slider(minimum=1, maximum=32, value=16, step=1, label="Number of frames", precision=0)
                fps = gr.Number(value=8, label="Frames per second (FPS)", precision=0)
                loop_number = gr.Number(minimum=0, value=0, label="Display loop number (0 = infinite loop)", precision=0)
            with gr.Row():
                unload = gr.Button(value="Move motion module to CPU (default if lowvram)")
                remove = gr.Button(value="Remove motion module from any memory")
                unload.click(fn=self.unload_motion_module)
                remove.click(fn=self.remove_motion_module)
        return enable, loop_number, video_length, fps, model

    def inject_motion_modules(self, p: StableDiffusionProcessing, model_name="mm_sd_v15.ckpt"):
        model_path = os.path.join(shared.opts.data.get("animatediff_model_path", os.path.join(script_dir, "model")), model_name)
        if not os.path.isfile(model_path):
            raise RuntimeError("Please download models manually.")
        def get_mm_hash(model_name="mm_sd_v15.ckpt"):
            model_hash = hashes.sha256(model_path, f"AnimateDiff/{model_name}")
            using_v2 = False
            if model_hash == 'aa7fd8a200a89031edd84487e2a757c5315460eca528fa70d4b3885c399bffd5':
                self.logger.info('You are using mm_sd_14.ckpt, which has been tested and supported.')
            elif model_hash == "cf16ea656cb16124990c8e2c70a29c793f9841f3a2223073fac8bd89ebd9b69a":
                self.logger.info('You are using mm_sd_15.ckpt, which has been tested and supported.')
            elif model_hash == "0aaf157b9c51a0ae07cb5d9ea7c51299f07bddc6f52025e1f9bb81cd763631df":
                self.logger.info('You are using mm-Stabilized_high.pth, which has been tested and supported.')
            elif model_hash == '39de8b71b1c09f10f4602f5d585d82771a60d3cf282ba90215993e06afdfe875':
                self.logger.info('You are using mm-Stabilized_mid.pth, which has been tested and supported.')
            elif model_hash == '3cb569f7ce3dc6a10aa8438e666265cb9be3120d8f205de6a456acf46b6c99f4':
                self.logger.info('You are using temporaldiff-v1-animatediff.ckpt, which has been tested and supported.')
            elif model_hash == '69ed0f5fef82b110aca51bcab73b21104242bc65d6ab4b8b2a2a94d31cad1bf0':
                self.logger.info('You are using mm_sd_v15_v2.ckpt, which has been tested and supported.')
                using_v2 = True
            else:
                self.logger.warn(f"Your model {model_name} has not been tested and supported. "
                                 "Either your download is incomplete or your model has not been tested. "
                                 "Please use at your own risk.")
            return model_hash, using_v2
        model_hash, using_v2 = get_mm_hash(model_name)
        if AnimateDiffScript.motion_module is None or AnimateDiffScript.motion_module.mm_hash != model_hash:
            self.logger.info(f"Loading motion module {model_name} from {model_path}")
            mm_state_dict = torch.load(model_path, map_location=device)
            AnimateDiffScript.motion_module = MotionWrapper(model_hash, using_v2)
            missed_keys = AnimateDiffScript.motion_module.load_state_dict(mm_state_dict)
            self.logger.warn(f"Missing keys {missed_keys}")
        AnimateDiffScript.motion_module.to(device)
        if not shared.cmd_opts.no_half:
            AnimateDiffScript.motion_module.half()
        unet = p.sd_model.model.diffusion_model
        if shared.opts.data.get("animatediff_hack_gn", False) and (not AnimateDiffScript.motion_module.using_v2):
            self.logger.info(f"Hacking GroupNorm32 forward function.")
            def groupnorm32_mm_forward(self, x):
                x = rearrange(x, '(b f) c h w -> b c f h w', b=2)
                x = groupnorm32_original_forward(self, x)
                x = rearrange(x, 'b c f h w -> (b f) c h w', b=2)
                return x
            GroupNorm32.forward = groupnorm32_mm_forward
        self.logger.info(f"Injecting motion module {model_name} into SD1.5 UNet input blocks.")
        for mm_idx, unet_idx in enumerate([1, 2, 4, 5, 7, 8, 10, 11]):
            mm_idx0, mm_idx1 = mm_idx // 2, mm_idx % 2
            unet.input_blocks[unet_idx].append(AnimateDiffScript.motion_module.down_blocks[mm_idx0].motion_modules[mm_idx1])
        self.logger.info(f"Injecting motion module {model_name} into SD1.5 UNet output blocks.")
        for unet_idx in range(12):
            mm_idx0, mm_idx1 = unet_idx // 3, unet_idx % 3
            if unet_idx % 3 == 2 and unet_idx != 11:
                unet.output_blocks[unet_idx].insert(-1, AnimateDiffScript.motion_module.up_blocks[mm_idx0].motion_modules[mm_idx1])
            else:
                unet.output_blocks[unet_idx].append(AnimateDiffScript.motion_module.up_blocks[mm_idx0].motion_modules[mm_idx1])
        if using_v2:
            self.logger.info(f"Injecting motion module {model_name} into SD1.5 UNet middle block.")
            unet.middle_block.insert(-1, AnimateDiffScript.motion_module.mid_block.motion_modules[0])
        self.logger.info(f"Injection finished.")

    def remove_motion_modules(self, p: StableDiffusionProcessing):
        unet = p.sd_model.model.diffusion_model
        self.logger.info(f"Removing motion module from SD1.5 UNet input blocks.")
        for unet_idx in [1, 2, 4, 5, 7, 8, 10, 11]:
            unet.input_blocks[unet_idx].pop(-1)
        self.logger.info(f"Removing motion module from SD1.5 UNet output blocks.")
        for unet_idx in range(12):
            if unet_idx % 3 == 2 and unet_idx != 11:
                unet.output_blocks[unet_idx].pop(-2)
            else:
                unet.output_blocks[unet_idx].pop(-1)
        if AnimateDiffScript.motion_module.using_v2:
            self.logger.info(f"Removing motion module from SD1.5 UNet middle block.")
            unet.middle_block.pop(-2)
        if shared.opts.data.get("animatediff_hack_gn", False) and (not AnimateDiffScript.motion_module.using_v2):
            self.logger.info(f"Restoring GroupNorm32 forward function.")
            GroupNorm32.forward = groupnorm32_original_forward
        self.logger.info(f"Removal finished.")
        if shared.cmd_opts.lowvram:
            self.unload_motion_module()
    
    def set_ddim_alpha(self, p: StableDiffusionProcessing):
        self.logger.info(f"Setting DDIM alpha.")
        beta_start = 0.00085
        beta_end = 0.012
        betas = torch.linspace(beta_start, beta_end, p.sd_model.num_timesteps, dtype=torch.float32, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            (torch.tensor([1.0], dtype=torch.float32, device=device), alphas_cumprod[:-1]))
        self.prev_beta = p.sd_model.betas
        p.sd_model.betas = betas
        self.prev_alpha_cumprod = p.sd_model.alphas_cumprod
        p.sd_model.alphas_cumprod = alphas_cumprod
        self.prev_alpha_cumprod_prev = p.sd_model.alphas_cumprod_prev
        p.sd_model.alphas_cumprod_prev = alphas_cumprod_prev
    
    def restore_ddim_alpha(self, p: StableDiffusionProcessing):
        self.logger.info(f"Restoring DDIM alpha.")
        p.sd_model.betas = self.prev_beta
        p.sd_model.alphas_cumprod = self.prev_alpha_cumprod
        p.sd_model.alphas_cumprod_prev = self.prev_alpha_cumprod_prev
        self.prev_beta = None
        self.prev_alpha_cumprod = None
        self.prev_alpha_cumprod_prev = None

    def before_process(self, p: StableDiffusionProcessing, enable_animatediff=False, loop_number=0, video_length=16, fps=8, model="mm_sd_v15.ckpt"):
        if enable_animatediff:
            self.logger.info(f"AnimateDiff process start with video Max frames {video_length}, FPS {fps}, duration {video_length/fps},  motion module {model}.")
            assert video_length > 0 and fps > 0, "Video length and FPS should be positive."
            p.batch_size = video_length
            self.inject_motion_modules(p, model)
            self.set_ddim_alpha(p)

    def postprocess(self, p: StableDiffusionProcessing, res: Processed, enable_animatediff=False, loop_number=0, video_length=16, fps=8, model="mm_sd_v15.ckpt"):
        if enable_animatediff:
            self.restore_ddim_alpha(p)
            self.remove_motion_modules(p)
            video_paths = []
            self.logger.info("Merging images into GIF.")
            from pathlib import Path
            Path(f"{p.outpath_samples}/AnimateDiff").mkdir(exist_ok=True, parents=True)
            for i in range(res.index_of_first_image, len(res.images), video_length):
                video_list = res.images[i:i+video_length]
                seq = images.get_next_sequence_number(f"{p.outpath_samples}/AnimateDiff", "")
                filename = f"{seq:05}-{res.seed}"
                video_path = f"{p.outpath_samples}/AnimateDiff/{filename}.gif"
                video_paths.append(video_path)
                imageio.mimsave(video_path, video_list, duration=(1/fps), loop=loop_number)
            res.images = video_paths
            self.logger.info("AnimateDiff process end.")

def on_ui_settings():
    section = ('animatediff', "AnimateDiff")
    shared.opts.add_option("animatediff_model_path", shared.OptionInfo(os.path.join(script_dir, "model"), "Path to save AnimateDiff motion modules", gr.Textbox, section=section))
    shared.opts.add_option("animatediff_hack_gn", shared.OptionInfo(
        True, "Check if you want to hack GroupNorm. By default, V1 hacks GroupNorm, which avoids a performance degradation. "
        "If you choose not to hack GroupNorm for V1, you will be able to use this extension in img2img in all cases, but the generated GIF will have flickers. "
        "V2 does not hack GroupNorm, so that this option will not influence v2 inference.", gr.Checkbox, section=section))


script_callbacks.on_ui_settings(on_ui_settings)
