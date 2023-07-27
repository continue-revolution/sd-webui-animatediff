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


class AnimateDiffScript(scripts.Script):
    motion_module: MotionWrapper = None
    
    def __init__(self):
        self.logger = logger_animatediff

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
        with gr.Accordion('AnimateDiff', open=False):
            model = gr.Dropdown(choices=["mm_sd_v15.ckpt", "mm_sd_v14.ckpt"], value="mm_sd_v15.ckpt", label="Motion module", type="value")
            with gr.Row():
                enable = gr.Checkbox(value=False, label='Enable AnimateDiff')
                video_length = gr.Slider(minimum=1, maximum=24, value=16, step=1, label="Number of frames", precision=0)
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
        if AnimateDiffScript.motion_module is None or AnimateDiffScript.motion_module.mm_type != model_name:
            if shared.opts.data.get("animatediff_check_hash", True):
                def get_mm_hash(model_name="mm_sd_v15.ckpt"):
                    if model_name == "mm_sd_v14.ckpt":
                        return 'aa7fd8a200a89031edd84487e2a757c5315460eca528fa70d4b3885c399bffd5'
                    elif model_name == "mm_sd_v15.ckpt":
                        return 'b925e6a0275bdd63e04a6f20e88f3b96aa2043958bd858bef303374f3c2ca0bc'
                    else:
                        raise RuntimeError(f"Unsupported model filename {model_name}. Should be one of mm_sd_v14.ckpt or mm_sd_v15.ckpt")
                if hashes.sha256(model_path, f"AnimateDiff/{model_name}") != get_mm_hash(model_name):
                    raise RuntimeError(f"{model_name} hash mismatch. You probably need to re-download the motion module.")
            self.logger.info(f"Loading motion module {model_name} from {model_path}")
            mm_state_dict = torch.load(model_path, map_location=device)
            AnimateDiffScript.motion_module = MotionWrapper(model_name)
            missed_keys = AnimateDiffScript.motion_module.load_state_dict(mm_state_dict)
            self.logger.warn(f"Missing keys {missed_keys}")
        AnimateDiffScript.motion_module.to(device)
        unet = p.sd_model.model.diffusion_model
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
        self.logger.info(f"Restoring GroupNorm32 forward function.")
        GroupNorm32.forward = groupnorm32_original_forward
        self.logger.info(f"Removal finished.")
        if shared.cmd_opts.lowvram:
            self.unload_motion_module()
    
    def before_process(self, p: StableDiffusionProcessing, enable_animatediff=False, loop_number=0, video_length=16, fps=8, model="mm_sd_v15.ckpt"):
        if enable_animatediff:
            self.logger.info(f"AnimateDiff process start with video Max frames {video_length}, FPS {fps}, duration {video_length/fps},  motion module {model}.")
            assert video_length > 0 and fps > 0, "Video length and FPS should be positive."
            p.batch_size = video_length
            self.inject_motion_modules(p, model)
    
    def postprocess(self, p: StableDiffusionProcessing, res: Processed, enable_animatediff=False, loop_number=0, video_length=16, fps=8, model="mm_sd_v15.ckpt"):
        if enable_animatediff:
            self.remove_motion_modules(p)
            video_paths = []
            self.logger.info("Merging images into GIF.")
            from pathlib import Path
            Path(f"{p.outpath_samples}/AnimateDiff").mkdir(exist_ok=True)
            for i in range(res.index_of_first_image, len(res.images), video_length):
                video_list = res.images[i:i+video_length]
                seq = images.get_next_sequence_number(f"{p.outpath_samples}/AnimateDiff", "")
                filename = f"{seq:05}-{res.seed}"
                video_path = f"{p.outpath_samples}/AnimateDiff/{filename}.gif"
                video_paths.append(video_path)
                imageio.mimsave(video_path, video_list, duration=(video_length/fps), loop=loop_number)
            res.images = video_paths
            self.logger.info("AnimateDiff process end.")

def on_ui_settings():
    section = ('animatediff', "AnimateDiff")
    shared.opts.add_option("animatediff_model_path", shared.OptionInfo(os.path.join(script_dir, "model"), "Path to save AnimateDiff motion modules", gr.Textbox, section=section))
    shared.opts.add_option("animatediff_check_hash", shared.OptionInfo(True, "Check hash for motion modules. Disable checking if you want to use your own.", gr.Checkbox, section=section))


script_callbacks.on_ui_settings(on_ui_settings)
