import os
import gradio as gr
import imageio
import torch

from modules import scripts, images, shared, script_callbacks
from modules.devices import device
from modules.processing import StableDiffusionProcessing, Processed, StableDiffusionProcessingImg2Img
from scripts.animatediff_logger import logger_animatediff as logger
from scripts.animatediff_ui import AnimateDiffProcess, AnimateDiffUiGroup
from scripts.animatediff_mm import mm_animatediff as motion_module


script_dir = scripts.basedir()
motion_module.set_script_dir(script_dir)


class AnimateDiffScript(scripts.Script):

    def title(self):
        return "AnimateDiff"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        model_dir = shared.opts.data.get("animatediff_model_path", os.path.join(script_dir, "model"))
        ui_group = AnimateDiffUiGroup()
        return (ui_group.render(is_img2img, model_dir),)

    def before_process(self, p: StableDiffusionProcessing, params: AnimateDiffProcess):
        if params.enable:
            logger.info(f"AnimateDiff process start with video Max frames {params.video_length}, FPS {params.fps}, duration {params.video_length/params.fps},  motion module {params.model}.")
            assert params.video_length > 0 and params.fps > 0, "Video length and FPS should be positive."
            p.batch_size = params.video_length
            motion_module.inject(p.sd_model, params.model)
    
    def before_process_batch(self, p: StableDiffusionProcessing, params: AnimateDiffProcess, **kwargs):
        if params.enable and isinstance(p, StableDiffusionProcessingImg2Img):
            init_alpha = []
            for i in range(params.video_length):
                init_alpha.append(params.video_length - i / params.video_length)
            logger.info(f'Randomizing init_latent according to {init_alpha}.')
            init_alpha = torch.tensor(init_alpha, dtype=torch.float32, device=device)[:, None, None, None]
            p.init_latent = p.init_latent * init_alpha + p.rng.next() * (1 - init_alpha)

    def postprocess(self, p: StableDiffusionProcessing, res: Processed, params: AnimateDiffProcess):
        if params.enable:
            motion_module.restore(p.sd_model)
            video_paths = []
            logger.info("Merging images into GIF.")
            from pathlib import Path
            Path(f"{p.outpath_samples}/AnimateDiff").mkdir(exist_ok=True, parents=True)
            for i in range(res.index_of_first_image, len(res.images), params.video_length):
                video_list = res.images[i:i+params.video_length]
                seq = images.get_next_sequence_number(f"{p.outpath_samples}/AnimateDiff", "")
                filename = f"{seq:05}-{res.seed}"
                video_path = f"{p.outpath_samples}/AnimateDiff/{filename}.gif"
                video_paths.append(video_path)
                imageio.mimsave(video_path, video_list, duration=(1/params.fps), loop=params.loop_number)
            res.images = video_paths
            logger.info("AnimateDiff process end.")

def on_ui_settings():
    section = ('animatediff', "AnimateDiff")
    shared.opts.add_option("animatediff_model_path", shared.OptionInfo(os.path.join(script_dir, "model"), "Path to save AnimateDiff motion modules", gr.Textbox, section=section))
    shared.opts.add_option("animatediff_hack_gn", shared.OptionInfo(
        True, "Check if you want to hack GroupNorm. By default, V1 hacks GroupNorm, which avoids a performance degradation. "
        "If you choose not to hack GroupNorm for V1, you will be able to use this extension in img2img in all cases, but the generated GIF will have flickers. "
        "V2 does not hack GroupNorm, so that this option will not influence v2 inference.", gr.Checkbox, section=section))


script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_after_component(AnimateDiffUiGroup.on_after_component)
