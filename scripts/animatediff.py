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
from scripts.animatediff_output import AnimateDiffOutput


script_dir = scripts.basedir()
motion_module.set_script_dir(script_dir)


class AnimateDiffScript(scripts.Script):

    def title(self):
        return "AnimateDiff"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        model_dir = shared.opts.data.get("animatediff_model_path", os.path.join(script_dir, "model"))
        return (AnimateDiffUiGroup().render(is_img2img, model_dir),)

    def before_process(self, p: StableDiffusionProcessing, params: AnimateDiffProcess):
        if params.enable:
            logger.info("AnimateDiff process start.")
            params.set_p(p)
            motion_module.inject(p.sd_model, params.model)
    
    def before_process_batch(self, p: StableDiffusionProcessing, params: AnimateDiffProcess, **kwargs):
        if params.enable and isinstance(p, StableDiffusionProcessingImg2Img):
            init_alpha = []
            for i in range(params.video_length):
                init_alpha.append(1 - i / params.video_length)
            logger.info(f'Randomizing init_latent according to {init_alpha}.')
            init_alpha = torch.tensor(init_alpha, dtype=torch.float32, device=device)[:, None, None, None]
            p.init_latent = p.init_latent * init_alpha + p.rng.next() * (1 - init_alpha)

    def postprocess(self, p: StableDiffusionProcessing, res: Processed, params: AnimateDiffProcess):
        if params.enable:
            motion_module.restore(p.sd_model)
            AnimateDiffOutput().output(p, res, params)
            logger.info("AnimateDiff process end.")


def on_ui_settings():
    section = ('animatediff', "AnimateDiff")
    shared.opts.add_option("animatediff_model_path", shared.OptionInfo(os.path.join(script_dir, "model"), "Path to save AnimateDiff motion modules", gr.Textbox, section=section))


script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_after_component(AnimateDiffUiGroup.on_after_component)
