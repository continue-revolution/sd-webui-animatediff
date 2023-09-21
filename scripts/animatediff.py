import os
import gradio as gr

from modules import scripts, shared, script_callbacks
from modules.processing import (
    StableDiffusionProcessing,
    Processed,
    StableDiffusionProcessingImg2Img,
)
from scripts.animatediff_logger import logger_animatediff as logger
from scripts.animatediff_mm import mm_animatediff as motion_module
from scripts.animatediff_latent import AnimateDiffI2VLatent
from scripts.animatediff_ui import AnimateDiffProcess, AnimateDiffUiGroup
from scripts.animatediff_output import AnimateDiffOutput


script_dir = scripts.basedir()
motion_module.set_script_dir(script_dir)


class AnimateDiffScript(scripts.Script):
    def title(self):
        return "AnimateDiff"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        model_dir = shared.opts.data.get(
            "animatediff_model_path", os.path.join(script_dir, "model")
        )
        return (AnimateDiffUiGroup().render(is_img2img, model_dir),)

    def before_process(self, p: StableDiffusionProcessing, params: AnimateDiffProcess):
        if isinstance(params, dict): params = AnimateDiffProcess(**params)
        if params.enable:
            logger.info("AnimateDiff process start.")
            params.set_p(p)
            motion_module.inject(p.sd_model, params.model)

    def before_process_batch(
        self, p: StableDiffusionProcessing, params: AnimateDiffProcess, **kwargs
    ):
        if isinstance(params, dict): params = AnimateDiffProcess(**params)
        if params.enable and isinstance(p, StableDiffusionProcessingImg2Img):
            AnimateDiffI2VLatent().randomize(p, params)

    def postprocess(
        self, p: StableDiffusionProcessing, res: Processed, params: AnimateDiffProcess
    ):
        if isinstance(params, dict): params = AnimateDiffProcess(**params)
        if params.enable:
            motion_module.restore(p.sd_model)
            AnimateDiffOutput().output(p, res, params)
            logger.info("AnimateDiff process end.")


def on_ui_settings():
    section = ("animatediff", "AnimateDiff")
    shared.opts.add_option(
        "animatediff_model_path",
        shared.OptionInfo(
            os.path.join(script_dir, "model"),
            "Path to save AnimateDiff motion modules",
            gr.Textbox,
            section=section,
        ),
    )


script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_after_component(AnimateDiffUiGroup.on_after_component)
