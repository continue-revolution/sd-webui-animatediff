from typing import List, Tuple

import gradio as gr

from modules import script_callbacks, scripts
from modules.processing import (Processed, StableDiffusionProcessing,
                                StableDiffusionProcessingImg2Img)
from modules.scripts import PostprocessBatchListArgs

from scripts.animatediff_infv2v import AnimateDiffInfV2V
from scripts.animatediff_latent import AnimateDiffI2VLatent
from scripts.animatediff_logger import logger_animatediff as logger
from scripts.animatediff_mm import mm_animatediff as motion_module
from scripts.animatediff_prompt import AnimateDiffPromptSchedule
from scripts.animatediff_output import AnimateDiffOutput
from scripts.animatediff_xyz import patch_xyz, xyz_attrs
from scripts.animatediff_ui import AnimateDiffProcess, AnimateDiffUiGroup
from scripts.animatediff_settings import on_ui_settings
from scripts.animatediff_infotext import update_infotext, infotext_pasted
from scripts.animatediff_utils import get_animatediff_arg
from scripts.animatediff_i2ibatch import animatediff_hook_i2i_batch, animatediff_unhook_i2i_batch

script_dir = scripts.basedir()
motion_module.set_script_dir(script_dir)


class AnimateDiffScript(scripts.Script):

    def __init__(self):
        self.infotext_fields: List[Tuple[gr.components.IOComponent, str]] = []
        self.paste_field_names: List[str] = []


    def title(self):
        return "AnimateDiff"


    def show(self, is_img2img):
        return scripts.AlwaysVisible


    def ui(self, is_img2img):
        unit = AnimateDiffUiGroup().render(
            is_img2img,
            self.infotext_fields,
            self.paste_field_names
        )
        return (unit,)


    def before_process(self, p: StableDiffusionProcessing, params: AnimateDiffProcess):
        if p.is_api:
            params = get_animatediff_arg(p)
        motion_module.set_ad_params(params)

        # apply XYZ settings
        params.apply_xyz()
        xyz_attrs.clear()

        if params.enable:
            logger.info("AnimateDiff process start.")
            motion_module.load(params.model)
            params.set_p(p)
            params.prompt_scheduler = AnimateDiffPromptSchedule(p, params)
            update_infotext(p, params)


    def before_process_batch(self, p: StableDiffusionProcessing, params: AnimateDiffProcess, **kwargs):
        if params.enable and isinstance(p, StableDiffusionProcessingImg2Img) and not params.is_i2i_batch:
            AnimateDiffI2VLatent().randomize(p, params)


    def process_batch(self, p, params: AnimateDiffProcess, **kwargs):
        if params.enable:
            motion_module.set_ddim_alpha(p.sd_model)


    def process_before_every_sampling(self, p, params: AnimateDiffProcess, **kwargs):
        if params.enable:
            motion_module.inject(p.sd_model, params.model)


    def postprocess_batch_list(self, p: StableDiffusionProcessing, pp: PostprocessBatchListArgs, params: AnimateDiffProcess, **kwargs):
        if params.enable:
            params.prompt_scheduler.save_infotext_img(p)


    def postprocess(self, p: StableDiffusionProcessing, res: Processed, params: AnimateDiffProcess):
        if params.enable:
            params.prompt_scheduler.save_infotext_txt(res)
            AnimateDiffOutput().output(p, res, params)
            logger.info("AnimateDiff process end.")


patch_xyz()

script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_after_component(AnimateDiffUiGroup.on_after_component)
script_callbacks.on_cfg_denoiser(AnimateDiffInfV2V.animatediff_on_cfg_denoiser)
script_callbacks.on_infotext_pasted(infotext_pasted)
script_callbacks.on_before_ui(animatediff_hook_i2i_batch)
script_callbacks.on_script_unloaded(animatediff_unhook_i2i_batch)
