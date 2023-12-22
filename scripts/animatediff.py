from modules import script_callbacks, scripts
from modules.processing import (Processed, StableDiffusionProcessing,
                                StableDiffusionProcessingImg2Img)
from modules.scripts import PostprocessBatchListArgs, PostprocessImageArgs

from scripts.animatediff_cn import AnimateDiffControl
from scripts.animatediff_infv2v import AnimateDiffInfV2V
from scripts.animatediff_latent import AnimateDiffI2VLatent
from scripts.animatediff_logger import logger_animatediff as logger
from scripts.animatediff_lora import AnimateDiffLora
from scripts.animatediff_mm import mm_animatediff as motion_module
from scripts.animatediff_prompt import AnimateDiffPromptSchedule
from scripts.animatediff_output import AnimateDiffOutput
from scripts.animatediff_ui import AnimateDiffProcess, AnimateDiffUiGroup
from scripts.animatediff_infotext import update_infotext
from scripts.animatediff_settings import on_ui_settings

script_dir = scripts.basedir()
motion_module.set_script_dir(script_dir)


class AnimateDiffScript(scripts.Script):

    def __init__(self):
        self.lora_hacker = None
        self.cfg_hacker = None
        self.cn_hacker = None
        self.prompt_scheduler = None
        self.hacked = False


    def title(self):
        return "AnimateDiff"


    def show(self, is_img2img):
        return scripts.AlwaysVisible


    def ui(self, is_img2img):
        return (AnimateDiffUiGroup().render(is_img2img, motion_module.get_model_dir()),)


    def before_process(self, p: StableDiffusionProcessing, params: AnimateDiffProcess):
        if p.is_api and isinstance(params, dict):
            self.ad_params = AnimateDiffProcess(**params)
            params = self.ad_params
        if params.enable:
            logger.info("AnimateDiff process start.")
            params.set_p(p)
            motion_module.inject(p.sd_model, params.model)
            self.prompt_scheduler = AnimateDiffPromptSchedule()
            self.lora_hacker = AnimateDiffLora(motion_module.mm.is_v2)
            self.lora_hacker.hack()
            self.cfg_hacker = AnimateDiffInfV2V(p, self.prompt_scheduler)
            self.cfg_hacker.hack(params)
            self.cn_hacker = AnimateDiffControl(p, self.prompt_scheduler)
            self.cn_hacker.hack(params)
            update_infotext(p, params)
            self.hacked = True
        elif self.hacked:
            self.cn_hacker.restore()
            self.cfg_hacker.restore()
            self.lora_hacker.restore()
            motion_module.restore(p.sd_model)
            self.hacked = False


    def before_process_batch(self, p: StableDiffusionProcessing, params: AnimateDiffProcess, **kwargs):
        if p.is_api and isinstance(params, dict): params = self.ad_params
        if params.enable and isinstance(p, StableDiffusionProcessingImg2Img) and not hasattr(p, '_animatediff_i2i_batch'):
            AnimateDiffI2VLatent().randomize(p, params)


    def postprocess_batch_list(self, p: StableDiffusionProcessing, pp: PostprocessBatchListArgs, params: AnimateDiffProcess, **kwargs):
        if p.is_api and isinstance(params, dict): params = self.ad_params
        if params.enable:
            self.prompt_scheduler.save_infotext_img(p)


    def postprocess_image(self, p: StableDiffusionProcessing, pp: PostprocessImageArgs, params: AnimateDiffProcess, *args):
        if p.is_api and isinstance(params, dict): params = self.ad_params
        if params.enable and isinstance(p, StableDiffusionProcessingImg2Img) and hasattr(p, '_animatediff_paste_to_full'):
            p.paste_to = p._animatediff_paste_to_full[p.batch_index]


    def postprocess(self, p: StableDiffusionProcessing, res: Processed, params: AnimateDiffProcess):
        if p.is_api and isinstance(params, dict): params = self.ad_params
        if params.enable:
            self.prompt_scheduler.save_infotext_txt(res)
            self.cn_hacker.restore()
            self.cfg_hacker.restore()
            self.lora_hacker.restore()
            motion_module.restore(p.sd_model)
            self.hacked = False
            AnimateDiffOutput().output(p, res, params)
            logger.info("AnimateDiff process end.")


script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_after_component(AnimateDiffUiGroup.on_after_component)
script_callbacks.on_before_ui(AnimateDiffUiGroup.on_before_ui)
