from modules.processing import StableDiffusionProcessing, StableDiffusionProcessingImg2Img

from scripts.animatediff_ui import AnimateDiffProcess

def update_infotext(p: StableDiffusionProcessing, params: AnimateDiffProcess):
    if p.extra_generation_params is not None:
        p.extra_generation_params["AnimateDiff"] = params.get_dict(isinstance(p, StableDiffusionProcessingImg2Img))
