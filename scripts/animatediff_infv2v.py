from typing import List
from types import MethodType

import numpy as np
import torch

from modules import devices, shared
from modules.script_callbacks import CFGDenoiserParams
from scripts.animatediff_logger import logger_animatediff as logger
from scripts.animatediff_mm import mm_animatediff as motion_module


class AnimateDiffInfV2V:

    # Returns fraction that has denominator that is a power of 2
    @staticmethod
    def ordered_halving(val):
        # get binary value, padded with 0s for 64 bits
        bin_str = f"{val:064b}"
        # flip binary value, padding included
        bin_flip = bin_str[::-1]
        # convert binary to int
        as_int = int(bin_flip, 2)
        # divide by 1 << 64, equivalent to 2**64, or 18446744073709551616,
        # or b10000000000000000000000000000000000000000000000000000000000000000 (1 with 64 zero's)
        final = as_int / (1 << 64)
        return final


    # Generator that returns lists of latent indeces to diffuse on
    @staticmethod
    def uniform(
        step: int,
        video_length: int = 0,
        batch_size: int = 16,
        stride: int = 1,
        overlap: int = 4,
        loop_setting: str = 'R-P',
    ):
        if video_length <= batch_size:
            yield list(range(batch_size))
            return

        closed_loop = (loop_setting == 'A')
        stride = min(stride, int(np.ceil(np.log2(video_length / batch_size))) + 1)

        for context_step in 1 << np.arange(stride):
            pad = int(round(video_length * AnimateDiffInfV2V.ordered_halving(step)))
            both_close_loop = False
            for j in range(
                int(AnimateDiffInfV2V.ordered_halving(step) * context_step) + pad,
                video_length + pad + (0 if closed_loop else -overlap),
                (batch_size * context_step - overlap),
            ):
                if loop_setting == 'N' and context_step == 1:
                    current_context = [e % video_length for e in range(j, j + batch_size * context_step, context_step)]
                    first_context = [e % video_length for e in range(0, batch_size * context_step, context_step)]
                    last_context = [e % video_length for e in range(video_length - batch_size * context_step, video_length, context_step)]
                    def get_unsorted_index(lst):
                        for i in range(1, len(lst)):
                            if lst[i] < lst[i-1]:
                                return i
                        return None
                    unsorted_index = get_unsorted_index(current_context)
                    if unsorted_index is None:
                        yield current_context
                    elif both_close_loop: # last and this context are close loop
                        both_close_loop = False
                        yield first_context
                    elif unsorted_index < batch_size - overlap: # only this context is close loop
                        yield last_context
                        yield first_context
                    else: # this and next context are close loop
                        both_close_loop = True
                        yield last_context
                else:
                    yield [e % video_length for e in range(j, j + batch_size * context_step, context_step)]


    @staticmethod
    def animatediff_on_cfg_denoiser(cfg_params: CFGDenoiserParams):
        ad_params = motion_module.ad_params
        if ad_params is None or not ad_params.enable:
            return

        # !adetailer accomodation
        if not motion_module.mm_injected:
            if cfg_params.denoiser.step == 0:
                logger.warning(
                    "No motion module detected, falling back to the original forward. You are most likely using !Adetailer. "
                    "!Adetailer post-process your outputs sequentially, and there will NOT be motion module in your UNet, "
                    "so there might be NO temporal consistency within the inpainted face. Use at your own risk. "
                    "If you really want to pursue inpainting with AnimateDiff inserted into UNet, "
                    "use Segment Anything to generate masks for each frame and inpaint them with AnimateDiff + ControlNet. "
                    "Note that my proposal might be good or bad, do your own research to figure out the best way.")
            return

        if cfg_params.denoiser.step == 0 and getattr(cfg_params.denoiser.inner_model, 'original_forward', None) is None:

            # prompt travel
            prompt_closed_loop = (ad_params.video_length > ad_params.batch_size) and (ad_params.closed_loop in ['R+P', 'A'])
            ad_params.text_cond = ad_params.prompt_scheduler.multi_cond(cfg_params.text_cond, prompt_closed_loop)
            try:
                from scripts.external_code import find_cn_script
                cn_script = find_cn_script(cfg_params.denoiser.p.scripts)
            except:
                cn_script = None

            # infinite generation
            def mm_cn_select(context: List[int]):
                # take control images for current context.
                if cn_script and cn_script.latest_network:
                    from scripts.hook import ControlModelType
                    for control in cn_script.latest_network.control_params:
                        if control.control_model_type not in [ControlModelType.IPAdapter, ControlModelType.Controlllite]:
                            if control.hint_cond.shape[0] > len(context):
                                control.hint_cond_backup = control.hint_cond
                                control.hint_cond = control.hint_cond[context]
                            control.hint_cond = control.hint_cond.to(device=devices.get_device_for("controlnet"))
                            if control.hr_hint_cond is not None:
                                if control.hr_hint_cond.shape[0] > len(context):
                                    control.hr_hint_cond_backup = control.hr_hint_cond
                                    control.hr_hint_cond = control.hr_hint_cond[context]
                                control.hr_hint_cond = control.hr_hint_cond.to(device=devices.get_device_for("controlnet"))
                        # IPAdapter and Controlllite are always on CPU.
                        elif control.control_model_type == ControlModelType.IPAdapter and control.control_model.image_emb.cond_emb.shape[0] > len(context):
                            from scripts.controlmodel_ipadapter import ImageEmbed
                            if getattr(control.control_model.image_emb, "cond_emb_backup", None) is None:
                                control.control_model.cond_emb_backup = control.control_model.image_emb.cond_emb
                            control.control_model.image_emb = ImageEmbed(control.control_model.cond_emb_backup[context], control.control_model.image_emb.uncond_emb)
                        elif control.control_model_type == ControlModelType.Controlllite:
                            for module in control.control_model.modules.values():
                                if module.cond_image.shape[0] > len(context):
                                    module.cond_image_backup = module.cond_image
                                    module.set_cond_image(module.cond_image[context])
            
            def mm_cn_restore(context: List[int]):
                # restore control images for next context
                if cn_script and cn_script.latest_network:
                    from scripts.hook import ControlModelType
                    for control in cn_script.latest_network.control_params:
                        if control.control_model_type not in [ControlModelType.IPAdapter, ControlModelType.Controlllite]:
                            if getattr(control, "hint_cond_backup", None) is not None:
                                control.hint_cond_backup[context] = control.hint_cond.to(device="cpu")
                                control.hint_cond = control.hint_cond_backup
                            if control.hr_hint_cond is not None and getattr(control, "hr_hint_cond_backup", None) is not None:
                                control.hr_hint_cond_backup[context] = control.hr_hint_cond.to(device="cpu")
                                control.hr_hint_cond = control.hr_hint_cond_backup
                        elif control.control_model_type == ControlModelType.Controlllite:
                            for module in control.control_model.modules.values():
                                if getattr(module, "cond_image_backup", None) is not None:
                                    module.set_cond_image(module.cond_image_backup)

            def mm_sd_forward(self, x_in, sigma_in, cond):
                logger.debug("Running special forward for AnimateDiff")
                x_out = torch.zeros_like(x_in)
                for context in AnimateDiffInfV2V.uniform(ad_params.step, ad_params.video_length, ad_params.batch_size, ad_params.stride, ad_params.overlap, ad_params.closed_loop):
                    if shared.opts.batch_cond_uncond:
                        _context = context + [c + ad_params.video_length for c in context]
                    else:
                        _context = context
                    mm_cn_select(_context)
                    out = self.original_forward(
                        x_in[_context], sigma_in[_context],
                        cond={k: ([v[0][_context]] if isinstance(v, list) else v[_context]) for k, v in cond.items()})
                    x_out = x_out.to(dtype=out.dtype)
                    x_out[_context] = out
                    mm_cn_restore(_context)
                return x_out

            logger.info("inner model forward hooked")
            cfg_params.denoiser.inner_model.original_forward = cfg_params.denoiser.inner_model.forward
            cfg_params.denoiser.inner_model.forward = MethodType(mm_sd_forward, cfg_params.denoiser.inner_model)

        cfg_params.text_cond = ad_params.text_cond
        ad_params.step = cfg_params.denoiser.step
