import gc
import numpy as np
import torch

from ldm_patched.modules.model_management import get_torch_device, soft_empty_cache
from modules import shared
from modules.sd_samplers_cfg_denoiser import pad_cond
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

        ad_params.step = cfg_params.denoiser.step
        if cfg_params.denoiser.step == 0:
            prompt_closed_loop = (ad_params.video_length > ad_params.batch_size) and (ad_params.closed_loop in ['R+P', 'A'])
            ad_params.text_cond = ad_params.prompt_scheduler.multi_cond(cfg_params.text_cond, prompt_closed_loop)

        #TODO: move this to cond modifier patch
        def pad_cond_uncond(cond, uncond):
            empty = shared.sd_model.cond_stage_model_empty_prompt
            num_repeats = (cond.shape[1] - uncond.shape[1]) // empty.shape[1]
            if num_repeats < 0:
                cond = pad_cond(cond, -num_repeats, empty)
            elif num_repeats > 0:
                uncond = pad_cond(uncond, num_repeats, empty)
            return cond, uncond
        cfg_params.text_cond, cfg_params.text_uncond = pad_cond_uncond(ad_params.text_cond, cfg_params.text_uncond)


    @staticmethod
    def mm_sd_forward(apply_model, info):
        logger.debug("Running special forward for AnimateDiff")
        x_out = torch.zeros_like(info["input"])
        ad_params = motion_module.ad_params
        for context in AnimateDiffInfV2V.uniform(ad_params.step, ad_params.video_length, ad_params.batch_size, ad_params.stride, ad_params.overlap, ad_params.closed_loop):
            if x_out.shape[0] == 2 * ad_params.video_length:
                _context = context + [c + ad_params.video_length for c in context]
            else:
                _context = context

            info_c = {}
            for k, v in info["c"].items():
                if isinstance(v, torch.Tensor):
                    if v.shape[0] == 2 * ad_params.video_length:
                        info_c[k] = v[_context]
                    elif v.shape[0] == ad_params.video_length:
                        info_c[k] = v[context]
                    else:
                        info_c[k] = v
                elif isinstance(v, list):
                    if len(v) == 2 * ad_params.video_length:
                        info_c[k] = [v[i] for i in _context]
                    elif len(v) == ad_params.video_length:
                        info_c[k] = [v[i] for i in context]
                    else:
                        info_c[k] = v
                else:
                    if k == "control":
                        current_control = {}
                        for c_k, c_v in v.items(): # c_k: "input" | "middle" | "output"
                            current_control[c_k] = []
                            for c_tensor in c_v:
                                if c_tensor.shape[0] == 2 * ad_params.video_length:
                                    current_control[c_k].append(c_tensor[_context].to(get_torch_device()))
                                elif c_tensor.shape[0] == ad_params.video_length:
                                    current_control[c_k].append(c_tensor[context].to(get_torch_device()))
                                else:
                                    current_control[c_k].append(c_tensor.to(get_torch_device()))
                        info_c[k] = current_control
                    else:          
                        info_c[k] = v

            out = apply_model(info["input"][_context], info["timestep"][_context], **info_c)
            x_out = x_out.to(dtype=out.dtype)
            x_out[_context] = out

            del info_c
            soft_empty_cache(True)
            gc.collect()

        return x_out
