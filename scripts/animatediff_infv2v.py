import numpy as np
import torch

from modules import shared
from modules.script_callbacks import CFGDenoiserParams
from scripts.animatediff_logger import logger_animatediff as logger
from scripts.animatediff_utils import get_animatediff_arg


class AnimateDiffInfV2V:
    cached_text_cond = None


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
        step: int = ...,
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
        ad_params = get_animatediff_arg(cfg_params.denoiser.p)
        if not ad_params.enable:
            return

        if cfg_params.denoiser.step == 0:
            # prompt travel
            prompt_closed_loop = (ad_params.video_length > ad_params.batch_size) and (ad_params.closed_loop in ['R+P', 'A'])
            AnimateDiffInfV2V.cached_text_cond = ad_params.prompt_scheduler.multi_cond(cfg_params.text_cond, prompt_closed_loop)
            from motion_module import MotionWrapper
            MotionWrapper.video_length = ad_params.batch_size

            # infinite generation
            def mm_sd_forward(self, x_in, sigma_in, cond):
                logger.debug("Running special forward for AnimateDiff")
                x_out = torch.zeros_like(x_in)
                for context in AnimateDiffInfV2V.uniform(cfg_params.denoiser.step, ad_params.video_length, ad_params.batch_size, ad_params.stride, ad_params.overlap, ad_params.closed_loop):
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

        cfg_params.text_cond = AnimateDiffInfV2V.cached_text_cond


    @staticmethod
    
