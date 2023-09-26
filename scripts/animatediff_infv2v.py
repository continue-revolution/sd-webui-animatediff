import numpy as np
from modules.sd_samplers_cfg_denoiser import CFGDenoiser

from scripts.animatediff_logger import logger_animatediff as logger
from scripts.animatediff_ui import AnimateDiffProcess


class AnimateDiffInfV2V:

    def __init__(self):
        self.cfg_original_forward = None


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
        closed_loop: bool = True,
    ):
        if video_length <= batch_size:
            yield list(range(batch_size))
            return

        stride = min(stride, int(np.ceil(np.log2(video_length / batch_size))) + 1)

        for context_step in 1 << np.arange(stride):
            pad = int(round(video_length * AnimateDiffInfV2V.ordered_halving(step)))
            for j in range(
                int(AnimateDiffInfV2V.ordered_halving(step) * context_step) + pad,
                video_length + pad + (0 if closed_loop else -overlap),
                (batch_size * context_step - overlap),
            ):
                batch_list = [e % video_length for e in range(j, j + batch_size * context_step, context_step)]
                if not closed_loop and batch_list[-1] < batch_list[0]:
                    batch_list_end = batch_list[: video_length - batch_list[0]]
                    batch_list_front = batch_list[video_length - batch_list[0] :]
                    if len(batch_list_end) < len(batch_list_front):
                        batch_list_front_end = batch_list_front[-1]
                        for i in range(len(batch_list_end)):
                            batch_list_front.append(batch_list_front_end + i + 1)
                        yield batch_list_front
                    else:
                        batch_list_end_front = batch_list_end[0]
                        for i in range(len(batch_list_front)):
                            batch_list_end.insert(0, batch_list_end_front - i - 1)
                        yield batch_list_end
                else:
                    yield batch_list


    def hack_cfg_forward(self, params: AnimateDiffProcess):
        logger.info(f"Hacking CFGDenoiser forward function.")
        self.cfg_original_forward = CFGDenoiser.forward
        cfg_original_forward = self.cfg_original_forward
        def mm_cfg_forward(self, x, sigma, uncond, cond, cond_scale, s_min_uncond, image_cond):
            for context in AnimateDiffInfV2V.uniform(self.step, params.video_length, params.batch_size, params.stride, params.overlap, params.closed_loop):
                x[context] = cfg_original_forward(self, x[context], sigma, uncond[context], cond[context], cond_scale, s_min_uncond, image_cond)
                self.step -= 1
            self.step += 1
            return x
        CFGDenoiser.forward = mm_cfg_forward


    def restore_cfg_forward(self):
        logger.info(f"Restoring CFGDenoiser forward function.")
        CFGDenoiser.forward = self.cfg_original_forward
