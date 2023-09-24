import numpy as np
from typing import Optional


class AnimateDiffInfV2V:

    def __init__(self):
        pass

    @staticmethod
    # Returns fraction that has denominator that is a power of 2
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
    def uniform(
        step: int = ...,
        video_length: int = ...,
        batch_size: Optional[int] = None,
        context_stride: int = 3,
        context_overlap: int = 4,
        closed_loop: bool = True,
    ):
        if video_length <= batch_size:
            yield list(range(video_length))
            return

        context_stride = min(context_stride, int(np.ceil(np.log2(video_length / batch_size))) + 1)

        for context_step in 1 << np.arange(context_stride):
            pad = int(round(video_length * AnimateDiffInfV2V.ordered_halving(step)))
            for j in range(
                int(AnimateDiffInfV2V.ordered_halving(step) * context_step) + pad,
                video_length + pad + (0 if closed_loop else -context_overlap),
                (batch_size * context_step - context_overlap),
            ):
                yield [e % video_length for e in range(j, j + batch_size * context_step, context_step)]


    def uniform_constant(
        step: int = ...,
        num_steps: Optional[int] = None,
        num_frames: int = ...,
        context_size: Optional[int] = None,
        context_stride: int = 3,
        context_overlap: int = 4,
        closed_loop: bool = True,
        print_final: bool = False,
    ):
        if num_frames <= context_size:
            yield list(range(num_frames))
            return

        context_stride = min(context_stride, int(np.ceil(np.log2(num_frames / context_size))) + 1)

        # want to avoid loops that connect end to beginning

        for context_step in 1 << np.arange(context_stride):
            pad = int(round(num_frames * AnimateDiffInfV2V.ordered_halving(step, print_final)))
            for j in range(
                int(AnimateDiffInfV2V.ordered_halving(step) * context_step) + pad,
                num_frames + pad + (0 if closed_loop else -context_overlap),
                (context_size * context_step - context_overlap),
            ):
                skip_this_window = False
                prev_val = -1
                to_yield = []
                for e in range(j, j + context_size * context_step, context_step):
                    e = e % num_frames
                    # if not a closed loop and loops back on itself, should be skipped
                    if not closed_loop and e < prev_val:
                        skip_this_window = True
                        break
                    to_yield.append(e)
                    prev_val = e
                if skip_this_window:
                    continue
                # yield if not skipped
                yield to_yield