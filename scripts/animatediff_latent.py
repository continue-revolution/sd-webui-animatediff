import torch
from modules import images, shared
from modules.devices import device, dtype_vae, torch_gc
from modules.processing import StableDiffusionProcessingImg2Img
from modules.sd_samplers_common import (approximation_indexes,
                                        images_tensor_to_samples)

from scripts.animatediff_logger import logger_animatediff as logger
from scripts.animatediff_ui import AnimateDiffProcess


class AnimateDiffI2VLatent:
    def randomize(
        self, p: StableDiffusionProcessingImg2Img, params: AnimateDiffProcess
    ):
        # Get init_alpha
        init_alpha = [
            1 - pow(i, params.latent_power) / params.latent_scale
            for i in range(params.video_length)
        ]
        logger.info(f"Randomizing init_latent according to {init_alpha}.")
        init_alpha = torch.tensor(init_alpha, dtype=torch.float32, device=device)[
            :, None, None, None
        ]
        init_alpha[init_alpha < 0] = 0

        if params.last_frame is not None:
            # Get last_alpha
            last_alpha = [
                1 - pow(i, params.latent_power_last) / params.latent_scale_last
                for i in range(params.last_frame)
            ]
            last_alpha.reverse()
            logger.info(f"Randomizing last_latent according to {init_alpha}.")
            last_alpha = torch.tensor(last_alpha, dtype=torch.float32, device=device)[
                :, None, None, None
            ]
            last_alpha[last_alpha < 0] = 0

            # Normalize alpha
            sum_alpha = init_alpha + last_alpha
            mask_alpha = sum_alpha > 1
            scaling_factor = 1 / sum_alpha[mask_alpha]
            init_alpha[mask_alpha] *= scaling_factor
            last_alpha[mask_alpha] *= scaling_factor

            # Calculate last_latent
            last_frame = params.last_frame
            if p.resize_mode != 3:
                last_frame = images.resize_image(
                    p.resize_mode, last_frame, p.width, p.height
                )[None, ...]
            last_frame = torch.from_numpy(last_frame).to(device).to(dtype_vae)
            last_latent = images_tensor_to_samples(
                last_frame,
                approximation_indexes.get(shared.opts.sd_vae_encode_method),
                p.sd_model,
            )
            torch_gc()
            if p.resize_mode == 3:
                opt_f = 8
                last_latent = torch.nn.functional.interpolate(
                    last_latent,
                    size=(p.height // opt_f, p.width // opt_f),
                    mode="bilinear",
                )

            # Modify init_latent
            p.init_latent = (
                p.init_latent * init_alpha
                + p.last_latent * last_alpha
                + p.rng.next() * (1 - init_alpha - last_alpha)
            )
        else:
            p.init_latent = p.init_latent * init_alpha + p.rng.next() * (1 - init_alpha)
