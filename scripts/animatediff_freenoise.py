import torch
import torch.fft as fft
import math
import random


import os
import re
import sys

from modules import sd_models, shared, sd_samplers, devices
from modules.paths import extensions_builtin_dir
from modules.processing import StableDiffusionProcessing, opt_C, opt_f, StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img
from types import MethodType

from scripts.animatediff_logger import logger_animatediff as logger
from scripts.animatediff_ui import AnimateDiffProcess



def ddim_add_noise(
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:

        alphas_cumprod = shared.sd_model.alphas_cumprod
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        alphas_cumprod = alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples



class AnimateDiffFreeNoise:
    def __init__(self, params):
        pass


    def hack(self, p: StableDiffusionProcessing, params: AnimateDiffProcess):
        
        # set model to window attention
        for name, module in p.sd_model.named_modules():
            if name.endswith('temporal_transformer'):
                for temporal_transformer_block in module.transformer_blocks:
                    temporal_transformer_block.local_window = True


        def sample_t2i(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
            self.sampler = sd_samplers.create_sampler(self.sampler_name, self.sd_model)

            # Sampling with FreeNoise
            x = self.rng.next()
            x_dtype = x.dtype

            window_size = 16
            window_stride = 4
            video_length = x.shape[0]
            for frame_index in range(window_size, video_length, window_stride):
                list_index = list(range(frame_index-window_size, frame_index+window_stride-window_size))
                random.shuffle(list_index)  #TODO may reset random process
                x[frame_index:frame_index+window_stride] = x[list_index]
        
            x = self.sampler.sample(self, x, conditioning, unconditional_conditioning, image_conditioning=self.txt2img_image_conditioning(x))
            samples = x
            del x

            if not self.enable_hr:
                return samples

            if self.latent_scale_mode is None:
                decoded_samples = torch.stack(decode_latent_batch(self.sd_model, samples, target_device=devices.cpu, check_for_nans=True)).to(dtype=torch.float32)
            else:
                decoded_samples = None

            with sd_models.SkipWritingToConfig():
                sd_models.reload_model_weights(info=self.hr_checkpoint_info)

            devices.torch_gc()

            return self.sample_hr_pass(samples, decoded_samples, seeds, subseeds, subseed_strength, prompts)

        if isinstance(p, StableDiffusionProcessingTxt2Img):
            p.sample = MethodType(sample_t2i, p)


    def restore(self, p: StableDiffusionProcessing):
        # set model to window attention
        for name, module in p.sd_model.named_modules():
            if name.endswith('temporal_transformer'):
                for temporal_transformer_block in module.transformer_blocks:
                    temporal_transformer_block.local_window = False
        print("free noise restore")
