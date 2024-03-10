import torch
import torch.fft as fft
import math
import random


import os
import re
import sys

from modules import sd_models, sd_samplers, devices
from modules.processing import StableDiffusionProcessing, StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img
from types import MethodType

from scripts.animatediff_logger import logger_animatediff as logger
from scripts.animatediff_ui import AnimateDiffProcess




class AnimateDiffFreeNoise:
    def __init__(self, params):
        self.window_size = params.freenoise_window_size
        self.window_stride = params.freenoise_window_stride


    def hack(self, p: StableDiffusionProcessing, params: AnimateDiffProcess):
        # set model to window attention
        for name, module in p.sd_model.named_modules():
            if name.endswith('temporal_transformer'):
                for temporal_transformer_block in module.transformer_blocks:
                    temporal_transformer_block.local_window = True

        setattr(p, 'window_size', self.window_size)  
        setattr(p, 'window_stride', self.window_stride)  

        def sample_t2i(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
            self.sampler = sd_samplers.create_sampler(self.sampler_name, self.sd_model)

            # Sampling with FreeNoise
            x = self.rng.next()
            x_dtype = x.dtype
            window_size = int(self.window_size)
            window_stride = int(self.window_stride)
            video_length = x.shape[0]
            for frame_index in range(window_size, video_length, window_stride):
                list_index = list(range(frame_index-window_size, frame_index+window_stride-window_size))
                random.shuffle(list_index) 
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



        def sample_i2i(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
            x = self.rng.next()

            # Sampling with FreeNoise
            x_dtype = x.dtype
            window_size = self.window_size
            window_stride = self.window_stride
            video_length = x.shape[0]
            for frame_index in range(window_size, video_length, window_stride):
                list_index = list(range(frame_index-window_size, frame_index+window_stride-window_size))
                random.shuffle(list_index) 
                x[frame_index:frame_index+window_stride] = x[list_index]


            if self.initial_noise_multiplier != 1.0:
                self.extra_generation_params["Noise multiplier"] = self.initial_noise_multiplier
                x *= self.initial_noise_multiplier

            samples = self.sampler.sample_img2img(self, self.init_latent, x, conditioning, unconditional_conditioning, image_conditioning=self.image_conditioning)

            if self.mask is not None:
                blended_samples = samples * self.nmask + self.init_latent * self.mask

                if self.scripts is not None:
                    mba = scripts.MaskBlendArgs(samples, self.nmask, self.init_latent, self.mask, blended_samples)
                    self.scripts.on_mask_blend(self, mba)
                    blended_samples = mba.blended_latent

                samples = blended_samples

            del x
            devices.torch_gc()

            return samples

        if isinstance(p, StableDiffusionProcessingTxt2Img):
            p.sample = MethodType(sample_t2i, p)
        elif isinstance(p, StableDiffusionProcessingImg2Img):
            p.sample = MethodType(sample_i2i, p)
        else:
            raise NotImplementedError



    def restore(self, p: StableDiffusionProcessing):
        # set model to window attention
        for name, module in p.sd_model.named_modules():
            if name.endswith('temporal_transformer'):
                for temporal_transformer_block in module.transformer_blocks:
                    temporal_transformer_block.local_window = False
