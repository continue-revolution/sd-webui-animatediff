import torch
import torch.fft as fft
import math

from modules import sd_models, shared, sd_samplers, devices
from modules.processing import StableDiffusionProcessing, opt_C, opt_f, StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img, decode_latent_batch
from types import MethodType

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



class AnimateDiffFreeInit:
    def __init__(self, params):
        self.num_iters = params.freeinit_iters
        self.method = params.freeinit_filter
        self.d_s = params.freeinit_ds
        self.d_t = params.freeinit_dt

    @torch.no_grad()
    def init_filter(self, video_length, height, width, filter_params):
        # initialize frequency filter for noise reinitialization
        batch_size = 1
        filter_shape = [
            batch_size, 
            opt_C, 
            video_length, 
            height // opt_f, 
            width // opt_f
        ]
        self.freq_filter = get_freq_filter(filter_shape, device=devices.device, params=filter_params)


    def hack(self, p: StableDiffusionProcessing, params: AnimateDiffProcess):
        # init filter
        filter_params = { 
            'method': self.method,
            'd_s': self.d_s,
            'd_t': self.d_t,
        }
        self.init_filter(params.video_length, p.height, p.width, filter_params)


        def sample_t2i(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
            self.sampler = sd_samplers.create_sampler(self.sampler_name, self.sd_model)

            # hack total progress bar （works in an ugly way)
            setattr(self.sampler, 'freeinit_num_iters', self.num_freeinit_iters)  
            setattr(self.sampler, 'freeinit_num_iter', 0)  

            def callback_hack(self, d):
                step = d['i'] // self.freeinit_num_iters + self.freeinit_num_iter * (shared.state.sampling_steps // self.freeinit_num_iters)

                if self.stop_at is not None and step > self.stop_at:
                    raise InterruptedException

                shared.state.sampling_step = step

                if d['i'] % self.freeinit_num_iters == 0:
                    shared.total_tqdm.update()

            self.sampler.callback_state = MethodType(callback_hack, self.sampler) 

            # Sampling with FreeInit
            x = self.rng.next()
            x_dtype = x.dtype
 
            for iter in range(self.num_freeinit_iters):
                self.sampler.freeinit_num_iter = iter
                if iter == 0:
                    initial_x = x.detach().clone()
                else:
                    # z_0
                    diffuse_timesteps = torch.tensor(1000 - 1)
                    z_T = ddim_add_noise(x, initial_x, diffuse_timesteps)   # [16, 4, 64, 64]
                    # z_T
                    # 2. create random noise z_rand for high-frequency
                    z_T = z_T.permute(1, 0, 2, 3)[None, ...]    # [bs, 4, 16, 64, 64]
                    #z_rand = torch.randn(z_T.shape, device=devices.device)
                    z_rand = initial_x.detach().clone().permute(1, 0, 2, 3)[None, ...]
                    # 3. Roise Reinitialization
                    x = freq_mix_3d(z_T.to(dtype=torch.float32), z_rand, LPF=self.freq_filter)
                    
                    x = x[0].permute(1, 0, 2, 3)
                    x = x.to(x_dtype)

                x = self.sampler.sample(self, x, conditioning, unconditional_conditioning, image_conditioning=self.txt2img_image_conditioning(x))
                devices.torch_gc()

            samples = x
            del x

            if not self.enable_hr:
                return samples

            devices.torch_gc()

            if self.latent_scale_mode is None:
                decoded_samples = torch.stack(decode_latent_batch(self.sd_model, samples, target_device=devices.cpu, check_for_nans=True)).to(dtype=torch.float32)
            else:
                decoded_samples = None

            with sd_models.SkipWritingToConfig():
                sd_models.reload_model_weights(info=self.hr_checkpoint_info)

            return self.sample_hr_pass(samples, decoded_samples, seeds, subseeds, subseed_strength, prompts)


        def sample_i2i(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
            x = self.rng.next()
            x_dtype = x.dtype
 

            if self.initial_noise_multiplier != 1.0:
                self.extra_generation_params["Noise multiplier"] = self.initial_noise_multiplier
                x *= self.initial_noise_multiplier

            for iter in range(self.num_freeinit_iters):
                if iter == 0:
                    initial_x = x.detach().clone()
                else:
                    # z_0
                    diffuse_timesteps = torch.tensor(1000 - 1)
                    z_T = ddim_add_noise(x, initial_x, diffuse_timesteps)   # [16, 4, 64, 64]
                    # z_T
                    # 2. create random noise z_rand for high-frequency
                    z_T = z_T.permute(1, 0, 2, 3)[None, ...]    # [bs, 4, 16, 64, 64]
                    #z_rand = torch.randn(z_T.shape, device=devices.device)
                    z_rand = initial_x.detach().clone().permute(1, 0, 2, 3)[None, ...]
                    # 3. Roise Reinitialization
                    x = freq_mix_3d(z_T.to(dtype=torch.float32), z_rand, LPF=self.freq_filter)
                    
                    x = x[0].permute(1, 0, 2, 3)
                    x = x.to(x_dtype)

                x = self.sampler.sample_img2img(self, self.init_latent, x, conditioning, unconditional_conditioning, image_conditioning=self.image_conditioning)
            samples = x

            if self.mask is not None:
                samples = samples * self.nmask + self.init_latent * self.mask

            del x
            devices.torch_gc()

            return samples

        if isinstance(p, StableDiffusionProcessingTxt2Img):
            p.sample = MethodType(sample_t2i, p)
        elif isinstance(p, StableDiffusionProcessingImg2Img):
            p.sample = MethodType(sample_i2i, p)
        else:
            raise NotImplementedError

        setattr(p, 'freq_filter', self.freq_filter)  
        setattr(p, 'num_freeinit_iters', self.num_iters)  


def freq_mix_3d(x, noise, LPF):
    """
    Noise reinitialization.

    Args:
        x: diffused latent
        noise: randomly sampled noise
        LPF: low pass filter
    """
    # FFT
    x_freq = fft.fftn(x, dim=(-3, -2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-3, -2, -1))
    noise_freq = fft.fftn(noise, dim=(-3, -2, -1))
    noise_freq = fft.fftshift(noise_freq, dim=(-3, -2, -1))

    # frequency mix
    HPF = 1 - LPF
    x_freq_low = x_freq * LPF
    noise_freq_high = noise_freq * HPF
    x_freq_mixed = x_freq_low + noise_freq_high # mix in freq domain

    # IFFT
    x_freq_mixed = fft.ifftshift(x_freq_mixed, dim=(-3, -2, -1))
    x_mixed = fft.ifftn(x_freq_mixed, dim=(-3, -2, -1)).real

    return x_mixed


def get_freq_filter(shape, device, params: dict):
    """
    Form the frequency filter for noise reinitialization.

    Args:
        shape: shape of latent (B, C, T, H, W)
        params: filter parameters
    """
    if params['method'] == "gaussian":
        return gaussian_low_pass_filter(shape=shape, d_s=params['d_s'], d_t=params['d_t']).to(device)
    elif params['method'] == "ideal":
        return ideal_low_pass_filter(shape=shape, d_s=params['d_s'], d_t=params['d_t']).to(device)
    elif params['method'] == "box":
        return box_low_pass_filter(shape=shape, d_s=params['d_s'], d_t=params['d_t']).to(device)
    elif params['method'] == "butterworth":
        return butterworth_low_pass_filter(shape=shape, n=4, d_s=params['d_s'], d_t=params['d_t']).to(device)
    else:
        raise NotImplementedError

def gaussian_low_pass_filter(shape, d_s=0.25, d_t=0.25):
    """
    Compute the gaussian low pass filter mask.

    Args:
        shape: shape of the filter (volume)
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    T, H, W = shape[-3], shape[-2], shape[-1]
    mask = torch.zeros(shape)
    if d_s==0 or d_t==0:
        return mask
    for t in range(T):
        for h in range(H):
            for w in range(W):
                d_square = (((d_s/d_t)*(2*t/T-1))**2 + (2*h/H-1)**2 + (2*w/W-1)**2)
                mask[..., t,h,w] = math.exp(-1/(2*d_s**2) * d_square)
    return mask


def butterworth_low_pass_filter(shape, n=4, d_s=0.25, d_t=0.25):
    """
    Compute the butterworth low pass filter mask.

    Args:
        shape: shape of the filter (volume)
        n: order of the filter, larger n ~ ideal, smaller n ~ gaussian
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    T, H, W = shape[-3], shape[-2], shape[-1]
    mask = torch.zeros(shape)
    if d_s==0 or d_t==0:
        return mask
    for t in range(T):
        for h in range(H):
            for w in range(W):
                d_square = (((d_s/d_t)*(2*t/T-1))**2 + (2*h/H-1)**2 + (2*w/W-1)**2)
                mask[..., t,h,w] = 1 / (1 + (d_square / d_s**2)**n)
    return mask


def ideal_low_pass_filter(shape, d_s=0.25, d_t=0.25):
    """
    Compute the ideal low pass filter mask.

    Args:
        shape: shape of the filter (volume)
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    T, H, W = shape[-3], shape[-2], shape[-1]
    mask = torch.zeros(shape)
    if d_s==0 or d_t==0:
        return mask
    for t in range(T):
        for h in range(H):
            for w in range(W):
                d_square = (((d_s/d_t)*(2*t/T-1))**2 + (2*h/H-1)**2 + (2*w/W-1)**2)
                mask[..., t,h,w] =  1 if d_square <= d_s*2 else 0
    return mask


def box_low_pass_filter(shape, d_s=0.25, d_t=0.25):
    """
    Compute the ideal low pass filter mask (approximated version).

    Args:
        shape: shape of the filter (volume)
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    T, H, W = shape[-3], shape[-2], shape[-1]
    mask = torch.zeros(shape)
    if d_s==0 or d_t==0:
        return mask

    threshold_s = round(int(H // 2) * d_s)
    threshold_t = round(T // 2 * d_t)

    cframe, crow, ccol = T // 2, H // 2, W //2
    mask[..., cframe - threshold_t:cframe + threshold_t, crow - threshold_s:crow + threshold_s, ccol - threshold_s:ccol + threshold_s] = 1.0

    return mask
