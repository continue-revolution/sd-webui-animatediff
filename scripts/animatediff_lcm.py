
# TODO: remove this file when LCM is merged to A1111
import torch

from k_diffusion import utils, sampling
from k_diffusion.external import DiscreteEpsDDPMDenoiser
from k_diffusion.sampling import default_noise_sampler, trange

from modules import shared, sd_samplers_cfg_denoiser, sd_samplers_kdiffusion
from scripts.animatediff_logger import logger_animatediff as logger


class LCMCompVisDenoiser(DiscreteEpsDDPMDenoiser):
    def __init__(self, model):
        timesteps = 1000
        beta_start = 0.00085
        beta_end = 0.012

        betas = torch.linspace(beta_start**0.5, beta_end**0.5, timesteps, dtype=torch.float32) ** 2
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        original_timesteps = 50     # LCM Original Timesteps (default=50, for current version of LCM)
        self.skip_steps = timesteps // original_timesteps


        alphas_cumprod_valid = torch.zeros((original_timesteps), dtype=torch.float32, device=model.device)
        for x in range(original_timesteps):
            alphas_cumprod_valid[original_timesteps - 1 - x] = alphas_cumprod[timesteps - 1 - x * self.skip_steps]

        super().__init__(model, alphas_cumprod_valid, quantize=None)


    def get_sigma(self, n=None, sgm=False):
        if n is None:
            return sampling.append_zero(self.sigmas.flip(0))

        start = self.sigma_to_t(self.sigma_max)
        end = self.sigma_to_t(self.sigma_min)

        if sgm:
            t = torch.linspace(start, end, n + 1)[:-1]
        else:
            t = torch.linspace(start, end, n)

        return sampling.append_zero(self.t_to_sigma(t))


    def sigma_to_t(self, sigma, quantize=None):
        log_sigma = sigma.log()
        dists = log_sigma - self.log_sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape) * self.skip_steps + (self.skip_steps - 1)


    def t_to_sigma(self, timestep):
        t = torch.clamp(((timestep - (self.skip_steps - 1)) / self.skip_steps).float(), min=0, max=(len(self.sigmas) - 1))
        return super().t_to_sigma(t)


    def get_eps(self, *args, **kwargs):
        return self.inner_model.apply_model(*args, **kwargs)


    def get_c_in(self, sigma):
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_in


    def get_scaled_out(self, sigma, model_output, model_input):
        x0 = model_input - model_output * utils.append_dims(sigma, model_output.ndim)

        sigma_data = 0.5
        scaled_timestep = utils.append_dims(self.sigma_to_t(sigma), model_output.ndim) * 10.0

        c_skip = sigma_data**2 / (scaled_timestep**2 + sigma_data**2)
        c_out = scaled_timestep / (scaled_timestep**2 + sigma_data**2) ** 0.5

        return c_out * x0 + c_skip * model_input


    def forward(self, input, sigma, **kwargs):
        c_in = utils.append_dims(self.get_c_in(sigma), input.ndim)
        eps = self.get_eps(input * c_in, self.sigma_to_t(sigma), **kwargs)
        return self.get_scaled_out(sigma, eps, input * c_in)


def sample_lcm(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        x = denoised
        if sigmas[i + 1] > 0:
            x += sigmas[i + 1] * noise_sampler(sigmas[i], sigmas[i + 1])
    return x


class CFGDenoiserLCM(sd_samplers_cfg_denoiser.CFGDenoiser):
    @property
    def inner_model(self):
        if self.model_wrap is None:
            denoiser = LCMCompVisDenoiser
            self.model_wrap = denoiser(shared.sd_model)

        return self.model_wrap


class LCMSampler(sd_samplers_kdiffusion.KDiffusionSampler):
    def __init__(self, funcname, sd_model, options=None):
        super().__init__(funcname, sd_model, options)
        self.model_wrap_cfg = CFGDenoiserLCM(self)
        self.model_wrap = self.model_wrap_cfg.inner_model


class AnimateDiffLCM:
    lcm_ui_injected = False


    @staticmethod
    def hack_kdiff_ui():
        if AnimateDiffLCM.lcm_ui_injected:
            logger.info(f"LCM UI already injected.")
            return

        logger.info(f"Injecting LCM to UI.")
        from modules import sd_samplers, sd_samplers_common
        samplers_lcm = [('LCM', sample_lcm, ['k_lcm'], {})]
        samplers_data_lcm = [
            sd_samplers_common.SamplerData(label, lambda model, funcname=funcname: LCMSampler(funcname, model), aliases, options)
            for label, funcname, aliases, options in samplers_lcm
        ]
        sd_samplers.all_samplers.extend(samplers_data_lcm)
        sd_samplers.all_samplers_map = {x.name: x for x in sd_samplers.all_samplers}
        sd_samplers.set_samplers()
        AnimateDiffLCM.lcm_ui_injected = True
