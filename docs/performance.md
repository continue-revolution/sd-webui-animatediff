# Performance

## Optimizations

Optimizations can be significantly helpful if you want to improve speed and reduce VRAM usage.

### Attention
We will always apply scaled dot product attention from PyTorch.

### FP8
FP8 requires torch >= 2.1.0. Go to `Settings/Optimizations` and select `Enable` for `FP8 weight`. Don't forget to click `Apply settings` button.

### LCM
[Latent Consistency Model](https://github.com/luosiallen/latent-consistency-model) is a recent breakthrough in Stable Diffusion community. You can generate images / videos within 6-8 steps if you
- select `LCM` / `Euler A` / `Euler` / `DDIM` sampler
- apply [LCM LoRA](https://civitai.com/models/195519/lcm-lora-weights-stable-diffusion-acceleration-module)
- apply low CFG denoising strength (1-2 is recommended)

I have [PR-ed](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/14583) this sampler to Stable Diffusion WebUI and you no longer need this extension to have LCM sampler. I have removed LCM sampler in this repository.


## VRAM
Actual VRAM usage depends on your image size and context batch size. You can try to reduce image size to reduce VRAM usage. You are discouraged from changing context batch size, because this conflicts training specification.

The following data are SD1.5 + AnimateDiff, tested on Ubuntu 22.04, NVIDIA 4090, torch 2.0.1+cu117, H=W=512, frame=16 (default setting). `w/`/`w/o` means `Batch cond/uncond` in `Settings/Optimization` is checked/unchecked.
| Optimization | VRAM w/ | VRAM w/o |
| --- | --- | --- |
| No optimization | 12.13GB |  |
| xformers/sdp | 5.60GB | 4.21GB |
| sub-quadratic | 10.39GB |  |

For SDXL + HotShot + SDP, tested on Ubuntu 22.04, NVIDIA 4090, torch 2.0.1+cu117, H=W=512, frame=8 (default setting), you need 8.66GB VRAM.

For SDXL + AnimateDiff + SDP, tested on Ubuntu 22.04, NVIDIA 4090, torch 2.0.1+cu117, H=1024, W=768, frame=16, you need 13.87GB VRAM.


## Batch Size 
Batch size on WebUI will be replaced by GIF frame number internally: 1 full GIF generated in 1 batch. If you want to generate multiple GIF at once, please change batch number. 

Batch number is NOT the same as batch size. In A1111 WebUI, batch number is above batch size. Batch number means the number of sequential steps, but batch size means the number of parallel steps. You do not have to worry too much when you increase batch number, but you do need to worry about your VRAM when you increase your batch size (where in this extension, video frame number). You do not need to change batch size at all when you are using this extension.

We might develope approach to support batch size on WebUI, but this is with very low priority and we cannot commit a specific date for this.
