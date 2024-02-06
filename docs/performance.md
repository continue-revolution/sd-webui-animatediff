# Performance

## Optimizations
> Information about running ControlNet V2V with the following optimizations will be updated soon. Forge has a lot more optimizations

Optimizations can be significantly helpful if you want to improve speed and reduce VRAM usage.

### Attention
In forge, we will always use scaled dot product attention from PyTorch.

### FP8
FP8 requires torch >= 2.1.0. Add `--unet-in-fp8-e4m3fn` to command line arguments if you want fp8.

### LCM
[Latent Consistency Model](https://github.com/luosiallen/latent-consistency-model) is a recent breakthrough in Stable Diffusion community. You can generate images / videos within 6-8 steps if you
- select `Euler A` / `Euler` / `LCM` sampler (other samplers may also work, subject to further experiments)
- use [LCM LoRA](https://civitai.com/models/195519/lcm-lora-weights-stable-diffusion-acceleration-module)
- use a low CFG denoising strength (1-2 is recommended)


## VRAM
> These are for OG A1111. Information about forge will be updated soon.

Actual VRAM usage depends on your image size and context batch size. You can try to reduce image size or context batch size to reduce VRAM usage. 

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

We are currently developing approach to support batch size on WebUI in the near future.
