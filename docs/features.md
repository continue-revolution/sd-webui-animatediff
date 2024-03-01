# Features

## Img2Vid
> I believe that there are better ways to do i2v. New methods will be implemented soon and this old and unstable way might be subject to removal.

You need to go to img2img and submit an init frame via A1111 panel. You can optionally submit a last frame via extension panel.

By default: your `init_latent` will be changed to 
```
init_alpha = (1 - frame_number ^ latent_power / latent_scale)
init_latent = init_latent * init_alpha + random_tensor * (1 - init_alpha)
```

If you upload a last frame: your `init_latent` will be changed in a similar way. Read [this code](https://github.com/continue-revolution/sd-webui-animatediff/tree/v1.5.0/scripts/animatediff_latent.py#L28-L65) to understand how it works.


## Prompt Travel

Write positive prompt following the example below.

The first line is head prompt, which is optional. You can write no/single/multiple lines of head prompts.

All following lines in format `frame number`: `prompt` are for prompt interpolation. Your `frame number` should be in ascending order, smaller than the total `Number of frames`. The first frame is 0 index.

The last line is tail prompt, which is optional. You can write no/single/multiple lines of tail prompts. If you don't need this feature, just write prompts in the old way.
```
1girl, yoimiya (genshin impact), origen, line, comet, wink, Masterpiece, BestQuality. UltraDetailed, <lora:LineLine2D:0.7>,  <lora:yoimiya:0.8>, 
0: closed mouth
8: open mouth
smile
```


## ControlNet V2V
> TODO: keyframe and mask demo, including ControlNet / IP-Adapter / SparseCtrl. We also need a better demo.

You need to go to txt2img / img2img-batch and submit source video or path to frames. Each ControlNet will find control images according to this priority:
1. ControlNet `Single Image` tab or `Batch Folder` tab. Simply upload a control image or a path to folder of control frames is enough.
1. Img2img Batch tab `Input directory` if you are using img2img batch. If you upload a directory of control frames, it will be the source control for ALL ControlNet units that you enable without submitting a control image or a path to ControlNet panel.
1. AnimateDiff `Video Path`. If you upload a path to frames through `Video Path`, it will be the source control for ALL ControlNet units that you enable without submitting a control image or a path to ControlNet panel.
1. AnimateDiff `Video Source`. If you upload a video through `Video Source`, it will be the source control for ALL ControlNet units that you enable without submitting a control image or a path to ControlNet panel.

`Number of frames` will be capped to the minimum number of images among all **folders** you provide. Each control image in each folder will be applied to one single frame. If you upload one single image for a ControlNet unit, that image will control **ALL** frames.

Example input parameter fill-in:
1. Specify different `Input Directory` and `Mask Directory` for different individual ControlNet Units.
![cn1](https://github.com/continue-revolution/sd-webui-animatediff/assets/63914308/c1e0e04d-d2c7-4150-89e1-3befe2f3c750)
1. Specify a global `Videl path` and `Mask path` and leave ControlNet Unit `Input Directory` input blank.
    - You can arbitratily change ControlNet Unit tab to `Single Image` / `Batch Folder` / `Batch Upload` as long as you leave it blank.
    - If you specify a global mask path, all ControlNet Units that you do not give a `Mask Directory` will use this path.
    - Please only have one of `Video source` and `Video path`. They cannot be applied at the same time.
![cn2](https://github.com/continue-revolution/sd-webui-animatediff/assets/63914308/dc8d71df-60ea-4dd9-a040-b7bd35161587)

There are a lot of amazing demo online. Here I provide a very simple demo. The dataset is from [streamline](https://twitter.com/kaizirod), but the workflow is an arbitrary setup by me. You can find a lot more much more amazing examples (and potentially available workflows / infotexts) on Reddit, Twitter, YouTube and Bilibili. The easiest way to share your workflow created by my software is to share one output frame with infotext.
| input | output |
| --- | --- |
| <img height='512px' src='https://github.com/continue-revolution/sd-webui-animatediff/assets/63914308/ff066808-fc00-43e1-a2a6-b16e41dad603'> | <img height='512px' src='https://github.com/continue-revolution/sd-webui-animatediff/assets/63914308/dc3d833b-a113-4278-9e48-5f2a8ee06704'> |


## Model Spec
> BREAKING CHANGE: You need to use Motion LoRA, HotShot-XL and V3 Motion Adapter from my HuggingFace repository instead of the original one.

### Motion LoRA
[Download](https://huggingface.co/conrevo/AnimateDiff-A1111/tree/main/lora) and use them like any other LoRA you use (example: download Motion LoRA to `stable-diffusion-webui/models/Lora` and add `<lora:mm_sd15_v2_lora_PanLeft:0.8>` to your positive prompt). **Motion LoRAs cam only be applied to V2 motion modules**.

### V3
V3 has identical state dict keys as V1 but slightly different inference logic (GroupNorm is not hacked for V3). You may optionally use [adapter](https://huggingface.co/conrevo/AnimateDiff-A1111/resolve/main/lora/mm_sd15_v3_adapter.safetensors?download=true) for V3, in the same way as how you apply LoRA. You MUST use [my link](https://huggingface.co/conrevo/AnimateDiff-A1111/resolve/main/lora/mm_sd15_v3_adapter.safetensors?download=true) instead of the [official link](https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_adapter.ckpt?download=true). The official adapter won't work for A1111 due to state dict incompatibility.

### SDXL
[AnimateDiff-XL](https://github.com/guoyww/AnimateDiff/tree/sdxl) and [HotShot-XL](https://github.com/hotshotco/Hotshot-XL) have identical architecture to AnimateDiff-SD1.5. The only difference are
- HotShot-XL is trained with 8 frames instead of 16 frames. You are recommended to set `Context batch size` to 8 for HotShot-XL.
- AnimateDiff-XL is still trained with 16 frames. You do not need to change `Context batch size` for AnimateDiffXL.
- AnimateDiff-XL & HotShot-XL have fewer layers compared to AnimateDiff-SD1.5 because of SDXL.
- AnimateDiff-XL is trained with higher resolution compared to HotShot-XL.

Although AnimateDiff-XL & HotShot-XL have identical structure as AnimateDiff-SD1.5, I strongly discourage you from using AnimateDiff-SD1.5 for SDXL, or using HotShot-XL / AnimateDiff-XL for SD1.5 - you will get severe artifect if you do that. I have decided not to supported that, despite the fact that it is not hard for me to do that.

Technically all features available for AnimateDiff + SD1.5 are also available for (AnimateDiff / HotShot) + SDXL. However, I have not tested all of them. I have tested infinite context generation and prompt travel; I have not tested ControlNet. If you find any bug, please report it to me.

Unfortunately, neither of these 2 motion modules are as good as those for SD1.5, and there is NOTHING I can do about it (they are just poorly trained). I strongly discourage anyone from applying SDXL for video generation. You will be VERY disappointed if you do that.
