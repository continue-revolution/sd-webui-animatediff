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
> This has not been ready for now in forge.

You need to go to txt2img / img2img-batch and submit source video or path to frames. Each ControlNet will find control images according to this priority:
1. ControlNet `Single Image` tab or `Batch` tab. Simply upload a control image or a directory of control frames is enough.
1. Img2img Batch tab `Input directory` if you are using img2img batch. If you upload a directory of control frames, it will be the source control for ALL ControlNet units that you enable without submitting a control image or a path to ControlNet panel.
1. AnimateDiff `Video Source`. If you upload a video through `Video Source`, it will be the source control for ALL ControlNet units that you enable without submitting a control image or a path to ControlNet panel.
1. AnimateDiff `Video Path`. If you upload a path to frames through `Video Path`, it will be the source control for ALL ControlNet units that you enable without submitting a control image or a path to ControlNet panel.

`Number of frames` will be capped to the minimum number of images among all **folders** you provide. Each control image in each folder will be applied to one single frame. If you upload one single image for a ControlNet unit, that image will control **ALL** frames.

For people who want to inpaint videos: enter a folder which contains two sub-folders `image` and `mask` on ControlNet inpainting unit. These two sub-folders should contain the same number of images. This extension will match them according to the same sequence. Using my [Segment Anything](https://github.com/continue-revolution/sd-webui-segment-anything) extension can make your life much easier.


## Model Spec

### Motion LoRA
[Download](https://huggingface.co/conrevo/AnimateDiff-A1111/tree/main/lora) and use them like any other LoRA you use (example: download motion lora to `stable-diffusion-webui/models/Lora` and add `<lora:mm_sd15_v2_lora_PanLeft:0.8>` to your positive prompt). **Motion LoRA only supports V2 motion modules**.

### V3
V3 has identical state dict keys as V1 but slightly different inference logic (GroupNorm is not hacked for V3). You may optionally use [adapter](https://huggingface.co/conrevo/AnimateDiff-A1111/resolve/main/lora/mm_sd15_v3_adapter.safetensors?download=true) for V3, in the same way as the way you use LoRA. You MUST use [my link](https://huggingface.co/conrevo/AnimateDiff-A1111/resolve/main/lora/mm_sd15_v3_adapter.safetensors?download=true) instead of the [official link](https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_adapter.ckpt?download=true). The official adapter won't work for A1111 due to state dict incompatibility.

### SDXL
[AnimateDiffXL](https://github.com/guoyww/AnimateDiff/tree/sdxl) and [HotShot-XL](https://github.com/hotshotco/Hotshot-XL) have identical architecture to AnimateDiff-SD1.5. The only 2 difference are
- HotShot-XL is trained with 8 frames instead of 16 frames. You are recommended to set `Context batch size` to 8 for HotShot-XL.
- AnimateDiffXL is still trained with 16 frames. You do not need to change `Context batch size` for AnimateDiffXL.
- AnimateDiffXL & HotShot-XL have fewer layers compared to AnimateDiff-SD1.5 because of SDXL.
- AnimateDiffXL is trained with higher resolution compared to HotShot-XL.

Although AnimateDiffXL & HotShot-XL have identical structure with AnimateDiff-SD1.5, I strongly discourage you from using AnimateDiff-SD1.5 for SDXL, or using HotShot / AnimateDiffXL for SD1.5 - you will get severe artifect if you do that. I have decided not to supported that, despite the fact that it is not hard for me to do that.

Technically all features available for AnimateDiff + SD1.5 are also available for (AnimateDiff / HotShot) + SDXL. However, I have not tested all of them. I have tested infinite context generation and prompt travel; I have not tested ControlNet. If you find any bug, please report it to me.