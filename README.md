# AnimateDiff for Stable Diffusion WebUI
This extension aim for integrating [AnimateDiff](https://github.com/guoyww/AnimateDiff/) w/ [CLI](https://github.com/s9roll7/animatediff-cli-prompt-travel) into [AUTOMATIC1111 Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) w/ [ControlNet](https://github.com/Mikubill/sd-webui-controlnet). You can generate GIFs in exactly the same way as generating images after enabling this extension.

This extension implements AnimateDiff in a different way. It does not require you to clone the whole SD1.5 repository. It also applied (probably) the least modification to `ldm`, so that you do not need to reload your model weights if you don't want to.

You might also be interested in another extension I created: [Segment Anything for Stable Diffusion WebUI](https://github.com/continue-revolution/sd-webui-segment-anything).


## Table of Contents
- [Update](#update)
- [How to Use](#how-to-use)
  - [WebUI](#webui)
  - [API](#api)
- [WebUI Parameters](#webui-parameters)
- [Img2GIF](#img2gif)
- [Prompt Travel](#prompt-travel)
- [ControlNet V2V](#controlnet-v2v)
- [Model Spec](#model-spec)
  - [Motion LoRA](#motion-lora)
  - [V3](#v3)
  - [SDXL](#sdxl)
- [Optimizations](#optimizations)
  - [Attention](#attention)
  - [FP8](#fp8)
  - [LCM](#lcm)
  - [Others](#others)
- [Model Zoo](#model-zoo)
- [VRAM](#vram)
- [Batch Size](#batch-size)
- [Demo](#demo)
  - [Basic Usage](#basic-usage)
  - [Motion LoRA](#motion-lora-1)
  - [Prompt Travel](#prompt-travel-1)
  - [AnimateDiff V3](#animatediff-v3)
  - [AnimateDiff SDXL](#animatediff-sdxl)
  - [ControlNet V2V](#controlnet-v2v-1)
- [Tutorial](#tutorial)
- [Thanks](#thanks)
- [Star History](#star-history)
- [Sponsor](#sponsor)


## Update
- `2023/07/20` [v1.1.0](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.1.0): Fix gif duration, add loop number, remove auto-download, remove xformers, remove instructions on gradio UI, refactor README, add [sponsor](#sponsor) QR code.
- `2023/07/24` [v1.2.0](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.2.0): Fix incorrect insertion of motion modules, add option to change path to motion modules in `Settings/AnimateDiff`, fix loading different motion modules.
- `2023/09/04` [v1.3.0](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.3.0): Support any community models with the same architecture; fix grey problem via [#63](https://github.com/continue-revolution/sd-webui-animatediff/issues/63)
- `2023/09/11` [v1.4.0](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.4.0): Support official v2 motion module (different architecture: GroupNorm not hacked, UNet middle layer has motion module).    
- `2023/09/14`: [v1.4.1](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.4.1): Always change `beta`, `alpha_comprod` and `alpha_comprod_prev` to resolve grey problem in other samplers.
- `2023/09/16`: [v1.5.0](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.5.0): Randomize init latent to support [better img2gif](#img2gif); add other output formats and infotext output; add appending reversed frames; refactor code to ease maintaining.
- `2023/09/19`: [v1.5.1](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.5.1): Support xformers, sdp, sub-quadratic attention optimization - [VRAM](#vram) usage decrease to 5.60GB with default setting.
- `2023/09/22`: [v1.5.2](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.5.2): Option to disable xformers at `Settings/AnimateDiff` [due to a bug in xformers](https://github.com/facebookresearch/xformers/issues/845), [API support](#api), option to enable GIF paletter optimization at `Settings/AnimateDiff`, gifsicle optimization move to `Settings/AnimateDiff`.
- `2023/09/25`: [v1.6.0](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.6.0): [Motion LoRA](https://github.com/guoyww/AnimateDiff#features) supported. See [Motion Lora](#motion-lora) for more information.
- `2023/09/27`: [v1.7.0](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.7.0): [ControlNet](https://github.com/Mikubill/sd-webui-controlnet) supported. See [ControlNet V2V](#controlnet-v2v) for more information. [Safetensors](#model-zoo) for some motion modules are also available now.
- `2023/09/29`: [v1.8.0](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.8.0): Infinite generation supported. See [WebUI Parameters](#webui-parameters) for more information.
- `2023/10/01`: [v1.8.1](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.8.1): Now you can uncheck `Batch cond/uncond` in `Settings/Optimization` if you want. This will reduce your [VRAM](#vram) (5.31GB -> 4.21GB for SDP) but take longer time.
- `2023/10/08`: [v1.9.0](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.9.0): Prompt travel supported. You must have ControlNet installed (you do not need to enable ControlNet) to try it. See [Prompt Travel](#prompt-travel) for how to trigger this feature.
- `2023/10/11`: [v1.9.1](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.9.1): Use state_dict key to guess mm version, replace match case with if else to support python<3.10, option to save PNG to custom dir
 (see `Settings/AnimateDiff` for detail), move hints to js, install imageio\[ffmpeg\] automatically when MP4 save fails.
- `2023/10/16`: [v1.9.2](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.9.2): Add context generator to completely remove any closed loop, prompt travel support closed loop, infotext fully supported including prompt travel, README refactor
- `2023/10/19`: [v1.9.3](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.9.3): Support webp output format. See [#233](https://github.com/continue-revolution/sd-webui-animatediff/pull/233) for more information.
- `2023/10/21`: [v1.9.4](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.9.4): Save prompt travel to output images, `Reverse` merged to `Closed loop` (See [WebUI Parameters](#webui-parameters)), remove `TimestepEmbedSequential` hijack, remove `hints.js`, better explanation of several context-related parameters.
- `2023/10/25`: [v1.10.0](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.10.0): Support img2img batch. You need ControlNet installed to make it work properly (you do not need to enable ControlNet). See [ControlNet V2V](#controlnet-v2v) for more information.
- `2023/10/29`: [v1.11.0](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.11.0): [HotShot-XL](https://github.com/hotshotco/Hotshot-XL) supported. See [SDXL](#sdxl) for more information.
- `2023/11/06`: [v1.11.1](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.11.1): Optimize VRAM for ControlNet V2V, patch [encode_pil_to_base64](https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/api/api.py#L104-L133) for api return a video, save frames to `AnimateDiff/yy-mm-dd/`, recover from assertion error, optional [request id](#api) for API.
- `2023/11/10`: [v1.12.0](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.12.0): [AnimateDiff for SDXL](https://github.com/guoyww/AnimateDiff/tree/sdxl) supported. See [SDXL](#sdxl) for more information.
- `2023/11/16`: [v1.12.1](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.12.1): FP8 precision and LCM sampler supported. See [Optimizations](#optimizations) for more information. You can also optionally upload videos to AWS S3 storage by configuring appropriately via `Settings/AnimateDiff AWS`.
- `2023/12/19`: [v1.13.0](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.13.0): [AnimateDiff V3](https://github.com/guoyww/AnimateDiff?tab=readme-ov-file#202312-animatediff-v3-and-sparsectrl) supported. See [V3](#v3) for more information. Also: release all official models in fp16 & safetensors format [here](https://huggingface.co/conrevo/AnimateDiff-A1111/tree/main), add option to disable LCM sampler in `Settings/AnimateDiff`, remove patch [encode_pil_to_base64](https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/api/api.py#L104-L133) because A1111 [v1.7.0](https://github.com/AUTOMATIC1111/stable-diffusion-webui/tree/v1.7.0) now supports video return for API.
- `2024/01/12`: [v1.13.1](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.13.1): This small version update completely comes from the community. We fix mp4 encode error [#402](https://github.com/continue-revolution/sd-webui-animatediff/pull/402), support infotext copy-paste [#400](https://github.com/continue-revolution/sd-webui-animatediff/pull/400), validate prompt travel frame numbers [#401](https://github.com/continue-revolution/sd-webui-animatediff/pull/401).

For `v2` update progress, please query [#381](https://github.com/continue-revolution/sd-webui-animatediff/pull/381). For updates (most likely) later than v2, please query [#366](https://github.com/continue-revolution/sd-webui-animatediff/pull/366). All update checklist is tentative and subject to change. `v1.13.x` is the last version update for `v1`. SparseCtrl, Magic Animate and other control methods will be supported in `v2` via updating both this repo and sd-webui-controlnet.


## How to Use
1. Update your WebUI to v1.6.0 and ControlNet to v1.1.410, then install this extension via link. I do not plan to support older version.
1. Download motion modules and put the model weights under `stable-diffusion-webui/extensions/sd-webui-animatediff/model/`. If you want to use another directory to save model weights, please go to `Settings/AnimateDiff`. See [model zoo](#model-zoo) for a list of available motion modules.
1. Enable `Pad prompt/negative prompt to be same length` in `Settings/Optimization` and click `Apply settings`. You must do this to prevent generating two separate unrelated GIFs. Checking `Batch cond/uncond` is optional, which can improve speed but increase VRAM usage.
1. DO NOT disable hash calculation, otherwise AnimateDiff will have trouble figuring out when you switch motion module.

### WebUI
1. Go to txt2img if you want to try txt2gif and img2img if you want to try img2gif.
1. Choose an SD1.5 checkpoint, write prompts, set configurations such as image width/height. If you want to generate multiple GIFs at once, please [change batch number, instead of batch size](#batch-size).
1. Enable AnimateDiff extension, set up [each parameter](#webui-parameters), then click `Generate`.
1. You should see the output GIF on the output gallery. You can access GIF output at `stable-diffusion-webui/outputs/{txt2img or img2img}-images/AnimateDiff/{yy-mm-dd}`. You can also access image frames at `stable-diffusion-webui/outputs/{txt2img or img2img}-images/{yy-mm-dd}`. You may choose to save frames for each generation into separate directories in `Settings/AnimateDiff`.

### API
It is quite similar to the way you use ControlNet. API will return a video in base64 format. In `format`, `PNG` means to save frames to your file system without returning all the frames. If you want your API to return all frames, please add `Frame` to `format` list. For most up-to-date parameters, please read [here](https://github.com/continue-revolution/sd-webui-animatediff/blob/master/scripts/animatediff_ui.py#L26).
```
'alwayson_scripts': {
  'AnimateDiff': {
    'args': [{
      'model': 'mm_sd_v15_v2.ckpt',   # Motion module
      'format': ['GIF'],      # Save format, 'GIF' | 'MP4' | 'PNG' | 'WEBP' | 'WEBM' | 'TXT' | 'Frame'
      'enable': True,         # Enable AnimateDiff
      'video_length': 16,     # Number of frames
      'fps': 8,               # FPS
      'loop_number': 0,       # Display loop number
      'closed_loop': 'R+P',   # Closed loop, 'N' | 'R-P' | 'R+P' | 'A'
      'batch_size': 16,       # Context batch size
      'stride': 1,            # Stride 
      'overlap': -1,          # Overlap
      'interp': 'Off',        # Frame interpolation, 'Off' | 'FILM'
      'interp_x': 10          # Interp X
      'video_source': 'path/to/video.mp4',  # Video source
      'video_path': 'path/to/frames',       # Video path
      'latent_power': 1,      # Latent power
      'latent_scale': 32,     # Latent scale
      'last_frame': None,     # Optional last frame
      'latent_power_last': 1, # Optional latent power for last frame
      'latent_scale_last': 32,# Optional latent scale for last frame
      'request_id': ''        # Optional request id. If provided, outputs will have request id as filename suffix
      }
    ]
  }
},
```


## WebUI Parameters
1. **Save format** — Format of the output. Choose at least one of "GIF"|"MP4"|"WEBP"|"WEBM"|"PNG". Check "TXT" if you want infotext, which will live in the same directory as the output GIF. Infotext is also accessible via `stable-diffusion-webui/params.txt` and outputs in all formats.
    1. You can optimize GIF with `gifsicle` (`apt install gifsicle` required, read [#91](https://github.com/continue-revolution/sd-webui-animatediff/pull/91) for more information) and/or `palette` (read [#104](https://github.com/continue-revolution/sd-webui-animatediff/pull/104) for more information). Go to `Settings/AnimateDiff` to enable them.
    1. You can set quality and lossless for WEBP via `Settings/AnimateDiff`. Read [#233](https://github.com/continue-revolution/sd-webui-animatediff/pull/233) for more information.
    1. If you are using API, by adding "PNG" to `format`, you can save all frames to your file system without returning all the frames. If you want your API to return all frames, please add `Frame` to `format` list.
1. **Number of frames** — Choose whatever number you like. 

    If you enter 0 (default):
    - If you submit a video via `Video source` / enter a video path via `Video path` / enable ANY batch ControlNet, the number of frames will be the number of frames in the video (use shortest if more than one videos are submitted).
    - Otherwise, the number of frames will be your `Context batch size` described below.

    If you enter something smaller than your `Context batch size` other than 0: you will get the first `Number of frames` frames as your output GIF from your whole generation. All following frames will not appear in your generated GIF, but will be saved as PNGs as usual. Do not set `Number of frames` to be something smaler than `Context batch size` other than 0 because of [#213](https://github.com/continue-revolution/sd-webui-animatediff/issues/213).
1. **FPS** — Frames per second, which is how many frames (images) are shown every second. If 16 frames are generated at 8 frames per second, your GIF’s duration is 2 seconds. If you submit a source video, your FPS will be the same as the source video.
1. **Display loop number** — How many times the GIF is played. A value of `0` means the GIF never stops playing.
1. **Context batch size** — How many frames will be passed into the motion module at once. The SD1.5 motion modules are trained with 16 frames, so it’ll give the best results when the number of frames is set to `16`. SDXL HotShotXL motion modules are trained with 8 frames instead. Choose [1, 24] for V1 / HotShotXL motion modules and [1, 32] for V2 / AnimateDiffXL motion modules.
1. **Closed loop** — Closed loop means that this extension will try to make the last frame the same as the first frame.
    1. When `Number of frames` > `Context batch size`, including when ControlNet is enabled and the source video frame number > `Context batch size` and `Number of frames` is 0, closed loop will be performed by AnimateDiff infinite context generator.
    1. When `Number of frames` <= `Context batch size`, AnimateDiff infinite context generator will not be effective. Only when you choose `A` will AnimateDiff append reversed list of frames to the original list of frames to form closed loop.

    See below for explanation of each choice:

    - `N` means absolutely no closed loop - this is the only available option if `Number of frames` is smaller than `Context batch size` other than 0.
    - `R-P` means that the extension will try to reduce the number of closed loop context. The prompt travel will not be interpolated to be a closed loop.
    - `R+P` means that the extension will try to reduce the number of closed loop context. The prompt travel will be interpolated to be a closed loop.
    - `A` means that the extension will aggressively try to make the last frame the same as the first frame. The prompt travel will be interpolated to be a closed loop.
1. **Stride** — Max motion stride as a power of 2 (default: 1).
    1. Due to the limitation of the infinite context generator, this parameter is effective only when `Number of frames` > `Context batch size`, including when ControlNet is enabled and the source video frame number > `Context batch size` and `Number of frames` is 0.
    1. "Absolutely no closed loop" is only possible when `Stride` is 1.
    1. For each 1 <= $2^i$ <= `Stride`, the infinite context generator will try to make frames $2^i$ apart temporal consistent. For example, if `Stride` is 4 and `Number of frames` is 8, it will make the following frames temporal consistent:
        - `Stride` == 1: [0, 1, 2, 3, 4, 5, 6, 7]
        - `Stride` == 2: [0, 2, 4, 6], [1, 3, 5, 7]
        - `Stride` == 4: [0, 4], [1, 5], [2, 6], [3, 7]
1. **Overlap** — Number of frames to overlap in context. If overlap is -1 (default): your overlap will be `Context batch size` // 4.
    1. Due to the limitation of the infinite context generator, this parameter is effective only when `Number of frames` > `Context batch size`, including when ControlNet is enabled and the source video frame number > `Context batch size` and `Number of frames` is 0.
1. **Frame Interpolation** — Interpolate between frames with Deforum's FILM implementation. Requires Deforum extension. [#128](https://github.com/continue-revolution/sd-webui-animatediff/pull/128)
1. **Interp X** — Replace each input frame with X interpolated output frames. [#128](https://github.com/continue-revolution/sd-webui-animatediff/pull/128).
1. **Video source** — [Optional] Video source file for [ControlNet V2V](#controlnet-v2v). You MUST enable ControlNet. It will be the source control for ALL ControlNet units that you enable without submitting a control image or a path to ControlNet panel. You can of course submit one control image via `Single Image` tab or an input directory via `Batch` tab, which will override this video source input and work as usual.
1. **Video path** — [Optional] Folder for source frames for [ControlNet V2V](#controlnet-v2v), but lower priority than `Video source`. You MUST enable ControlNet. It will be the source control for ALL ControlNet units that you enable without submitting a control image or a path to ControlNet. You can of course submit one control image via `Single Image` tab or an input directory via `Batch` tab, which will override this video path input and work as usual.
    - For people who want to inpaint videos: enter a folder which contains two sub-folders `image` and `mask` on ControlNet inpainting unit. These two sub-folders should contain the same number of images. This extension will match them according to the same sequence. Using my [Segment Anything](https://github.com/continue-revolution/sd-webui-segment-anything) extension can make your life much easier.

Please read
- [Img2GIF](#img2gif) for extra parameters on img2gif panel.
- [Prompt Travel](#prompt-travel) for how to trigger prompt travel.
- [ControlNet V2V](#controlnet-v2v) for how to use ControlNet V2V.
- [Model Spec](#model-spec) for how to use Motion LoRA, V3 and SDXL.


## Img2GIF
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

The second and third lines are for prompt interpolation, in format `frame number`: `prompt`. Your `frame number` should be in ascending order, smaller than the total `Number of frames`. The first frame is 0 index.

The last line is tail prompt, which is optional. You can write no/single/multiple lines of tail prompts. If you don't need this feature, just write prompts in the old way.
```
1girl, yoimiya (genshin impact), origen, line, comet, wink, Masterpiece, BestQuality. UltraDetailed, <lora:LineLine2D:0.7>,  <lora:yoimiya:0.8>, 
0: closed mouth
8: open mouth
smile
```


## ControlNet V2V
You need to go to txt2img / img2img-batch and submit source video or path to frames. Each ControlNet will find control images according to this priority:
1. ControlNet `Single Image` tab or `Batch` tab. Simply upload a control image or a directory of control frames is enough.
1. Img2img Batch tab `Input directory` if you are using img2img batch. If you upload a directory of control frames, it will be the source control for ALL ControlNet units that you enable without submitting a control image or a path to ControlNet panel.
1. AnimateDiff `Video Source`. If you upload a video through `Video Source`, it will be the source control for ALL ControlNet units that you enable without submitting a control image or a path to ControlNet panel.
1. AnimateDiff `Video Path`. If you upload a path to frames through `Video Path`, it will be the source control for ALL ControlNet units that you enable without submitting a control image or a path to ControlNet panel.

`Number of frames` will be capped to the minimum number of images among all **folders** you provide. Each control image in each folder will be applied to one single frame. If you upload one single image for a ControlNet unit, that image will control **ALL** frames.

For people who want to inpaint videos: enter a folder which contains two sub-folders `image` and `mask` on ControlNet inpainting unit. These two sub-folders should contain the same number of images. This extension will match them according to the same sequence. Using my [Segment Anything](https://github.com/continue-revolution/sd-webui-segment-anything) extension can make your life much easier.

AnimateDiff in img2img batch will be available in [v1.10.0](https://github.com/continue-revolution/sd-webui-animatediff/pull/224).


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


## Optimizations
Optimizations can be significantly helpful if you want to improve speed and reduce VRAM usage. With [attention optimization](#attention), [FP8](#fp8) and unchecking `Batch cond/uncond` in `Settings/Optimization`, I am able to run 4 x ControlNet + AnimateDiff + Stable Diffusion to generate 36 frames of 1024 * 1024 images with 18GB VRAM.

### Attention
Adding `--xformers` / `--opt-sdp-attention` to your command lines can significantly reduce VRAM and improve speed. However, due to a bug in xformers, you may or may not get CUDA error. If you get CUDA error, please either completely switch to `--opt-sdp-attention`, or preserve `--xformers` -> go to `Settings/AnimateDiff` -> choose "Optimize attention layers with sdp (torch >= 2.0.0 required)".

### FP8
FP8 requires torch >= 2.1.0 and WebUI [test-fp8](https://github.com/AUTOMATIC1111/stable-diffusion-webui/tree/test-fp8) branch by [@KohakuBlueleaf](https://github.com/KohakuBlueleaf). Follow these steps to enable FP8:
1. Switch to `test-fp8` branch via `git checkout test-fp8` in your `stable-diffusion-webui` directory.
1. Reinstall torch via adding `--reinstall-torch` ONCE to your command line arguments.
1. Goto Settings Tab > Optimizations > FP8 weight, change it to `Enable`

### LCM
[Latent Consistency Model](https://github.com/luosiallen/latent-consistency-model) is a recent breakthrough in Stable Diffusion community. I provide a "gift" to everyone who update this extension to >= [v1.12.1](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.12.1) - you will find `LCM` sampler in the normal place you select samplers in WebUI. You can generate images / videos within 6-8 steps if you
- select `Euler A` / `Euler` / `LCM` sampler (other samplers may also work, subject to further experiments)
- use [LCM LoRA](https://civitai.com/models/195519/lcm-lora-weights-stable-diffusion-acceleration-module)
- use a low CFG denoising strength (1-2 is recommended)

Note that LCM sampler is still under experiment and subject to change adhering to [@luosiallen](https://github.com/luosiallen)'s wish.

Benefits of using this extension instead of [sd-webui-lcm](https://github.com/0xbitches/sd-webui-lcm) are
- you do not need to install diffusers
- you can use LCM sampler with any other extensions, such as ControlNet and AnimateDiff

### Others
- Remove any VRAM heavy arguments such as `--no-half`. These arguments can significantly increase VRAM usage and reduce speed.
- Check `Batch cond/uncond` in `Settings/Optimization` to improve speed; uncheck it to reduce VRAM usage.


## Model Zoo
I am maintaining a [huggingface repo](https://huggingface.co/conrevo/AnimateDiff-A1111/tree/main) to provide all official models in fp16 & safetensors format. You are highly recommended to use my link. You MUST use my link to download adapter for V3. You may still use the old links if you want, for all models except adapter for V3.

- "Official" models by [@guoyww](https://github.com/guoyww): [Google Drive](https://drive.google.com/drive/folders/1EqLC65eR1-W-sGD0Im7fkED6c8GkiNFI) | [HuggingFace](https://huggingface.co/guoyww/animatediff/tree/main) | [CivitAI](https://civitai.com/models/108836)
- "Stabilized" community models by [@manshoety](https://huggingface.co/manshoety): [HuggingFace](https://huggingface.co/manshoety/AD_Stabilized_Motion/tree/main)
- "TemporalDiff" models by [@CiaraRowles](https://huggingface.co/CiaraRowles): [HuggingFace](https://huggingface.co/CiaraRowles/TemporalDiff/tree/main)
- "HotShotXL" models by [@hotshotco](https://huggingface.co/hotshotco/): [HuggingFace](https://huggingface.co/hotshotco/Hotshot-XL/tree/main)


## VRAM
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


## Demo

### Basic Usage
| AnimateDiff | Extension | img2img |
| --- | --- | --- |
| ![image](https://user-images.githubusercontent.com/63914308/255306527-5105afe8-d497-4ab1-b5c4-37540e9601f8.gif) |![00013-10788741199826055000](https://github.com/continue-revolution/sd-webui-animatediff/assets/63914308/43b9cf34-dbd1-4120-b220-ea8cb7882272) | ![00018-727621716](https://github.com/continue-revolution/sd-webui-animatediff/assets/63914308/d04bb573-c8ca-4ae6-a2d9-81f8012bec3a) |

### Motion LoRA
| No LoRA | PanDown | PanLeft |
| --- | --- | --- |
| ![00094-1401397431](https://github.com/continue-revolution/sd-webui-animatediff/assets/63914308/d8d2b860-c781-4dd0-8c0a-0eb26970130b) | ![00095-3197605735](https://github.com/continue-revolution/sd-webui-animatediff/assets/63914308/aed2243f-5494-4fe3-a10a-96c57f6f2906) | ![00093-2722547708](https://github.com/continue-revolution/sd-webui-animatediff/assets/63914308/c32e9aaf-54f2-4f40-879b-e800c7c7848c) |

### Prompt Travel
![00201-2296305953](https://github.com/continue-revolution/sd-webui-animatediff/assets/63914308/881f317c-f1d2-4635-b84b-b4c4881650f6)

The prompt is similar to [above](#prompt-travel).

### AnimateDiff V3
You should be able to read infotext to understand how I generated this sample.
![00024-3973810345](https://github.com/continue-revolution/sd-webui-animatediff/assets/63914308/5f3e3858-8033-4a16-94b0-4dbc0d0a67fc)


### AnimateDiff SDXL
You should be able to read infotext to understand how I generated this sample.
![00025-1668075705](https://github.com/continue-revolution/sd-webui-animatediff/assets/63914308/6d32daf9-51c6-490f-a942-db36f84f23cf)

### ControlNet V2V
TODO


## Tutorial 
TODO


## Thanks
I thank researchers from [Shanghai AI Lab](https://www.shlab.org.cn/), especially [@guoyww](https://github.com/guoyww) for creating AnimateDiff. I also thank [@neggles](https://github.com/neggles) and [@s9roll7](https://github.com/s9roll7) for creating and improving [AnimateDiff CLI Prompt Travel](https://github.com/s9roll7/animatediff-cli-prompt-travel). This extension could not be made possible without these creative works.

I also thank community developers, especially
- [@zappityzap](https://github.com/zappityzap) who developed the majority of the [output features](https://github.com/continue-revolution/sd-webui-animatediff/blob/master/scripts/animatediff_output.py)
- [@TDS4874](https://github.com/TDS4874) and [@opparco](https://github.com/opparco) for resolving the grey issue which significantly improve the performance
- [@talesofai](https://github.com/talesofai) who developed i2v in [this forked repo](https://github.com/talesofai/AnimateDiff)
- [@rkfg](https://github.com/rkfg) for developing GIF palette optimization

and many others who have contributed to this extension.

I also thank community users, especially [@streamline](https://twitter.com/kaizirod) who provided dataset and workflow of ControlNet V2V. His workflow is extremely amazing and definitely worth checking out.


## Star History
<a href="https://star-history.com/#continue-revolution/sd-webui-animatediff&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=continue-revolution/sd-webui-animatediff&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=continue-revolution/sd-webui-animatediff&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=continue-revolution/sd-webui-animatediff&type=Date" />
  </picture>
</a>


## Sponsor
You can sponsor me via WeChat, AliPay or [PayPal](https://paypal.me/conrevo). You can also support me via [patreon](https://www.patreon.com/conrevo), [ko-fi](https://ko-fi.com/conrevo) or [afdian](https://afdian.net/a/conrevo).

| WeChat | AliPay | PayPal |
| --- | --- | --- |
| ![216aff0250c7fd2bb32eeb4f7aae623](https://user-images.githubusercontent.com/63914308/232824466-21051be9-76ce-4862-bb0d-a431c186fce1.jpg) | ![15fe95b4ada738acf3e44c1d45a1805](https://user-images.githubusercontent.com/63914308/232824545-fb108600-729d-4204-8bec-4fd5cc8a14ec.jpg) | ![IMG_1419_](https://github.com/continue-revolution/sd-webui-animatediff/assets/63914308/eaa7b114-a2e6-4ecc-a29f-253ace06d1ea) |
