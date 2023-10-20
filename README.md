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
- [Motion LoRA](#motion-lora)
- [Prompt Travel](#prompt-travel)
- [ControlNet V2V](#controlnet-v2v)
- [Motion Module Model Zoo](#motion-module-model-zoo)
- [VRAM](#vram)
- [Batch Size](#batch-size)
- [FAQ](#faq)
- [Demo](#demo)
  - [Basic Usage](#basic-usage)
  - [Motion LoRA](#motion-lora-1)
  - [Prompt Travel](#prompt-travel-1)
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
- `2023/09/27`: [v1.7.0](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.7.0): [ControlNet](https://github.com/Mikubill/sd-webui-controlnet) supported. See [ControlNet V2V](#controlnet-v2v) for more information. [Safetensors](#motion-module-model-zoo) for some motion modules are also available now.
- `2023/09/29`: [v1.8.0](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.8.0): Infinite generation supported. See [WebUI Parameters](#webui-parameters) for more information.
- `2023/10/01`: [v1.8.1](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.8.1): Now you can uncheck `Batch cond/uncond` in `Settings/Optimization` if you want. This will reduce your [VRAM](#vram) (5.31GB -> 4.21GB for SDP) but take longer time.
- `2023/10/08`: [v1.9.0](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.9.0): Prompt travel supported. You must have ControlNet installed (you do not need to enable ControlNet) to try it. See [Prompt Travel](#prompt-travel) for how to trigger this feature.
- `2023/10/11`: [v1.9.1](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.9.1): Use state_dict key to guess mm version, replace match case with if else to support python<3.10, option to save PNG to custom dir
 (see `Settings/AnimateDiff` for detail), move hints to js, install imageio\[ffmpeg\] automatically when MP4 save fails.
- `2023/10/16`: [v1.9.2](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.9.2): Add context generator to completely remove any closed loop, prompt travel support closed loop, infotext fully supported including prompt travel, README refactor
- `2023/10/??`: [v1.9,3](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.9.3): Support webp output format. See [#233](https://github.com/continue-revolution/sd-webui-animatediff/pull/233) for more information.


## How to Use
1. Update your WebUI to v1.6.0 and ControlNet to v1.1.410, then install this extension via link. I do not plan to support older version.
1. Download motion modules and put the model weights under `stable-diffusion-webui/extensions/sd-webui-animatediff/model/`. If you want to use another directory to save model weights, please go to `Settings/AnimateDiff`. See [model zoo](#motion-module-model-zoo) for a list of available motion modules.
1. Enable `Pad prompt/negative prompt to be same length` in `Settings/Optimization` and click `Apply settings`. You must do this to prevent generating two separate unrelated GIFs. Checking `Batch cond/uncond` is optional, which can improve speed but increase VRAM usage.

### WebUI
1. Go to txt2img if you want to try txt2gif and img2img if you want to try img2gif.
1. Choose an SD1.5 checkpoint, write prompts, set configurations such as image width/height. If you want to generate multiple GIFs at once, please [change batch number, instead of batch size](#batch-size).
1. Enable AnimateDiff extension, set up [each parameter](#webui-parameters), then click `Generate`.
1. You should see the output GIF on the output gallery. You can access GIF output at `stable-diffusion-webui/outputs/{txt2img or img2img}-images/AnimateDiff`. You can also access image frames at `stable-diffusion-webui/outputs/{txt2img or img2img}-images/{date}`. You may choose to save frames for each generation into separate directories in `Settings/AnimateDiff`.

### API
Just like how you use ControlNet. Here is a **stale** sample. (TODO: Update API Sample.) You will get a list of generated frames. You will have to view GIF in your file system, as mentioned at [WebUI](#webui) item 4. For most up-to-date parameters, please read [here](https://github.com/continue-revolution/sd-webui-animatediff/blob/master/scripts/animatediff_ui.py#L26).
```
'alwayson_scripts': {
  'AnimateDiff': {
  	'args': [{
  		  'enable': True,         # enable AnimateDiff
  		  'video_length': 16,     # video frame number, 0-24 for v1 and 0-32 for v2
  		  'format': ['GIF', 'PNG'],        # 'GIF' | 'MP4' | 'PNG' | 'TXT'
  		  'loop_number': 0,       # 0 = infinite loop
  		  'fps': 8,               # frames per second
  		  'model': 'mm_sd_v15_v2.ckpt',   # motion module name
  		  'reverse': [],          # 0 | 1 | 2 - 0: Add Reverse Frame, 1: Remove head, 2: Remove tail
        # parameters below are for img2gif only.
  		  'latent_power': 1,
  		  'latent_scale': 32,
  		  'last_frame': None,
  		  'latent_power_last': 1,
  		  'latent_scale_last': 32
  	  }
  	]
  }
},
```


## WebUI Parameters
1. **Number of frames** — Choose whatever number you like. 

    If you enter 0 (default):
    - If you submit a video via `Video source` / enter a video path via `Video path` / enable ANY batch ControlNet, the number of frames will be the number of frames in the video (use shortest if more than one videos are submitted).
    - Otherwise, the number of frames will be your `Context batch size` described below.

    If you enter something smaller than your `Context batch size` other than 0: you will get the first `Number of frames` frames as your output GIF from your whole generation. All following frames will not appear in your generated GIF, but will be saved as PNGs as usual. Do not set `Number of frames` to be something smaler than `Context batch size` other than 0 because of [#213](https://github.com/continue-revolution/sd-webui-animatediff/issues/213).
1. **FPS** — Frames per second, which is how many frames (images) are shown every second. If 16 frames are generated at 8 frames per second, your GIF’s duration is 2 seconds. If you submit a source video, your FPS will be the same as the source video.
1. **Display loop number** — How many times the GIF is played. A value of `0` means the GIF never stops playing.
1. **Context batch size** — How many frames will be passed into the motion module at once. The model is trained with 16 frames, so it’ll give the best results when the number of frames is set to `16`. Choose [1, 24] for V1 motion modules and [1, 32] for V2 motion modules.
1. **Closed loop** — Closed loop means that this extension will try to make the last frame the same as the first frame. Closed loop can only be made possible when `Number of frames` is greater than `Context batch size`, including when ControlNet is enabled and the source video frame number is greater than `Context batch size` and `Number of frames` is 0.
    - `N` means absolutely no closed loop - this is the only available option if `Number of frames` is smaller than `Context batch size` other than 0.
    - `R-P` means that the extension will try to reduce the number of closed loop context. The prompt travel will not be interpolated to be a closed loop.
    - `R+P` means that the extension will try to reduce the number of closed loop context. The prompt travel will be interpolated to be a closed loop.
    - `A` means that the extension will aggressively try to make the last frame the same as the first frame. The prompt travel will be interpolated to be a closed loop.
1. **Stride** — Max motion stride as a power of 2 (default: 1). TODO - Need more clear explanation.
1. **Overlap** — Number of frames to overlap in context. If overlap is -1 (default): your overlap will be `Context batch size` // 4.
1. **Save format** — Format of the output. Choose at least one of "GIF"|"MP4"|"WEBP"|"PNG". Check "TXT" if you want infotext, which will live in the same directory as the output GIF. Infotext is also accessible via `stable-diffusion-webui/params.txt` and outputs in all formats.
    1. You can optimize GIF with `gifsicle` (`apt install gifsicle` required, read [#91](https://github.com/continue-revolution/sd-webui-animatediff/pull/91) for more information) and/or `palette` (read [#104](https://github.com/continue-revolution/sd-webui-animatediff/pull/104) for more information). Go to `Settings/AnimateDiff` to enable them.
    2. You can set quality and lossless for WEBP via `Settings/AnimateDiff`. Read [#233](https://github.com/continue-revolution/sd-webui-animatediff/pull/233) for more information.
1. **Reverse** — Append reversed frames to your output. See [#112](https://github.com/continue-revolution/sd-webui-animatediff/issues/112) for instruction.
1. **Frame Interpolation** — Interpolate between frames with Deforum's FILM implementation. Requires Deforum extension. [#128](https://github.com/continue-revolution/sd-webui-animatediff/pull/128)
1. **Interp X** — Replace each input frame with X interpolated output frames. [#128](https://github.com/continue-revolution/sd-webui-animatediff/pull/128).
1. **Video source** — [Optional] Video source file for [ControlNet V2V](#controlnet-v2v). You MUST enable ControlNet. It will be the source control for ALL ControlNet units that you enable without submitting a control image or a path to ControlNet panel. You can of course submit one control image via `Single Image` tab or an input directory via `Batch` tab, which will override this video source input and work as usual.
1. **Video path** — [Optional] Folder for source frames for [ControlNet V2V](#controlnet-v2v), but lower priority than `Video source`. You MUST enable ControlNet. It will be the source control for ALL ControlNet units that you enable without submitting a control image or a path to ControlNet. You can of course submit one control image via `Single Image` tab or an input directory via `Batch` tab, which will override this video path input and work as usual.
    - For people who want to inpaint videos: enter a folder which contains two sub-folders `image` and `mask` on ControlNet inpainting unit. These two sub-folders should contain the same number of images. This extension will match them according to the same sequence. Using my [Segment Anything](https://github.com/continue-revolution/sd-webui-segment-anything) extension can make your life much easier.

Please read
- [Img2GIF](#img2gif) for extra parameters on img2gif panel. For how to use Motion LoRA, please read 
- [Motion LoRA](#motion-lora) for how to use Motion LoRA.
- [Prompt Travel](#prompt-travel) for how to trigger prompt travel.
- [ControlNet V2V](#controlnet-v2v) for how to use ControlNet V2V.


## Img2GIF
You need to go to img2img and submit an init frame via A1111 panel. You can optionally submit a last frame via extension panel.

By default: your `init_latent` will be changed to 
```
init_alpha = (1 - frame_number ^ latent_power / latent_scale)
init_latent = init_latent * init_alpha + random_tensor * (1 - init_alpha)
```

If you upload a last frame: your `init_latent` will be changed in a similar way. Read [this code](https://github.com/continue-revolution/sd-webui-animatediff/tree/v1.5.0/scripts/animatediff_latent.py#L28-L65) to understand how it works.


## Motion LoRA
[Download](https://huggingface.co/guoyww/animatediff) and use them like any other LoRA you use (example: download motion lora to `stable-diffusion-webui/models/Lora` and add `<lora:v2_lora_PanDown:0.8>` to your positive prompt). **Motion LoRA only supports V2 motion modules**.


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
Each ControlNet will find control images according to this priority:
1. ControlNet `Single Image` tab or `Batch` tab. Simply upload a control image or a directory of control frames is enough.
1. AnimateDiff `Video Source`. If you upload a video through `Video Source`, it will be the source control for ALL ControlNet units that you enable without submitting a control image or a path to ControlNet panel.
1. AnimateDiff `Video Path`. If you upload a video through `Video Path`, it will be the source control for ALL ControlNet units that you enable without submitting a control image or a path to ControlNet panel.

`Number of frames` will be capped to the minimum number of images among all **folders** you provide. Each control image in each folder will be applied to one single frame. If you upload one single image for a ControlNet unit, that image will control **ALL** frames.

For people who want to inpaint videos: enter a folder which contains two sub-folders `image` and `mask` on ControlNet inpainting unit. These two sub-folders should contain the same number of images. This extension will match them according to the same sequence. Using my [Segment Anything](https://github.com/continue-revolution/sd-webui-segment-anything) extension can make your life much easier.


## Motion Module Model Zoo
- `mm_sd_v14.ckpt` & `mm_sd_v15.ckpt` & `mm_sd_v15_v2.ckpt` by [@guoyww](https://github.com/guoyww): [Google Drive](https://drive.google.com/drive/folders/1EqLC65eR1-W-sGD0Im7fkED6c8GkiNFI) | [HuggingFace](https://huggingface.co/guoyww/animatediff) | [CivitAI](https://civitai.com/models/108836) | [Baidu NetDisk](https://pan.baidu.com/s/18ZpcSM6poBqxWNHtnyMcxg?pwd=et8y)
- `mm_sd_v14.safetensors` & `mm_sd_v15.safetensors` & `mm_sd_v15_v2.safetensors` by [@neph1](https://github.com/neph1): [HuggingFace](https://huggingface.co/guoyww/animatediff/tree/refs%2Fpr%2F3)
- `mm-Stabilized_high.pth` & `mm-Stabbilized_mid.pth` by [@manshoety](https://huggingface.co/manshoety): [HuggingFace](https://huggingface.co/manshoety/AD_Stabilized_Motion/tree/main)
- `temporaldiff-v1-animatediff.ckpt` by [@CiaraRowles](https://huggingface.co/CiaraRowles): [HuggingFace](https://huggingface.co/CiaraRowles/TemporalDiff/tree/main)


## VRAM
Actual VRAM usage depends on your image size and context batch size. You can try to reduce image size or context batch size to reduce VRAM usage. I list some data tested on Ubuntu 22.04, NVIDIA 4090, torch 2.0.1+cu117, H=W=512, frame=16 (default setting) below. `w/`/`w/o` means `Batch cond/uncond` in `Settings/Optimization` is checked/unchecked.
| Optimization | VRAM w/ | VRAM w/o |
| --- | --- | --- |
| No optimization | 12.13GB |  |
| xformers/sdp | 5.60GB | 4.21GB |
| sub-quadratic | 10.39GB |  |


## Batch Size 
Batch size on WebUI will be replaced by GIF frame number internally: 1 full GIF generated in 1 batch. If you want to generate multiple GIF at once, please change batch number. 

Batch number is NOT the same as batch size. In A1111 WebUI, batch number is above batch size. Batch number means the number of sequential steps, but batch size means the number of parallel steps. You do not have to worry too much when you increase batch number, but you do need to worry about your VRAM when you increase your batch size (where in this extension, video frame number). You do not need to change batch size at all when you are using this extension.

We are currently developing approach to support batch size on WebUI in the near future.


## FAQ
1.  Q: Can I use SDXL to generate GIFs?

    A: You will have to wait for someone to train SDXL-specific motion modules which will have a different model architecture. This extension essentially inject multiple motion modules into SD1.5 UNet. It does not work for other variations of SD, such as SD2.1 and SDXL.

1.  Q: Will ADetailer be supported?

    A: I'm not planning to support ADetailer. However, I plan to refactor my [Segment Anything](https://github.com/continue-revolution/sd-webui-segment-anything) to achieve similar effects.


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

### ControlNet V2V
TODO


## Tutorial 
TODO


## Thanks
I thank researchers from [Shanghai AI Lab](https://www.shlab.org.cn/), especially [@guoyww](https://github.com/guoyww) for creating AnimateDiff. I also thank [@neggles](https://github.com/neggles) and [@s9roll7](https://github.com/s9roll7) for creating and improving [AnimateDiff CLI Prompt Travel](https://github.com/s9roll7/animatediff-cli-prompt-travel). This extension could not be made possible without these creative works.

I also thank community developers, especially
- [@talesofai](https://github.com/talesofai) who developed i2v in [this forked repo](https://github.com/talesofai/AnimateDiff)
- [@zappityzap](https://github.com/zappityzap) who developed the majority of the [output features](https://github.com/continue-revolution/sd-webui-animatediff/blob/master/scripts/animatediff_output.py)
- [@TDS4874](https://github.com/TDS4874) and [@opparco](https://github.com/opparco) for resolving the grey issue which significantly improve the performance of this extension
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
