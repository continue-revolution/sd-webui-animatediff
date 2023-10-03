# AnimateDiff for Stable Diffusion WebUI

This extension aim for integrating [AnimateDiff](https://github.com/guoyww/AnimateDiff/) into [AUTOMATIC1111 Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui). You can generate GIFs in exactly the same way as generating images after enabling this extension.

This extension implements AnimateDiff in a different way. It does not require you to clone the whole SD1.5 repository. It also applied (probably) the least modification to `ldm`, so that you do not need to reload your model weights if you don't want to.

Batch size on WebUI will be replaced by GIF frame number internally: 1 full GIF generated in 1 batch. If you want to generate multiple GIF at once, please change batch number. 

Batch number is NOT the same as batch size. In A1111 WebUI, batch number is above batch size. Batch number means the number of sequential steps, but batch size means the number of parallel steps. You do not have to worry too much when you increase batch number, but you do need to worry about your VRAM when you increase your batch size (where in this extension, video frame number). You do not need to change batch size at all when you are using this extension.

You might also be interested in another extension I created: [Segment Anything for Stable Diffusion WebUI](https://github.com/continue-revolution/sd-webui-segment-anything).

## How to Use

1. Install this extension via link.
1. Download motion modules and put the model weights under `stable-diffusion-webui/extensions/sd-webui-animatediff/model/`. If you want to use another directory to save the model weights, please go to `Settings/AnimateDiff`. See [model zoo](#motion-module-model-zoo) for a list of available motion modules.
1. Enable `Pad prompt/negative prompt to be same length` in `Settings/Optimization` and click `Apply settings`. You must do this to prevent generating two separate unrelated GIFs. Checking `Batch cond/uncond` is optional, which can improve speed but increase VRAM usage.

### WebUI
1. Go to txt2img if you want to try txt2gif and img2img if you want to try img2gif.
1. Choose an SD1.5 checkpoint, write prompts, set configurations such as image width/height. If you want to generate multiple GIFs at once, please change batch number, instead of batch size.
1. Enable AnimateDiff extension, and set up each parameter, and click `Generate`.
   1. **Number of frames** — Choose whatever number you like. 
      
      If you enter 0 (default):
      - If you submit a video via `Video source` / enter a video path via `Video path` / enable ANY batch ControlNet, the number of frames will be the number of frames in the video (use shortest if more than one videos are submitted).
      - Otherwise, the number of frames will be your `Batch size` described below.

      If you enter something smaller than your `Batch size` other than 0: you will get the first `Number of frames` frames as your output GIF from your whole generation. All following frames will not appear in your generated GIF, but will be saved as PNGs as usual.
   1. **FPS** — Frames per second, which is how many frames (images) are shown every second. If 16 frames are generated at 8 frames per second, your GIF’s duration is 2 seconds. If you submit a source video, your FPS will be the same as the source video.
   1. **Display loop number** — How many times the GIF is played. A value of `0` means the GIF never stops playing.
   1. **Batch size** — How many frames will be passed into the motion module at once. The model is trained with 16 frames, so it’ll give the best results when the number of frames is set to `16`. Choose [1, 24] for V1 motion modules and [1, 32] for V2 motion modules.
   1. **Closed loop** — If you enable this option and your number of frames is greater than your batch size, this extension will try to make the last frame the same as the first frame.
   1. **Stride** — Max motion stride as a power of 2 (default: 1).
   1. **Overlap** — Number of frames to overlap in context. If overlap is -1 (default): your overlap will be `Batch size` // 4.
   1. **Save** — Format of the output. Choose at least one of "GIF"|"MP4"|"PNG". Check "TXT" if you want infotext, which will live in the same directory as the output GIF.
        1. You can optimize GIF with `gifsicle` (`apt install gifsicle` required, read [#91](https://github.com/continue-revolution/sd-webui-animatediff/pull/91) for more information) and/or `palette` (read [#104](https://github.com/continue-revolution/sd-webui-animatediff/pull/104) for more information). Go to `Settings/AnimateDiff` to enable them.
   1. **Reverse** — Append reversed frames to your output. See [#112](https://github.com/continue-revolution/sd-webui-animatediff/issues/112) for instruction.
   1. **Frame Interpolation** Interpolate between frames with Deforum's FILM implementation. Requires Deforum extension. [#128](https://github.com/continue-revolution/sd-webui-animatediff/pull/128)
   1. **Interp X** Replace each input frame with X interpolated output frames. [#128](https://github.com/continue-revolution/sd-webui-animatediff/pull/128).
   1. **Video source** — [Optional] Video source file for video to video generation. You MUST enable ControlNet. It will be the source control for ALL ControlNet units that you enable without submitting a control image or a path to ControlNet panel. You can of course submit a control image via `Single Image` tab or an input directory via `Batch` tab, they will override this video source input and work as usual.
   1. **Video path** — [Optional] Folder for source frames for video to video generation, but lower priority than `Video source`. You MUST enable ControlNet. It will be the source control for ALL ControlNet units that you enable without submitting a control image or a path to ControlNet. You can of course submit a control image via `Single Image` tab or an input directory via `Batch` tab, they will override this video path input and work as usual.
      - For people who want to inpaint videos: enter a folder which contains two sub-folders `image` and `mask`. They should contain the same number of images. This extension will match them according to the same sequence. You should not upload a video directly to **Video source** mentioned above. Using my [Segment Anything](https://github.com/continue-revolution/sd-webui-segment-anything) extension can make your life much easier.
1. You should see the output GIF on the output gallery. You can access GIF output at `stable-diffusion-webui/outputs/{txt2img or img2img}-images/AnimateDiff`. You can also access image frames at `stable-diffusion-webui/outputs/{txt2img or img2img}-images/{date}`.

#### img2GIF

You need to go to img2img and submit an init frame via A1111 panel. You can optionally submit a last frame via extension panel (experiment feature, not tested, not sure if it will work).

By default: your `init_latent` will be changed to 
```
init_alpha = (1 - frame_number ^ latent_power / latent_scale)
init_latent = init_latent * init_alpha + random_tensor * (1 - init_alpha)
```

If you upload a last frame: your `init_latent` will be changed in a similar way. Read [this code](https://github.com/continue-revolution/sd-webui-animatediff/tree/v1.5.0/scripts/animatediff_latent.py#L28-L65) to understand how it works.

### API
Just like how you use ControlNet. Here is a sample. You will get a list of generated frames. You will have to view GIF in your file system, as mentioned at [#WebUI](#webui) item 4.
```
'alwayson_scripts': {
  'AnimateDiff': {
  	'args': [{
  		  'enable': True,         # enable AnimateDiff
  		  'video_length': 16,     # video frame number, 0-24 for v1 and 0-32 for v2
  		  'format': 'MP4',        # 'GIF' | 'MP4' | 'PNG' | 'TXT'
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

## Motion Module Model Zoo
- `mm_sd_v14.ckpt` & `mm_sd_v15.ckpt` & `mm_sd_v15_v2.ckpt` by [@guoyww](https://github.com/guoyww): [Google Drive](https://drive.google.com/drive/folders/1EqLC65eR1-W-sGD0Im7fkED6c8GkiNFI) | [HuggingFace](https://huggingface.co/guoyww/animatediff) | [CivitAI](https://civitai.com/models/108836) | [Baidu NetDisk](https://pan.baidu.com/s/18ZpcSM6poBqxWNHtnyMcxg?pwd=et8y)
- `mm_sd_v14.safetensors` & `mm_sd_v15.safetensors` & `mm_sd_v15_v2.safetensors` by [@neph1](https://github.com/neph1): [HuggingFace](https://huggingface.co/guoyww/animatediff/tree/refs%2Fpr%2F3)
- `mm-Stabilized_high.pth` & `mm-Stabbilized_mid.pth` by [@manshoety](https://huggingface.co/manshoety): [HuggingFace](https://huggingface.co/manshoety/AD_Stabilized_Motion/tree/main)
- `temporaldiff-v1-animatediff.ckpt` by [@CiaraRowles](https://huggingface.co/CiaraRowles): [HuggingFace](https://huggingface.co/CiaraRowles/TemporalDiff/tree/main)

## Update

- `2023/07/20` [v1.1.0](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.1.0): Fix gif duration, add loop number, remove auto-download, remove xformers, remove instructions on gradio UI, refactor README, add [sponsor](#sponsor) QR code.
- `2023/07/24` [v1.2.0](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.2.0): Fix incorrect insertion of motion modules, add option to change path to save motion modules in Settings/AnimateDiff, fix loading different motion modules.
- `2023/09/04` [v1.3.0](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.3.0): Support any community models with the same architecture; fix grey problem via [#63](https://github.com/continue-revolution/sd-webui-animatediff/issues/63) (credit to [@TDS4874](https://github.com/TDS4874) and [@opparco](https://github.com/opparco))
- `2023/09/11` [v1.4.0](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.4.0): Support official v2 motion module (different architecture: GroupNorm not hacked, UNet middle layer has motion module).    
- `2023/09/14`: [v1.4.1](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.4.1): Always change `beta`, `alpha_comprod` and `alpha_comprod_prev` to resolve grey problem in other samplers.
- `2023/09/16`: [v1.5.0](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.5.0): Randomize init latent to support better img2gif, credit to [this forked repo](https://github.com/talesofai/AnimateDiff); add other output formats and infotext output, credit to [@zappityzap](https://github.com/zappityzap); add appending reversed frames; refactor code to ease maintaining.
- `2023/09/19`: [v1.5.1](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.5.1): Support xformers, sdp, sub-quadratic attention optimization - VRAM usage decrease to 5.60GB with default setting. See [FAQ](#faq) 1st item for more information.
- `2023/09/22`: [v1.5.2](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.5.2): Option to disable xformers at `Settings/AnimateDiff` [due to a bug in xformers](https://github.com/facebookresearch/xformers/issues/845), API support, option to enable GIF paletter optimization at `Settings/AnimateDiff` (credit to [@rkfg](https://github.com/rkfg)), gifsicle optimization move to `Settings/AnimateDiff`.
- `2023/09/25`: [v1.6.0](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.6.0): [Motion LoRA](https://github.com/guoyww/AnimateDiff#features) supported. Download and use them like any other LoRA you use (example: download motion lora to `stable-diffusion-webui/models/Lora` and add `<lora:v2_lora_PanDown:0.8>` to your positive prompt). **Motion LoRA only supports V2 motion modules**.
- `2023/09/27`: [v1.7.0](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.7.0): [ControlNet](https://github.com/Mikubill/sd-webui-controlnet) supported. Please closely follow the instructions in [How to Use](#how-to-use), especially the explanation of `Video source` and `Video path` attributes. ControlNet is way more complex than what I can test and I ask you to test for me. Please submit an issue whenever you find a bug. You may want to check `Do not append detectmap to output` in `Settings/ControlNet` to avoid having a series of control images in your output gallery. [Safetensors](#motion-module-model-zoo) for some motion modules are also available now.
- `2023/09/29`: [v1.8.0](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.8.0): Infinite generation (with/without ControlNet) supported. [Demo and video instructions](#demo-and-video-instructions) are coming soon.
- `2023/10/01`: [v1.8.1](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.8.1): Now you can uncheck `Batch cond/uncond` in `Settings/Optimization` if you want. This will reduce your VRAM (5.31GB -> 4.21GB for SDP) but take longer time.

Prompt Travel and other CLI features are currently work in progress inside [#71](https://github.com/continue-revolution/sd-webui-animatediff/pull/71). Stay tuned and they should be released soon.

## FAQ
1.  Q: How much VRAM do I need?

    A: Actual VRAM usage depends on your image size and context batch size. You can try to reduce image size or context batch size to reduce VRAM usage. I list some data tested on Ubuntu 22.04, NVIDIA 4090, torch 2.0.1+cu117, H=W=512, frame=16 (default setting) below. `w/`/`w/o` means `Batch cond/uncond` in `Settings/Optimization` is checked/unchecked.
    | Optimization | VRAM w/ | VRAM w/o |
    | --- | --- | --- |
    | No optimization | 12.13GB |  |
    | xformers/sdp | 5.60GB | 4.21GB |
    | sub-quadratic | 10.39GB |  |

1.  Q: Can I use SDXL to generate GIFs?

    A: You will have to wait for someone to train SDXL-specific motion modules which will have a different model architecture. This extension essentially inject multiple motion modules into SD1.5 UNet. It does not work for other variations of SD, such as SD2.1 and SDXL.


## Demo and Video Instructions

Coming soon.

## Samples

| AnimateDiff | Extension v1.2.0 | Extension v1.3.0 | img2img |
| --- | --- | --- | --- |
| ![image](https://user-images.githubusercontent.com/63914308/255306527-5105afe8-d497-4ab1-b5c4-37540e9601f8.gif) | ![00023-10788741199826055168](https://github.com/continue-revolution/sd-webui-animatediff/assets/63914308/c35a952a-a127-491b-876d-cda97771f7ee) | ![00013-10788741199826055000](https://github.com/continue-revolution/sd-webui-animatediff/assets/63914308/43b9cf34-dbd1-4120-b220-ea8cb7882272) | ![00018-727621716](https://github.com/continue-revolution/sd-webui-animatediff/assets/63914308/d04bb573-c8ca-4ae6-a2d9-81f8012bec3a) |

Note that I did not modify random tensor generation when producing v1.3.0 samples.

### Motion LoRA

| No LoRA | PanDown | PanLeft |
| --- | --- | --- |
| ![00094-1401397431](https://github.com/continue-revolution/sd-webui-animatediff/assets/63914308/d8d2b860-c781-4dd0-8c0a-0eb26970130b) | ![00095-3197605735](https://github.com/continue-revolution/sd-webui-animatediff/assets/63914308/aed2243f-5494-4fe3-a10a-96c57f6f2906) | ![00093-2722547708](https://github.com/continue-revolution/sd-webui-animatediff/assets/63914308/c32e9aaf-54f2-4f40-879b-e800c7c7848c) |

## Sponsor
You can sponsor me via WeChat, AliPay or Paypal.

| WeChat | AliPay | Paypal |
| --- | --- | --- |
| ![216aff0250c7fd2bb32eeb4f7aae623](https://user-images.githubusercontent.com/63914308/232824466-21051be9-76ce-4862-bb0d-a431c186fce1.jpg) | ![15fe95b4ada738acf3e44c1d45a1805](https://user-images.githubusercontent.com/63914308/232824545-fb108600-729d-4204-8bec-4fd5cc8a14ec.jpg) | ![IMG_1419_](https://github.com/continue-revolution/sd-webui-animatediff/assets/63914308/eaa7b114-a2e6-4ecc-a29f-253ace06d1ea) |
