# AnimateDiff for Stable Diffusion WebUI

This extension is aimed at integrating [AnimateDiff](https://github.com/guoyww/AnimateDiff/) into [AUTOMATIC1111 Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui), so you can generate videos as easily as you currently generate images.

This extension implements AnimateDiff in a different way. It does not require you to clone the whole SD1.5 repository. It also applied (probably) the least modification to `ldm`, so that you do not need to reload your model weights if you don't want to.

You might also be interested in another extension I created: [Segment Anything for Stable Diffusion WebUI](https://github.com/continue-revolution/sd-webui-segment-anything).

### Remarks:
- The extension replaces A111's `Batch Size` with its own `Number of frames` setting, which acts the same way by controlling parallel image generation. **Caution: this could eat up VRAM if set too high**. 
- `Batch Size` still works if you use a bigger number than `Number of frames`, the extra frames won't be put together as a GIF/Video but will still be generated as PNGs if you choose this format as `Save` format
- *Reminder: `Batch Count` manages sequential operations, which shouldn't affect VRAM.*
- For now, **this extension only works with models based on Stable Diffusion 1.5 base model**

## How to Use :
1. Install this extension using this URL of this repo via the `Install From URL` tab in A111's WebUI.
2. Download one or more [motion module](#motion-module-model-zoo) and put it under `stable-diffusion-webui/extensions/sd-webui-animatediff/model/`. *You can change this path in* `Settings/AnimateDiff`.
3. To avoid consistency issues, enable `Pad prompt/negative prompt to be same length` & `Batch cond/uncond` in A111's `Settings` (don't forget to `Apply settings`).
4. To generate Videos from simple text prompts, use the `Txt2Img` tab, to use an image as input, use the `Img2Img` tab.
5. Enable AnimateDiff, set your [parameters](#parameters) & click `Generate`.
6. Find the generated videos in your usual Output folder.

### Parameters:
*If you want to use **Infinite V2V**, set those parameters to the value indicated in parenthesis*
1. `Number of frames` — (**IV2V** : 0) How many frames will be generated (16 recommended).
   > *Set it to 0 if you want to use the number of frames of the input video you set using `Video source` or `Video path`, or ANY batch ControlNet (if more than one videos is submitted, it will use the shortest).*
2. `FPS` — Frames per second, which is how many images will be shown every second. Video duratio is then `Number of frames` / `FPS`, example : 16 Frames at 8 fps will create a 16/8 = 2 second video.
   > *Using an input video will override this FPS setting*
3. `Display loop number` — How many times the GIF will loop. Set it to `0` for infinite loops.
4. `Closed loop` — (**IV2V** : unchecked) Will try to make the loop seamless by making the last frame look like the first
5. `Stride` — (**IV2V** : 1) Max motion stride as a power of 2 (default: 1).  [**what does it mean ⁉️**]
6. `Overlap` — (**IV2V** : -1) Number of frames to overlap in context. If overlap is -1 (default): your overlap will be `Batch size` // 4.  [**what does it mean ⁉️**]
7. `Save` — Format of the generated videos. Choose at least one of "GIF"|"MP4"|"PNG". Check "TXT" if you want generation info as a .txt file alongside the video files
    > *You can optimize GIF outputs with [gifsicle](https://github.com/continue-revolution/sd-webui-animatediff/pull/91) and/or [palette](https://github.com/continue-revolution/sd-webui-animatediff/pull/104). Go to `Settings/AnimateDiff` to enable them.*
8.  `Reverse` — Options to make a boomerang effect on the video by reversing it at the end. Enable by checking `Add Reverse Frame`.
    > *`Remove head` or `Remove tail` will remove the first or last frame respectively on the reversed part to avoid frame duplication.*
9.  `Frame Interpolation` — ⚠️Requires Deforum extension⚠️. Interpolates frames with Deforum's FILM implementation .
    > *See discussion [#128](https://github.com/continue-revolution/sd-webui-animatediff/pull/128) for current progress on this implementation.*
10. `Interp X` — [**needs more intuitive name**] Number of frames to interpolate between generated frames.
11. `Video source` — [Optional] Video source file for video to video generation.
    >*⚠️ Requires Controlnet to be enabled, the video will then be the source control for ALL ControlNet units that you enable unless you manually set them in the ControlNet panel.*
12. `Video path` — [Optional & overriden by `Video source`] Same as `Video source`, but if you want the path of an image sequence instead.
    > ℹ️ **You can inpaint videos using this field!** Create two sub folders at this path : `image` containing the images of the source video & `mask` containing the inpaint masked frames. Using my [Segment Anything](https://github.com/continue-revolution/sd-webui-segment-anything) extension can make your life much easier.

#### img2vid

1. Use the img2img tab as usual to use an image as input.
2. (Experimental) You can optionally submit a last frame in the extension panel.

By default: your `init_latent` will be changed to 
```
init_alpha = (1 - frame_number ^ latent_power / latent_scale)
init_latent = init_latent * init_alpha + random_tensor * (1 - init_alpha)
``` 

If you upload a last frame: your `init_latent` will be changed in a similar way. Read [this code](https://github.com/continue-revolution/sd-webui-animatediff/tree/v1.5.0/scripts/animatediff_latent.py#L28-L65) to understand how it works.

[**whole part is confusing**]

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

- `2023/07/20` [v1.1.0](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.1.0): fix gif duration, add loop number, remove auto-download, remove xformers, remove instructions on gradio UI, refactor README, add [sponsor](#sponsor) QR code.
- `2023/07/24` [v1.2.0](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.2.0): fix incorrect insertion of motion modules, add option to change path to save motion modules in Settings/AnimateDiff, fix loading different motion modules.
- `2023/09/04` [v1.3.0](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.3.0): support any community models with the same architecture; fix grey problem via [#63](https://github.com/continue-revolution/sd-webui-animatediff/issues/63) (credit to [@TDS4874](https://github.com/TDS4874) and [@opparco](https://github.com/opparco))
- `2023/09/11` [v1.4.0](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.4.0): support official v2 motion module (different architecture: GroupNorm not hacked, UNet middle layer has motion module).    
- `2023/09/14`: [v1.4.1](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.4.1): always change `beta`, `alpha_comprod` and `alpha_comprod_prev` to resolve grey problem in other samplers.
- `2023/09/16`: [v1.5.0](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.5.0): randomize init latent to support better img2gif, credit to [this forked repo](https://github.com/talesofai/AnimateDiff); add other output formats and infotext output, credit to [@zappityzap](https://github.com/zappityzap); add appending reversed frames; refactor code to ease maintaining.
- `2023/09/19`: [v1.5.1](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.5.1): support xformers, sdp, sub-quadratic attention optimization - VRAM usage decrease to 5.60GB with default setting. See [FAQ](#faq) 1st item for more information.
- `2023/09/22`: [v1.5.2](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.5.2): option to disable xformers at `Settings/AnimateDiff` [due to a bug in xformers](https://github.com/facebookresearch/xformers/issues/845), API support, option to enable GIF paletter optimization at `Settings/AnimateDiff` (credit to [@rkfg](https://github.com/rkfg)), gifsicle optimization move to `Settings/AnimateDiff`.
- `2023/09/25`: [v1.6.0](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.6.0): [motion LoRA](https://github.com/guoyww/AnimateDiff#features) supported. Download and use them like any other LoRA you use (example: download motion lora to `stable-diffusion-webui/models/Lora` and add `<lora:v2_lora_PanDown:0.8>` to your positive prompt). **Motion LoRA only supports V2 motion modules**.
- `2023/09/27`: [v1.7.0](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.7.0): [ControlNet](https://github.com/Mikubill/sd-webui-controlnet) supported. Please closely follow the instructions in [How to Use](#how-to-use), especially the explanation of `Video source` and `Video path` attributes. ControlNet is way more complex than what I can test and I ask you to test for me. Please submit an issue whenever you find a bug. [Demo and video instructions](#demo-and-video-instructions) are coming soon. Safetensors for some motion modules are also available now. See [model zoo](#motion-module-model-zoo). You may want to check `Do not append detectmap to output` in `Settings/ControlNet` to avoid having a series of control images in your output gallery. You should not change some attributes in your extension UI because they are for infinite v2v, see [WebUI](#webui) for what you should not change.

Infinite V2V, Prompt Travel and other CLI features are currently work in progress inside [#121](https://github.com/continue-revolution/sd-webui-animatediff/pull/121). Stay tuned and they should be released within a week.

## FAQ
1.  Q: How much VRAM do I need?

    A: Actual VRAM usage depends on your image size and video frame number. You can try to reduce image size or video frame number to reduce VRAM usage. I list some data tested on Ubuntu 22.04, NVIDIA 4090, torch 2.0.1+cu117, H=W=512, frame=16 (default setting):
    | Optimization | VRAM usage |
    | --- | --- |
    | No optimization | 12.13GB |
    | xformers/sdp | 5.60GB |
    | sub-quadratic | 10.39GB |

1.  Q: Can I use SDXL to generate GIFs?

    A: You will have to wait for someone to train SDXL-specific motion modules which will have a different model architecture. This extension essentially inject multiple motion modules into SD1.5 UNet. It does not work for other variations of SD, such as SD2.1 and SDXL.

1.  Q: Can I override the limitation of 24/32 frames per generation?

    A: Not at this time, but will be supported via supporting [AnimateDIFF CLI Prompt Travel](https://github.com/s9roll7/animatediff-cli-prompt-travel) in the near future. This is a huge amount of work and life is busy, so expect to wait for a long time before updating.


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

| ----------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| ![216aff0250c7fd2bb32eeb4f7aae623](https://user-images.githubusercontent.com/63914308/232824466-21051be9-76ce-4862-bb0d-a431c186fce1.jpg) | ![15fe95b4ada738acf3e44c1d45a1805](https://user-images.githubusercontent.com/63914308/232824545-fb108600-729d-4204-8bec-4fd5cc8a14ec.jpg) | ![IMG_1419_](https://github.com/continue-revolution/sd-webui-animatediff/assets/63914308/eaa7b114-a2e6-4ecc-a29f-253ace06d1ea) |
| WeChat | AliPay | Paypal |
| --- | --- | --- |
| ![216aff0250c7fd2bb32eeb4f7aae623](https://user-images.githubusercontent.com/63914308/232824466-21051be9-76ce-4862-bb0d-a431c186fce1.jpg) | ![15fe95b4ada738acf3e44c1d45a1805](https://user-images.githubusercontent.com/63914308/232824545-fb108600-729d-4204-8bec-4fd5cc8a14ec.jpg) | ![IMG_1419_](https://github.com/continue-revolution/sd-webui-animatediff/assets/63914308/eaa7b114-a2e6-4ecc-a29f-253ace06d1ea) |