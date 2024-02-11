# How to Use

## Preparation

### You have A1111 and you know Git
If you have already had OG A1111 and you are familiar with git, I highly recommend running the following commands in your terminal in `/path/to/stable-diffusion-webui`
```bash
git remote add forge https://github.com/lllyasviel/stable-diffusion-webui-forge
git branch lllyasviel/main
git branch -u forge/main
git pull
```

Then you can `cd extensions/sd-webui-animatediff` and `git checkout forge/master`

### You don't have A1111 or you don't know Git
Otherwise, download zip package from [here](https://github.com/lllyasviel/stable-diffusion-webui-forge?tab=readme-ov-file#installing-forge). I believe for users in P.R.China, [Akegarasu](https://github.com/Akegarasu) will soon provide support for this software in his launcher.

Then install this extension like any other extensions you use, and download motion modules and put the model weights under `stable-diffusion-webui/extensions/sd-webui-animatediff/model/`. If you want to use another directory to save model weights, please go to `Settings/AnimateDiff`. See [model zoo](../README.md#model-zoo) for a list of available motion modules.

## WebUI
1. Go to txt2img if you want to try txt2gif and img2img if you want to try img2gif.
1. Choose an SD checkpoint, write prompts, set configurations such as image width/height. If you want to generate multiple GIFs at once, please [change batch number, instead of batch size](performance.md#batch-size).
1. Enable AnimateDiff extension, set up [each parameter](#parameters), then click `Generate`.
1. You should see the output GIF on the output gallery. You can access GIF output at `stable-diffusion-webui/outputs/{txt2img or img2img}-images/AnimateDiff/{yy-mm-dd}`. You can also access image frames at `stable-diffusion-webui/outputs/{txt2img or img2img}-images/{yy-mm-dd}`. You may choose to save frames for each generation into separate directories in `Settings/AnimateDiff`.

## API
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


## Parameters
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
1. **Video source** — [Optional] Video source file for [ControlNet V2V](features.md#controlnet-v2v). You MUST enable ControlNet. It will be the source control for ALL ControlNet units that you enable without submitting a control image or a path to ControlNet panel. You can of course submit one control image via `Single Image` tab or an input directory via `Batch` tab, which will override this video source input and work as usual.
1. **Video path** — [Optional] Folder for source frames for [ControlNet V2V](features.md#controlnet-v2v), but lower priority than `Video source`. You MUST enable ControlNet. It will be the source control for ALL ControlNet units that you enable without submitting a control image or a path to ControlNet. You can of course submit one control image via `Single Image` tab or an input directory via `Batch` tab, which will override this video path input and work as usual.
    - For people who want to inpaint videos: enter a folder which contains two sub-folders `image` and `mask` on ControlNet inpainting unit. These two sub-folders should contain the same number of images. This extension will match them according to the same sequence. Using my [Segment Anything](https://github.com/continue-revolution/sd-webui-segment-anything) extension can make your life much easier.