# AnimateDiff for Stable Diffusion WebUI

This extension aim for integrating [AnimateDiff](https://github.com/guoyww/AnimateDiff/) into [AUTOMATIC1111 Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui). You can generate GIFs in exactly the same way as generating images after enabling this extension.

This extension implements AnimateDiff in a different way. It does not require you to clone the whole SD1.5 repository. It also applied (probably) the least modification to `ldm`, so that you do not need to reload your model weights if you don't want to.

Batch size on WebUI will be replaced by GIF frame number internally: 1 full GIF generated in 1 batch. If you want to generate multiple GIF at once, please change batch number. 

Batch number is NOT the same as batch size. In A1111 WebUI, batch number is above batch size. Batch number means the number of sequential steps, but batch size means the number of parallel steps. You do not have to worry too much when you increase batch number, but you do need to worry about your VRAM when you increase your batch size (where in this extension, video frame number). You do not need to change batch size at all when you are using this extension.

You might also be interested in another extension I created: [Segment Anything for Stable Diffusion WebUI](https://github.com/continue-revolution/sd-webui-segment-anything).

## How to Use

1. Install this extension via link.
1. Download motion modules and put the model weights under `stable-diffusion-webui/extensions/sd-webui-animatediff/model/`. If you want to use another directory to save the model weights, please go to `Settings/AnimateDiff`. See [model zoo](#motion-module-model-zoo) for a list of available motion modules.
1. Enable `Pad prompt/negative prompt to be same length` and `Batch cond/uncond` and click `Apply settings` in `Settings`. You must do this to prevent generating two separate unrelated GIFs.

### WebUI
1. Go to txt2img if you want to try txt2gif and img2img if you want to try img2gif.
1. Choose an SD1.5 checkpoint, write prompts, set configurations such as image width/height. If you want to generate multiple GIFs at once, please change batch number, instead of batch size.
1. Enable AnimateDiff extension, and set up each parameter, and click `Generate`.
   1. **Number of frames** — The model is trained with 16 frames, so it’ll give the best results when the number of frames is set to `16`.
   1. **Frames per second** — How many frames (images) are shown every second. If 16 frames are generated at 8 frames per second, your GIF’s duration is 2 seconds.
   1. **Loop number** — How many times the GIF is played. A value of `0` means the GIF never stops playing.
1. You should see the output GIF on the output gallery. You can access GIF output at `stable-diffusion-webui/outputs/{txt2img or img2img}-images/AnimateDiff`. You can also access image frames at `stable-diffusion-webui/outputs/{txt2img or img2img}-images/{date}`.

### API
[#42](https://github.com/continue-revolution/sd-webui-animatediff/issues/42)

## Motion Module Model Zoo
- `mm_sd_v14.ckpt` & `mm_sd_v15.ckpt` & `mm_sd_v15_v2.ckpt` by [@guoyww](https://github.com/guoyww): [Google Drive](https://drive.google.com/drive/folders/1EqLC65eR1-W-sGD0Im7fkED6c8GkiNFI) | [HuggingFace](https://huggingface.co/guoyww/animatediff) | [CivitAI](https://civitai.com/models/108836) | [Baidu NetDisk](https://pan.baidu.com/s/18ZpcSM6poBqxWNHtnyMcxg?pwd=et8y)
- `mm-Stabilized_high.pth` & `mm-Stabbilized_mid.pth` by [@manshoety](https://huggingface.co/manshoety): [HuggingFace](https://huggingface.co/manshoety/AD_Stabilized_Motion/tree/main)
- `temporaldiff-v1-animatediff.ckpt` by [@CiaraRowles](https://huggingface.co/CiaraRowles): [HuggingFace](https://huggingface.co/CiaraRowles/TemporalDiff/tree/main)

## Update

- `2023/07/20` [v1.1.0](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.1.0): fix gif duration, add loop number, remove auto-download, remove xformers, remove instructions on gradio UI, refactor README, add [sponsor](#sponsor) QR code.
- `2023/07/24` [v1.2.0](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.2.0): fix incorrect insertion of motion modules, add option to change path to save motion modules in Settings/AnimateDiff, fix loading different motion modules.
- `2023/09/04` [v1.3.0](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.3.0): support any community models with the same architecture; fix grey problem via [#63](https://github.com/continue-revolution/sd-webui-animatediff/issues/63) (credit to [@TDS4874](https://github.com/TDS4874) and [@opparco](https://github.com/opparco))
- `2023/09/11` [v1.4.0](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.4.0): support official v2 motion module (different architecture: GroupNorm not hacked, UNet middle layer has motion module).    
    - If you are using V1 motion modules: starting from this version, you will be able to disable hacking GroupNorm in `Settings/AnimateDiff`. If you disable hacking GruopNorm, you will be able to use this extension in `img2img` in all settings, but the generated GIFs will have flickers. In WebUI >=v1.6.0, even if GroupNorm is hacked, you can still use this extension in `img2img` with `--no-half-vae` enabled.
    - If you are using V2 motion modules: you will always be able to use this extension in `img2img`, regardless of changing that setting or not.
- `2023/09/14`: [v1.4.1](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.4.1): always change `beta`, `alpha_comprod` and `alpha_comprod_prev` to resolve grey problem in other samplers.

## TODO
This TODO list will most likely be resolved sequentially.
- [ ] other attention optimization (e.g. sdp)
- [ ] [shape](https://github.com/continue-revolution/sd-webui-animatediff/issues/3)
- [ ] [reddit](https://www.reddit.com/r/StableDiffusion/comments/152n2cr/a1111_extension_of_animatediff_is_available/?sort=new)

## FAQ
1.  Q: I am using a remote server which blocks Google. What should I do?

    A: You will have to find a way to download motion modules locally and re-upload to your server.

2.  Q: How much VRAM do I need?

    A: Currently, you can run WebUI with this extension via NVIDIA 3090. I cannot guarantee any other variations of GPU. Actual VRAM usage depends on your image size and video frame number. You can try to reduce image size or video frame number to reduce VRAM usage. The default setting (displayed in [Samples/txt2img](#txt2img) section) consumes 12GB VRAM. More VRAM info will be added later.

3.  Q: Can I generate a video instead a GIF?

    A: Unfortunately, you cannot. This is because a whole batch of images will pass through a transformer module, which prevents us from generating videos sequentially. We look forward to future developments of deep learning for video generation.

4.  Q: Can I use SDXL to generate GIFs?

    A: At least at this time, you cannot. This extension essentially inject multiple motion modules into SD1.5 UNet. It does not work for other variations of SD, such as SD2.1 and SDXL. I'm not sure what will happen if you force-add motion modules to SD2.1 or SDXL. Future experiments are needed.

5.  Q: Can I use this extension to do gif2gif?

    A: Due to the 1-batch behavior of AnimateDiff, it is probably not possible to support gif2gif. However, I need to discuss this with the authors of AnimateDiff.

6.  Q: Can I use xformers?

    A: Yes, but it will not be applied to AnimateDiff due to [a weird bug](https://github.com/continue-revolution/sd-webui-animatediff/issues/2). I will try other optimizations. Note that xformers will change the GIF you generate.

7.  Q: How can I reproduce the result in [Samples/txt2img](#txt2img) section?

    A: You must use this logic to initialize random tensors:
    ```python
        torch.manual_seed(<seed>)
        from einops import rearrange
        x = rearrange(torch.randn((4, 16, 64, 64), device=shared.device), 'c f h w -> f c h w')
    ```


## Samples

### txt2img
| AnimateDiff | A1111 v1.2.0 | A1111 v1.3.0 |
| --- | --- | --- |
| ![image](https://user-images.githubusercontent.com/63914308/255306527-5105afe8-d497-4ab1-b5c4-37540e9601f8.gif) | ![00023-10788741199826055168](https://github.com/continue-revolution/sd-webui-animatediff/assets/63914308/c35a952a-a127-491b-876d-cda97771f7ee) | ![00013-10788741199826055000](https://github.com/continue-revolution/sd-webui-animatediff/assets/63914308/43b9cf34-dbd1-4120-b220-ea8cb7882272) |

Note that I did not modify random tensor generation when producing v1.3.0 samples.


## Sponsor
You can sponsor me via WeChat or Alipay.

| WeChat | Alipay |
| --- | --- |
| ![216aff0250c7fd2bb32eeb4f7aae623](https://user-images.githubusercontent.com/63914308/232824466-21051be9-76ce-4862-bb0d-a431c186fce1.jpg) | ![15fe95b4ada738acf3e44c1d45a1805](https://user-images.githubusercontent.com/63914308/232824545-fb108600-729d-4204-8bec-4fd5cc8a14ec.jpg) |
