# AnimateDiff for Stable Diffusion Webui

This extension aim for integrating [AnimateDiff](https://github.com/guoyww/AnimateDiff/) into [AUTOMATIC1111 Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui). I have tested this extension with WebUI v1.4.1 on Ubuntu 20.04 with NVIDIA 3090. You can generate GIFs in exactly the same way as generating images after enabling this extension.

This extension implements AnimateDiff in a different way. It does not require you to clone the whole SD1.5 repository. It also applied (probably) the least modification to `ldm`, so that you do not need to reload your model weights if you don't want to.

Batch size on WebUI will be replaced by GIF frame number: 1 full GIF generated in 1 batch. If you want to generate multiple GIF at once, please change batch number.

You might also be interested in another extension I created: [Segment Anything for Stable Diffusion WebUI](https://github.com/continue-revolution/sd-webui-segment-anything).

## How to Use

1. Install this extension via link.
2. Download motion modules from [Google Drive](https://drive.google.com/drive/folders/1EqLC65eR1-W-sGD0Im7fkED6c8GkiNFI) | [HuggingFace](https://huggingface.co/guoyww/animatediff) | [CivitAI](https://civitai.com/models/108836) | [Baidu NetDisk](https://pan.baidu.com/s/18ZpcSM6poBqxWNHtnyMcxg?pwd=et8y). You only need to download one of `mm_sd_v14.ckpt` | `mm_sd_v15.ckpt`. Put the model weights under `sd-webui-animatediff/model/`. DO NOT change model filename.
3. Go to txt2img if you want to try txt2gif and img2img if you want to try img2gif.
4. Choose an SD1.5 checkpoint, write prompts, set configurations such as image width/height. If you want to generate multiple GIFs at once, please change batch number, instead of batch size.
5. Enable AnimateDiff extension, set up each parameter (loop number means how many loop the GIF will be displayed, not the actual length of the GIF) and click `Generate`. 
5. You should see the output GIF on the output gallery. You can access GIF output at `stable-diffusion-webui/outputs/{txt2img or img2img}-images/AnimateDiff`. You can also access image frames at `stable-diffusion-webui/outputs/{txt2img or img2img}-images/{date}`.

## Update

- `2023/07/20` [v1.1.0](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.1.0): fix gif duration, add loop number, remove auto-download, remove xformers, remove instructions on gradio UI, refactor README, add [sponsor](#sponsor) QR code.
- `2023/07/24` [v1.2.0](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.2.0): fix incorrect insertion of motion modules, add option to change path to save motion modules in Settings/AnimateDiff, fix loading different motion modules.

## TODO
- [ ] other attention optimization (e.g. sdp)
- [ ] [token](https://github.com/continue-revolution/sd-webui-animatediff/issues/4)
- [ ] [shape](https://github.com/continue-revolution/sd-webui-animatediff/issues/3)
- [ ] [reddit](https://www.reddit.com/r/StableDiffusion/comments/152n2cr/a1111_extension_of_animatediff_is_available/?sort=new)
- [ ] img2img
- [ ] greyer sample

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

7.  Q: This extension perform worse than [AnimateDiff](https://github.com/guoyww/AnimateDiff/). There seem to be no motion but only glitches. Why?

    A: Because I inserted motion modules to the wrong place inside UNet output blocks. It is a very idiot typo (I wrote a 2 where it was supposed to be 3) but took me days to discover.

8.  Q: How can I reproduce the result in [Samples/txt2img](#txt2img) section?

    A: You must replace [create_random_tensors](https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/processing.py#L461-L519) with 
    ```python
        torch.manual_seed(<seed>)
        from einops import rearrange
        x = rearrange(torch.randn((4, 16, 64, 64), device=shared.device), 'c f h w -> f c h w')
    ```
    and retry. A1111 generate random tensors in a completely different way.

9. Q: [v1.2.0](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.2.0) does not work for img2img. Why?

    A: I don't know. I will try to figure out why very soon.

10. Q: [v1.2.0](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.2.0) seems to give a greyer sample compared to [AnimateDiff](https://github.com/guoyww/AnimateDiff/). Why?

    A: I don't know. I will try to figure out why very soon.

## Samples

### txt2img
| AnimateDiff | A1111 |
| --- | --- |
| ![image](https://user-images.githubusercontent.com/63914308/255306527-5105afe8-d497-4ab1-b5c4-37540e9601f8.gif) | ![00023-10788741199826055168](https://github.com/continue-revolution/sd-webui-animatediff/assets/63914308/c35a952a-a127-491b-876d-cda97771f7ee) |

### img2img
[v1.2.0](https://github.com/continue-revolution/sd-webui-animatediff/releases/tag/v1.2.0) does not work for img2img due to some unknown reason. Will be fixed later.

## Sponsor
You can sponsor me via WeChat or Alipay.

| WeChat | Alipay |
| --- | --- |
| ![216aff0250c7fd2bb32eeb4f7aae623](https://user-images.githubusercontent.com/63914308/232824466-21051be9-76ce-4862-bb0d-a431c186fce1.jpg) | ![15fe95b4ada738acf3e44c1d45a1805](https://user-images.githubusercontent.com/63914308/232824545-fb108600-729d-4204-8bec-4fd5cc8a14ec.jpg) |
