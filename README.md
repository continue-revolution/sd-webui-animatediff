# AnimateDiff for Stable Diffusion Webui

This extension aim for integrating [AnimateDiff](https://github.com/guoyww/AnimateDiff/) into [AUTOMATIC1111 Stable Diffusion Webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui).

This extension implements AnimateDiff in a different way. It does not require you to clone the whole SD1.5 repository. It also applied (probably) the least modification to `ldm`, so that you do not need to reload your model weights if you don't want to.

It essentially inject multiple motion modules into SD1.5 UNet. It does not work for other variations of SD, such as SD2.1 and SDXL.

Batch size on WebUI will be replaced by GIF frame number: 1 full GIF generated in 1 batch. If you want to generate multiple GIF at once, please change batch number.

You can try txt2gif on txt2img panel, img2gif on img2img panel with any LoRA/ControlNet. Due to the 1-batch behavior of AnimateDiff, it is probably not possible to support gif2gif. However, I need to discuss this with the authors of AnimateDiff.

Motion modules will be auto-downloaded from [here](https://drive.google.com/drive/folders1EqLC65eR1-W-sGD0Im7fkED6c8GkiNFI). If your terminal cannot access google due to whatever reason, please manually download andput to sd-webui-animatediff/model/. DO NOT change model filename.

You might also be interested in another extension I created: [Segment Anything for Stable Diffusion WebUI](https://github.com/continue-revolution/sd-webui-segment-anything).

## FAQ
1.  Q: Can I reproduce the result created by the orinal authors?

    A: Unfortunately, you cannot. This is because A1111 implements generation of random tensors in a completely different way. It is not possible to produce exactly the same random tensors as the original authors without an extremely large code modification.
2.  Q: I am using a remote server which blocks Google. What should I do?

    A: You will have to find a way to download motion modules locally and re-upload to your server. At this time, the motion modules are not available on huggingface, which some GPU leasers did provide some proxy access. I provide a [baidu netdisk link](https://pan.baidu.com/s/18ZpcSM6poBqxWNHtnyMcxg?pwd=et8y).
3.  Q: Can I generate a video instead a GIF? How much VRAM do I need?

    A: Currently, you can run webui with this extension via NVIDIA 3090. I cannot guarantee any other variations of GPU. You cannot generate a video. This is because a whole batch of images will pass through a transformer module, which prevents us from generating videos sequentially. We look forward to future developments of deep learning for video generation.
