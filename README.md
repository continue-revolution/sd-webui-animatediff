# AnimateDiff for Stable Diffusion WebUI Forge
This branch is specifically designed for [Stable Diffusion WebUI Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge) by lllyasviel. See [here](docs/how-to-use.md#preparation) for how to install forge and this extension. See [Update](#update) for current status.


This extension aim for integrating [AnimateDiff](https://github.com/guoyww/AnimateDiff/) w/ [CLI](https://github.com/s9roll7/animatediff-cli-prompt-travel) into [lllyasviel's Forge Adaption of AUTOMATIC1111 Stable Diffusion WebUI](https://github.com/lllyasviel/stable-diffusion-webui-forge) and form the most easy-to-use AI video toolkit. You can generate GIFs in exactly the same way as generating images after enabling this extension.

This extension implements AnimateDiff in a different way. It makes heavy use of [Unet Patcher](https://github.com/lllyasviel/stable-diffusion-webui-forge?tab=readme-ov-file#unet-patcher), so that you do not need to reload your model weights if you don't want to, and I can almostly get rif of monkey-patching WebUI and ControlNet.

You might also be interested in another extension I created: [Segment Anything for Stable Diffusion WebUI](https://github.com/continue-revolution/sd-webui-segment-anything). This extension will also be redesigned for forge later.


[TusiArt](https://tusiart.com/) (for users physically inside P.R.China mainland) and [TensorArt](https://tusiart.com/) (for others) offers online service of this extension.


## Table of Contents
[Update](#update) | [TODO](#todo) | [Model Zoo](#model-zoo) | [Documentation](#documentation) | [Tutorial](#tutorial) | [Thanks](#thanks) | [Star History](#star-history) | [Sponsor](#sponsor)


## Update
- [v2.0.0-f](https://github.com/continue-revolution/sd-webui-animatediff/tree/v2.0.0-f) in `02/05/2023`: txt2img, prompt travel, infinite generation, all kinds of optimizations have been proven to be working properly and elegantly.
- [v2.0.1-f](https://github.com/continue-revolution/sd-webui-animatediff/tree/v2.0.1-f) in `02/09/2023`: ControlNet V2V in txt2img panel is working properly and elegantly. You can also try adding mask and inpaint.


## TODO
- [ ] MotionLoRA and i2i batch are still under heavy construction, but I expect to release a working version soon in a week.
- [ ] When all previous features are working properly, I will soon release SparseCtrl, Magic Animate and Moore Animate Anyone.
- [ ] An official video tutorial will be available on YouTube and bilibili.
- [ ] A bunch of new models / advanced parameters / new features may be implented soon.
- [ ] All problems in master branch will be fixed soon, but new feature updates for OG A1111 + Mikubill ControlNet extension may be postponded to some time when I have time to rewrite ControlNet extension.


## Model Zoo
I am maintaining a [huggingface repo](https://huggingface.co/conrevo/AnimateDiff-A1111/tree/main) to provide all official models in fp16 & safetensors format. You are highly recommended to use my link. You MUST use my link to download adapter for V3. You may still use the old links if you want, for all models except adapter for V3.

- "Official" models by [@guoyww](https://github.com/guoyww): [Google Drive](https://drive.google.com/drive/folders/1EqLC65eR1-W-sGD0Im7fkED6c8GkiNFI) | [HuggingFace](https://huggingface.co/guoyww/animatediff/tree/main) | [CivitAI](https://civitai.com/models/108836)
- "Stabilized" community models by [@manshoety](https://huggingface.co/manshoety): [HuggingFace](https://huggingface.co/manshoety/AD_Stabilized_Motion/tree/main)
- "TemporalDiff" models by [@CiaraRowles](https://huggingface.co/CiaraRowles): [HuggingFace](https://huggingface.co/CiaraRowles/TemporalDiff/tree/main)
- "HotShotXL" models by [@hotshotco](https://huggingface.co/hotshotco/): [HuggingFace](https://huggingface.co/hotshotco/Hotshot-XL/tree/main)


## Documentation
- [How to Use](docs/how-to-use.md) -> [Preparation](docs/how-to-use.md#preparation) | [WebUI](docs/how-to-use.md#webui) | [API](docs/how-to-use.md#api) | [Parameters](docs/how-to-use.md#parameters)
- [Features](docs/features.md) -> [Img2Vid](docs/features.md#img2vid) | [Prompt Travel](docs/features.md#prompt-travel) | [ControlNet V2V](docs/features.md#controlnet-v2v) | [ [Model Spec](docs/features.md#model-spec) -> [Motion LoRA](docs/features.md#motion-lora) | [V3](docs/features.md#v3) | [SDXL](docs/features.md#sdxl) ]
- [Performance](docs/performance.md) -> [ [Optimizations](docs/performance.md#optimizations) -> [Attention](docs/performance.md#attention) | [FP8](docs/performance.md#fp8) | [LCM](docs/performance.md#lcm) ] | [VRAM](docs/performance.md#vram) | [#Batch Size](docs/performance.md#batch-size)
- [Demo](docs/demo.md) -> [Basic Usage](docs/demo.md#basic-usage) | [Motion LoRA](docs/demo.md#motion-lora) | [Prompt Travel](docs/demo.md#prompt-travel) | [AnimateDiff V3](docs/demo.md#animatediff-v3) | [AnimateDiff SDXL](docs/demo.md#animatediff-sdxl) | [ControlNet V2V](docs/demo.md#controlnet-v2v)


## Tutorial 
TODO


## Thanks
I thank researchers from [Shanghai AI Lab](https://www.shlab.org.cn/), especially [@guoyww](https://github.com/guoyww) for creating AnimateDiff. I also thank [@neggles](https://github.com/neggles) and [@s9roll7](https://github.com/s9roll7) for creating and improving [AnimateDiff CLI Prompt Travel](https://github.com/s9roll7/animatediff-cli-prompt-travel). This extension could not be made possible without these creative works.

I also thank community developers, especially
- [@zappityzap](https://github.com/zappityzap) who developed the majority of the [output features](https://github.com/continue-revolution/sd-webui-animatediff/blob/master/scripts/animatediff_output.py)
- [@TDS4874](https://github.com/TDS4874) and [@opparco](https://github.com/opparco) for resolving the grey issue which significantly improve the performance
- [@lllyasviel](https://github.com/lllyasviel) for offering forge technical support

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
You can sponsor me via WeChat, AliPay or [PayPal](https://paypal.me/conrevo). You can also support me via [ko-fi](https://ko-fi.com/conrevo) or [afdian](https://afdian.net/a/conrevo).

| WeChat | AliPay | PayPal |
| --- | --- | --- |
| ![216aff0250c7fd2bb32eeb4f7aae623](https://user-images.githubusercontent.com/63914308/232824466-21051be9-76ce-4862-bb0d-a431c186fce1.jpg) | ![15fe95b4ada738acf3e44c1d45a1805](https://user-images.githubusercontent.com/63914308/232824545-fb108600-729d-4204-8bec-4fd5cc8a14ec.jpg) | ![IMG_1419_](https://github.com/continue-revolution/sd-webui-animatediff/assets/63914308/eaa7b114-a2e6-4ecc-a29f-253ace06d1ea) |
