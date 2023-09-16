import os
import gc
import gradio as gr
import imageio
import torch
from einops import rearrange

from modules import scripts, images, shared, script_callbacks, hashes
from modules.devices import torch_gc, device, cpu
from modules.processing import StableDiffusionProcessing, Processed
from modules.images import FilenameGenerator
from scripts.logging_animatediff import logger_animatediff
from motion_module import MotionWrapper, VanillaTemporalModule


from ldm.modules.diffusionmodules.openaimodel import TimestepBlock, TimestepEmbedSequential
from ldm.modules.diffusionmodules.util import GroupNorm32
from ldm.modules.attention import SpatialTransformer
def mm_tes_forward(self, x, emb, context=None):
    for layer in self:
        if isinstance(layer, TimestepBlock):
            x = layer(x, emb)
        elif isinstance(layer, (SpatialTransformer, VanillaTemporalModule)):
            x = layer(x, context)
        else:
            x = layer(x)
    return x
script_dir = scripts.basedir()
gn32_original_forward = GroupNorm32.forward
tes_original_forward = TimestepEmbedSequential.forward


class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)

    def get_block_name(self):
        return "button"


class AnimateDiffScript(scripts.Script):
    motion_module: MotionWrapper = None

    def __init__(self):
        self.logger = logger_animatediff
        self.prev_beta = None
        self.prev_alpha_cumprod = None
        self.prev_alpha_cumprod_prev = None

    def title(self):
        return "AnimateDiff"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def unload_motion_module(self):
        self.logger.info("Moving motion module to CPU")
        if AnimateDiffScript.motion_module is not None:
            AnimateDiffScript.motion_module.to(cpu)
        torch_gc()
        gc.collect()

    def remove_motion_module(self):
        self.logger.info("Removing motion module from any memory")
        del AnimateDiffScript.motion_module
        AnimateDiffScript.motion_module = None
        torch_gc()
        gc.collect()

    def ui(self, is_img2img):
        model_dir = shared.opts.data.get("animatediff_model_path", os.path.join(script_dir, "model"))
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        model_list = [f for f in os.listdir(model_dir) if f != ".gitkeep"]
        with gr.Accordion('AnimateDiff', open=False):
            with gr.Row():
                enable = gr.Checkbox(value=False, label='Enable AnimateDiff')
                model = gr.Dropdown(choices=["mm_sd_v15.ckpt", "mm_sd_v14.ckpt"], value="mm_sd_v14.ckpt", label="Motion module", type="value")
            with gr.Row():
                with gr.Column(scale=1):
                    video_length = gr.Slider(minimum=1, maximum=24, value=16, step=1, label="Number of frames", precision=0)
                    fps = gr.Slider(minimum=1, maximum=30, value=8, step=1, label="Frames per second (FPS)", precision=0)
                    loop_number = gr.Slider(minimum=0, maximum=64, value=0, step=1, label="Display loop number (0 = infinite loop)", precision=0)
                with gr.Column(scale=1):
                    save_png = gr.Checkbox(value=True, label='Save PNG')
                    save_gif = gr.Checkbox(value=True, label='Save GIF')
                    save_mp4 = gr.Checkbox(value=True, label='Save MP4')
                    save_txt = gr.Checkbox(value=True, label='Save TXT')
            with gr.Row():
                optimize_gif = gr.Checkbox(value=False, label='Optimize GIF (requires pygifsicle and gifsicle)')
            with gr.Row():
                unload = gr.Button(value="Move motion module to CPU (default if lowvram)")
                remove = gr.Button(value="Remove motion module from any memory")
                unload.click(fn=self.unload_motion_module)
                remove.click(fn=self.remove_motion_module)
        return enable, save_png, save_gif, save_mp4, save_txt, optimize_gif, loop_number, video_length, fps, model

    def inject_motion_modules(self, p: StableDiffusionProcessing, model_name="mm_sd_v14.ckpt"):
        model_path = os.path.join(shared.opts.data.get("animatediff_model_path", os.path.join(script_dir, "model")), model_name)
        if not os.path.isfile(model_path):
            raise RuntimeError("Please download models manually.")
        def get_mm_hash(model_name="mm_sd_v15.ckpt"):
            model_hash = hashes.sha256(model_path, f"AnimateDiff/{model_name}")
            using_v2 = False
            if model_hash == 'aa7fd8a200a89031edd84487e2a757c5315460eca528fa70d4b3885c399bffd5':
                self.logger.info('You are using mm_sd_14.ckpt, which has been tested and supported.')
            elif model_hash == "cf16ea656cb16124990c8e2c70a29c793f9841f3a2223073fac8bd89ebd9b69a":
                self.logger.info('You are using mm_sd_15.ckpt, which has been tested and supported.')
            elif model_hash == "0aaf157b9c51a0ae07cb5d9ea7c51299f07bddc6f52025e1f9bb81cd763631df":
                self.logger.info('You are using mm-Stabilized_high.pth, which has been tested and supported.')
            elif model_hash == '39de8b71b1c09f10f4602f5d585d82771a60d3cf282ba90215993e06afdfe875':
                self.logger.info('You are using mm-Stabilized_mid.pth, which has been tested and supported.')
            elif model_hash == '3cb569f7ce3dc6a10aa8438e666265cb9be3120d8f205de6a456acf46b6c99f4':
                self.logger.info('You are using temporaldiff-v1-animatediff.ckpt, which has been tested and supported.')
            elif model_hash == '69ed0f5fef82b110aca51bcab73b21104242bc65d6ab4b8b2a2a94d31cad1bf0':
                self.logger.info('You are using mm_sd_v15_v2.ckpt, which has been tested and supported.')
                using_v2 = True
            else:
                self.logger.warn(f"Your model {model_name} has not been tested and supported. "
                                 "Either your download is incomplete or your model has not been tested. "
                                 "Please use at your own risk.")
            return model_hash, using_v2
        model_hash, using_v2 = get_mm_hash(model_name)
        if AnimateDiffScript.motion_module is None or AnimateDiffScript.motion_module.mm_hash != model_hash:
            self.logger.info(f"Loading motion module {model_name} from {model_path}")
            mm_state_dict = torch.load(model_path, map_location=device)
            AnimateDiffScript.motion_module = MotionWrapper(model_hash, using_v2)
            missed_keys = AnimateDiffScript.motion_module.load_state_dict(mm_state_dict)
            self.logger.warn(f"Missing keys {missed_keys}")
        AnimateDiffScript.motion_module.to(device)
        if not shared.cmd_opts.no_half:
            AnimateDiffScript.motion_module.half()
        unet = p.sd_model.model.diffusion_model
        TimestepEmbedSequential.forward = mm_tes_forward
        if shared.opts.data.get("animatediff_hack_gn", False) and (not AnimateDiffScript.motion_module.using_v2):
            self.logger.info(f"Hacking GroupNorm32 forward function.")
            def groupnorm32_mm_forward(self, x):
                x = rearrange(x, '(b f) c h w -> b c f h w', b=2)
                x = gn32_original_forward(self, x)
                x = rearrange(x, 'b c f h w -> (b f) c h w', b=2)
                return x
            GroupNorm32.forward = groupnorm32_mm_forward
        self.logger.info(f"Injecting motion module {model_name} into SD1.5 UNet input blocks.")
        for mm_idx, unet_idx in enumerate([1, 2, 4, 5, 7, 8, 10, 11]):
            mm_idx0, mm_idx1 = mm_idx // 2, mm_idx % 2
            unet.input_blocks[unet_idx].append(AnimateDiffScript.motion_module.down_blocks[mm_idx0].motion_modules[mm_idx1])
        self.logger.info(f"Injecting motion module {model_name} into SD1.5 UNet output blocks.")
        for unet_idx in range(12):
            mm_idx0, mm_idx1 = unet_idx // 3, unet_idx % 3
            if unet_idx % 3 == 2 and unet_idx != 11:
                unet.output_blocks[unet_idx].insert(-1, AnimateDiffScript.motion_module.up_blocks[mm_idx0].motion_modules[mm_idx1])
            else:
                unet.output_blocks[unet_idx].append(AnimateDiffScript.motion_module.up_blocks[mm_idx0].motion_modules[mm_idx1])
        if using_v2:
            self.logger.info(f"Injecting motion module {model_name} into SD1.5 UNet middle block.")
            unet.middle_block.insert(-1, AnimateDiffScript.motion_module.mid_block.motion_modules[0])
        self.logger.info(f"Injection finished.")

    def remove_motion_modules(self, p: StableDiffusionProcessing):
        unet = p.sd_model.model.diffusion_model
        self.logger.info(f"Removing motion module from SD1.5 UNet input blocks.")
        for unet_idx in [1, 2, 4, 5, 7, 8, 10, 11]:
            unet.input_blocks[unet_idx].pop(-1)
        self.logger.info(f"Removing motion module from SD1.5 UNet output blocks.")
        for unet_idx in range(12):
            if unet_idx % 3 == 2 and unet_idx != 11:
                unet.output_blocks[unet_idx].pop(-2)
            else:
                unet.output_blocks[unet_idx].pop(-1)
        if AnimateDiffScript.motion_module.using_v2:
            self.logger.info(f"Removing motion module from SD1.5 UNet middle block.")
            unet.middle_block.pop(-2)
        if shared.opts.data.get("animatediff_hack_gn", False) and (not AnimateDiffScript.motion_module.using_v2):
            self.logger.info(f"Restoring GroupNorm32 forward function.")
            GroupNorm32.forward = gn32_original_forward
        TimestepEmbedSequential.forward = tes_original_forward 
        self.logger.info(f"Removal finished.")
        if shared.cmd_opts.lowvram:
            self.unload_motion_module()
    
    def set_ddim_alpha(self, p: StableDiffusionProcessing):
        self.logger.info(f"Setting DDIM alpha.")
        beta_start = 0.00085
        beta_end = 0.012
        betas = torch.linspace(beta_start, beta_end, p.sd_model.num_timesteps, dtype=torch.float32, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            (torch.tensor([1.0], dtype=torch.float32, device=device), alphas_cumprod[:-1]))
        self.prev_beta = p.sd_model.betas
        p.sd_model.betas = betas
        self.prev_alpha_cumprod = p.sd_model.alphas_cumprod
        p.sd_model.alphas_cumprod = alphas_cumprod
        self.prev_alpha_cumprod_prev = p.sd_model.alphas_cumprod_prev
        p.sd_model.alphas_cumprod_prev = alphas_cumprod_prev
    
    def restore_ddim_alpha(self, p: StableDiffusionProcessing):
        self.logger.info(f"Restoring DDIM alpha.")
        p.sd_model.betas = self.prev_beta
        p.sd_model.alphas_cumprod = self.prev_alpha_cumprod
        p.sd_model.alphas_cumprod_prev = self.prev_alpha_cumprod_prev
        self.prev_beta = None
        self.prev_alpha_cumprod = None
        self.prev_alpha_cumprod_prev = None

    def before_process(self,
        p: StableDiffusionProcessing,
        enable_animatediff=False,
        save_png=True,
        save_gif=True,
        save_mp4=True,
        save_txt=True,
        optimize_gif=False,
        loop_number=0,
        video_length=16,
        fps=8,
        model="mm_sd_v14.ckpt"):
        if enable_animatediff:
            # create output directory for PNG, the others will be created as needed
            png_path = os.path.join(p.outpath_samples, "animatediff", "png")
            os.makedirs(png_path, exist_ok=True)

            # override save location for PNG frames
            p.outpath_samples = png_path
            
            if not save_png:
                p.do_not_save_samples = True

            assert save_png or save_gif or save_mp4, "No save option selected!"
            assert video_length > 0 and fps > 0, "Video length and FPS should be positive."

            self.logger.info(f"AnimateDiff process start with video Max frames {video_length}, FPS {fps}, duration {video_length/fps}, motion module {model}.")
            p.batch_size = video_length
            self.inject_motion_modules(p, model)
            self.set_ddim_alpha(p)

    def postprocess(self,
        p: StableDiffusionProcessing,
        res: Processed,
        enable_animatediff=False,
        save_png=True,
        save_gif=True,
        save_mp4=True,
        save_txt=True,
        optimize_gif=False,
        loop_number=0,
        video_length=16,
        fps=8,
        model="mm_sd_v14.ckpt"):
        if enable_animatediff:
            self.restore_ddim_alpha(p)
            self.remove_motion_modules(p)
            video_paths = []
            for i in range(res.index_of_first_image, len(res.images), video_length):
                video_list = res.images[i:i+video_length]

                seq = images.get_next_sequence_number(f"{p.outpath_samples}", "")
                filename = f"{seq:05}-{res.seed}"

                namegen = FilenameGenerator(p, res.seed, res.prompt, res.images[i])
                dirname = namegen.apply(shared.opts.directories_filename_pattern or "[prompt_words]").lstrip(' ').rstrip('\\ /')
                path = os.path.join(f"{p.outpath_samples}", dirname)
                os.makedirs(path, exist_ok=True)

                if save_gif:
                    gif_path = os.path.join(p.outpath_samples, "..", "gif", dirname)
                    os.makedirs(gif_path, exist_ok=True)
                    video_path = os.path.join(gif_path, f"{filename}.gif")
                    video_paths.append(video_path)
                    imageio.mimsave(video_path, video_list, duration=(1/fps), loop=loop_number)
                    if optimize_gif:
                        try:
                            import pygifsicle
                        except ImportError:
                            self.logger.warn("pygifsicle failed to import, required for optimized GIFs, try: pip install pygifsicle")
                        else:
                            try:
                                pygifsicle.optimize(video_path)
                            except FileNotFoundError:
                                self.logger.warn("gifsicle not found, required for optimized GIFs, try: apt install gifsicle")
                if save_mp4:
                    mp4_path = os.path.join(p.outpath_samples, "..", "mp4", dirname)
                    os.makedirs(mp4_path, exist_ok=True)
                    video_path = os.path.join(mp4_path, f"{filename}.mp4")
                    imageio.mimsave(video_path, video_list, fps=fps)
                if save_txt and res.images[i].info is not None:
                    res.images[i].info['motion_module'] = model
                    res.images[i].info['video_length'] = video_length
                    res.images[i].info['fps'] = fps
                    res.images[i].info['loop_number'] = 0
                    txt_path = os.path.join(p.outpath_samples, "..", "txt", dirname)
                    os.makedirs(txt_path, exist_ok=True)
                    file_path = os.path.join(txt_path, f"{filename}.txt")
                    with open(file_path, "w", encoding="utf8") as file:
                        file.write(f"{res.images[i].info}\n")

            res.images = video_paths
            self.logger.info("AnimateDiff process end.")

def on_ui_settings():
    section = ('animatediff', "AnimateDiff")
    shared.opts.add_option("animatediff_model_path", shared.OptionInfo(os.path.join(script_dir, "model"), "Path to save AnimateDiff motion modules", gr.Textbox, section=section))
    shared.opts.add_option("animatediff_hack_gn", shared.OptionInfo(
        True, "Check if you want to hack GroupNorm. By default, V1 hacks GroupNorm, which avoids a performance degradation. "
        "If you choose not to hack GroupNorm for V1, you will be able to use this extension in img2img in all cases, but the generated GIF will have flickers. "
        "V2 does not hack GroupNorm, so that this option will not influence v2 inference.", gr.Checkbox, section=section))


script_callbacks.on_ui_settings(on_ui_settings)
