import os
import gc
import gradio as gr
import imageio
import ast
import torch
import piexif
import piexif.helper
from einops import rearrange
from typing import List, Tuple
from pathlib import Path  # Added for directory manipulation

# Modules from your webui
from modules import scripts, images, shared, script_callbacks, hashes
from modules.devices import torch_gc, device, cpu
from modules.processing import StableDiffusionProcessing, Processed

# From AnimateDiff extension
from scripts.logging_animatediff import logger_animatediff
from scripts import unet_injection
from scripts.unet_injection import InjectionParams
from motion_module import MotionWrapper, VanillaTemporalModule

# Modules from ldm
from ldm.modules.diffusionmodules.openaimodel import TimestepBlock, TimestepEmbedSequential
from ldm.modules.diffusionmodules.util import GroupNorm32
from ldm.modules.attention import SpatialTransformer

EXTENSION_DIRECTORY = scripts.basedir()
MODULE_NAME = "AnimateDiff"
DDIM_SAMPLER_NAME = "DDIM"

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
        self.ui_controls = []

    def title(self):
        return MODULE_NAME

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def move_motion_module_to_cpu(self):
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
        
    def get_motion_modules_from_folder(self):
        return [f for f in os.listdir(shared.opts.data.get("animatediff_model_path", "") or 
                                      os.path.join(EXTENSION_DIRECTORY, "model")) if not f.lower().endswith('.gitkeep')]   
        
    def setup_ui_controls(self):
        # Setup UI controls
        with gr.Accordion('AnimateDiff', open=False):
            model_list = self.get_motion_modules_from_folder()
            with gr.Row():
                def refresh_models(*inputs):
                    new_model_list = self.get_motion_modules_from_folder()
                    model_dropdown = inputs[0]
                    if model_dropdown in new_model_list:
                        selected = model_dropdown
                    elif len(new_model_list) > 0:
                        selected = new_model_list[0]
                    else:
                        selected = None
                    return gr.Dropdown.update(choices=new_model_list, value=selected)
                model = gr.Dropdown(choices=model_list, value=(model_list[0] if len(model_list) > 0 else None), label="Motion module", type="value")
                refresh_model = ToolButton(value='\U0001f504')
                refresh_model.click(refresh_models, model, model)
            with gr.Row():
                enable = gr.Checkbox(value=False, label='Enable AnimateDiff')
                video_length = gr.Slider(minimum=1, maximum=32, value=16, step=1, label="Number of frames", precision=0)
                fps = gr.Number(minimum=1, value=8, label="FPS", info= "(Frames per second)", precision=0)
                loop_number = gr.Number(minimum=0, value=0, label="Display loop number", info="(0 = infinite loop)", precision=0)
            with gr.Row():
                move_mm = gr.Button(value="Move motion module to CPU (default if lowvram)")
                remove_mm = gr.Button(value="Remove motion module from any memory")
                move_mm.click(fn=self.move_motion_module_to_cpu)
                remove_mm.click(fn=self.remove_motion_module)
        self.ui_controls = enable, loop_number, video_length, fps, model
        return self.ui_controls
        
    def make_controls_compatible_with_infotext_copy_paste(self, ui_controls = []):
        # Set up controls to be copy-pasted using infotext
        infotext_fields: List[Tuple[gr.components.IOComponent, str]] = []
        paste_field_names: List[str] = []
        for control in ui_controls:
            control_locator = get_control_locator(control.label)
            infotext_fields.append((control, control_locator))
            paste_field_names.append(control_locator)
        self.infotext_fields = infotext_fields
        self.paste_field_names = paste_field_names 
        
        return ui_controls

    def ui(self, is_img2img):
        return self.make_controls_compatible_with_infotext_copy_paste(self.setup_ui_controls())
    
    def get_unet(self, p):
        return p.sd_model.model.diffusion_model
    
    def hack_groupnorm_enabled(self):
        return shared.opts.data.get("animatediff_hack_gn", True) and AnimateDiffScript.motion_module.version == "v1"
    
    def inject_motion_module_to_unet(self, p: StableDiffusionProcessing, injection_params: InjectionParams):               
        unet = self.get_unet(p)
        motion_module = AnimateDiffScript.motion_module
        
        if p.sampler_name == DDIM_SAMPLER_NAME:
            unet_injection.set_ddim_alpha_for_animatediff(p)
        
        if not shared.cmd_opts.no_half:
            AnimateDiffScript.motion_module.half()
        
        if self.hack_groupnorm_enabled():
            unet_injection.hack_groupnorm(injection_params)
        else: # Enforce original groupnorm to avoid bad state from switching hack on and off
            unet_injection.restore_original_groupnorm() 
            
        unet_injection.hack_timestep()
        
        unet_injection.inject_motion_module_to_unet(unet, motion_module, injection_params)
            
    def eject_motion_module_from_unet(self, p: StableDiffusionProcessing):
        unet = self.get_unet(p)
        
        unet_injection.eject_motion_module_from_unet(unet)
            
        if self.hack_groupnorm_enabled():
            unet_injection.restore_original_groupnorm()
            
        unet_injection.restore_original_timestep()
        
        if p.sampler_name == DDIM_SAMPLER_NAME:
            unet_injection.restore_original_ddim_alpha(p)
            
        if shared.cmd_opts.lowvram:
            self.move_motion_module_to_cpu()

    def load_motion_module_and_inject_motion_module_to_unet(
            self, p: StableDiffusionProcessing, injection_params: InjectionParams, model_name="mm_sd_v15.ckpt"):
        model_path = os.path.join(shared.opts.data.get("animatediff_model_path","") or os.path.join(EXTENSION_DIRECTORY, "model"), model_name)
        if not os.path.isfile(model_path):
            raise RuntimeError("Please download models manually.")
        
        model_hash = self.get_and_compare_model_hash(model_path, model_name)
        if AnimateDiffScript.motion_module is None or AnimateDiffScript.motion_module.mm_hash != model_hash:
                
            self.logger.info(f"Loading motion module {model_name} from {model_path}")
            mm_state_dict = torch.load(model_path, map_location=device)
            AnimateDiffScript.motion_module = MotionWrapper(mm_state_dict, model_hash)
            missed_keys = AnimateDiffScript.motion_module.load_state_dict(mm_state_dict)
            self.logger.warn(f"Missing keys {missed_keys}")
            AnimateDiffScript.motion_module.to(device)
        self.inject_motion_module_to_unet(p, injection_params)

    def get_and_compare_model_hash(self, model_path, model_name):
        model_hash = hashes.sha256(model_path, f"AnimateDiff/{model_name}")
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
        else:
            self.logger.warn(f"Your model {model_name} has not been tested and supported. "
                             "Either your download is incomplete or your model has not been tested. "
                             "Please use at your own risk.")
        return model_hash
        
    def serialize_args_to_infotext(self, p: StableDiffusionProcessing):
        # Serialize Animatediff UI controls for infotext and add them to p.extra_generation_params
        # Get the control values from the script_args instead of the controls directly because they don't update for some reason
        control_params = {}   
        control_count = 0
        for control in self.ui_controls:
            control_params[control.label] = p.script_args[self.args_from + control_count]
            control_count += 1
            
        if len(control_params) > 0:
            p.extra_generation_params[MODULE_NAME] = f"{control_params}"

    def before_process(
            self, p: StableDiffusionProcessing, enable_animatediff=False, loop_number=0, video_length=16, fps=8, model="mm_sd_v14.ckpt"):
        if enable_animatediff:
            self.logger.info(f"AnimateDiff process start with video Max frames {video_length}, FPS {fps}, duration {video_length/fps},  motion module {model}.")
            assert video_length > 0 and fps > 0, "Video length and FPS should be positive."
            p.batch_size = video_length
            
            injection_params = InjectionParams(
                video_length=video_length,
                unlimited_area_hack=False,
            )
            self.load_motion_module_and_inject_motion_module_to_unet(p, injection_params, model)
             
            self.serialize_args_to_infotext(p)
                
    def postprocess_batch_list(
            self, p, pp, enable_animatediff=False, loop_number=0, video_length=16, fps=8, model="mm_sd_v14.ckpt", **kwargs):
        if enable_animatediff:
            p.main_prompt = p.all_prompts[0] ## Ensure the video's infotext displays correctly below the video

    def save_video(self, p, res, loop_number, video_length, fps, video_paths, output_directory, image_itr, generated_filename):
        video_list = res.images[image_itr:image_itr + video_length]
        seq = images.get_next_sequence_number(output_directory, "")
        filename = f"{seq:05}-{generated_filename}"
        video_path_before_extension = f"{output_directory}/{filename}"
        video_extension = shared.opts.data.get("animatediff_file_format", "") or "gif"
        video_path = f"{video_path_before_extension}.{video_extension}"
        video_paths.append(video_path)
        video_duration = 1 / fps * 1000 # duration is defined in whole ms, not a fraction of a second
        video_use_lossless_quality = shared.opts.data.get("animatediff_use_lossless_quality", False)
        video_quality = shared.opts.data.get("animatediff_video_quality", 95)
        
        geninfo = res.infotext(p, res.index_of_first_image)
        use_geninfo = shared.opts.enable_pnginfo and geninfo is not None
            
        if video_extension == "gif":
            optimize = not video_use_lossless_quality
            imageio.mimsave(
                video_path, video_list, duration=video_duration, loop=loop_number, optimize=optimize, 
                comment=(geninfo if use_geninfo else ""))
        elif video_extension == "webp":
            if use_geninfo:
                exif_bytes = piexif.dump({
                        "Exif":{
                            piexif.ExifIFD.UserComment:piexif.helper.UserComment.dump(geninfo, encoding="unicode")}})
            imageio.mimsave(
                video_path, video_list, duration=video_duration, loop=loop_number, 
                quality=video_quality, lossless=video_use_lossless_quality, exif=(exif_bytes if use_geninfo else b''))

    def postprocess(
            self, p: StableDiffusionProcessing, res: Processed, 
            enable_animatediff=False, loop_number=0, video_length=16, fps=8, model="mm_sd_v14.ckpt"):
        
        if enable_animatediff:
            self.eject_motion_module_from_unet(p)
                
            if shared.opts.data.get("animatediff_always_save_videos", True):
                
                video_paths = []
                self.logger.info("Merging images into video.")
                
                namegen = images.FilenameGenerator(p, res.seed, res.prompt, res.images[0])
                
                from pathlib import Path
                output_directory = shared.opts.data.get("animatediff_outdir_videos", "") or f"{p.outpath_samples}/AnimateDiff"
                if shared.opts.data.get("animatediff_save_to_subdirectory", True):
                    dirname = namegen.apply(shared.opts.data.get("animatediff_subdirectories_filename_pattern","") or 
                                            "[date]").lstrip(' ').rstrip('\\ /')
                    output_directory = os.path.join(output_directory, dirname)
                Path(output_directory).mkdir(exist_ok=True, parents=True)
                
                generated_filename = namegen.apply(shared.opts.data.get("animatediff_filename_pattern","") or 
                                                   "[seed]").lstrip(' ').rstrip('\\ /')
                
                for image_itr in range(res.index_of_first_image, len(res.images), video_length):
                    self.save_video(p, res, loop_number, video_length, fps, video_paths, output_directory, image_itr, generated_filename)
                    
                res.images = video_paths
                self.logger.info("AnimateDiff process end.")
        
def get_control_locator(control_label):
    return f"{MODULE_NAME} {control_label}"
            
def infotext_pasted(infotext, results):
    """Parse AnimateDiff infotext string and write result to `results` dict."""
    
    if not shared.opts.data.get("animatediff_copy_paste_infotext"):
        return 
    
    for k, v in results.items():
        if not k.startswith(MODULE_NAME):
            continue

        assert isinstance(v, str), f"Expect string but got {v}."
        try:
            parsed_dictionary = ast.literal_eval(v)
            logger_animatediff.debug(parsed_dictionary)
            for field, value in parsed_dictionary.items():
                logger_animatediff.debug(f"{field} and {value}")
                results[get_control_locator(field)] = value
            results.pop(MODULE_NAME)
        except Exception:
            logger_animatediff.warn(
                f"Failed to parse infotext, legacy format infotext is no longer supported:\n{v}"
            )
        break
     
script_callbacks.on_infotext_pasted(infotext_pasted)
