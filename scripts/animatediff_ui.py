from typing import List

import os
import cv2
import subprocess
import gradio as gr

from modules import shared
from modules.launch_utils import git
from modules.processing import StableDiffusionProcessing, StableDiffusionProcessingImg2Img

from scripts.animatediff_mm import mm_animatediff as motion_module
from scripts.animatediff_xyz import xyz_attrs
from scripts.animatediff_logger import logger_animatediff as logger
from scripts.animatediff_utils import get_controlnet_units, extract_frames_from_video

supported_save_formats = ["GIF", "MP4", "WEBP", "WEBM", "PNG", "TXT"]

class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)


    def get_block_name(self):
        return "button"


class AnimateDiffProcess:

    def __init__(
        self,
        model="mm_sd15_v3.safetensors",
        enable=False,
        video_length=0,
        fps=8,
        loop_number=0,
        closed_loop='R-P',
        batch_size=16,
        stride=1,
        overlap=-1,
        format=shared.opts.data.get("animatediff_default_save_formats", ["GIF", "PNG"]),
        interp='Off',
        interp_x=10,
        video_source=None,
        video_path='',
        mask_path='',
        freeinit_enable=False,
        freeinit_filter="butterworth",
        freeinit_ds=0.25,
        freeinit_dt=0.25,
        freeinit_iters=3,
        latent_power=1,
        latent_scale=32,
        last_frame=None,
        latent_power_last=1,
        latent_scale_last=32,
        request_id = '',
    ):
        self.model = model
        self.enable = enable
        self.video_length = video_length
        self.fps = fps
        self.loop_number = loop_number
        self.closed_loop = closed_loop
        self.batch_size = batch_size
        self.stride = stride
        self.overlap = overlap
        self.format = format
        self.interp = interp
        self.interp_x = interp_x
        self.video_source = video_source
        self.video_path = video_path
        self.mask_path = mask_path
        self.freeinit_enable = freeinit_enable
        self.freeinit_filter = freeinit_filter
        self.freeinit_ds = freeinit_ds
        self.freeinit_dt = freeinit_dt
        self.freeinit_iters = freeinit_iters
        self.latent_power = latent_power
        self.latent_scale = latent_scale
        self.last_frame = last_frame
        self.latent_power_last = latent_power_last
        self.latent_scale_last = latent_scale_last

        # non-ui states
        self.request_id = request_id
        self.video_default = False
        self.is_i2i_batch = False
        self.prompt_scheduler = None


    def get_list(self, is_img2img: bool):
        return list(vars(self).values())[:(25 if is_img2img else 20)]


    def get_dict(self, is_img2img: bool):
        infotext = {
            "model": self.model,
            "video_length": self.video_length,
            "fps": self.fps,
            "loop_number": self.loop_number,
            "closed_loop": self.closed_loop,
            "batch_size": self.batch_size,
            "stride": self.stride,
            "overlap": self.overlap,
            "interp": self.interp,
            "interp_x": self.interp_x,
            "freeinit_enable": self.freeinit_enable,
        }
        if self.request_id:
            infotext['request_id'] = self.request_id
        if motion_module.mm is not None and motion_module.mm.mm_hash is not None:
            infotext['mm_hash'] = motion_module.mm.mm_hash[:8]
        if is_img2img:
            infotext.update({
                "latent_power": self.latent_power,
                "latent_scale": self.latent_scale,
                "latent_power_last": self.latent_power_last,
                "latent_scale_last": self.latent_scale_last,
            })

        try:
            ad_git_tag = subprocess.check_output(
                [git, "-C", motion_module.get_model_dir(), "describe", "--tags"],
                shell=False, encoding='utf8').strip()
            infotext['version'] = ad_git_tag
        except Exception as e:
            logger.warning(f"Failed to get git tag for AnimateDiff: {e}")

        infotext_str = ', '.join(f"{k}: {v}" for k, v in infotext.items())
        return infotext_str


    def get_param_names(self, is_img2img: bool):
        preserve = ["model", "enable", "video_length", "fps", "loop_number", "closed_loop", "batch_size", "stride", "overlap", "format", "interp", "interp_x"]
        if is_img2img:
            preserve.extend(["latent_power", "latent_power_last", "latent_scale", "latent_scale_last"])
        
        return preserve


    def _check(self):
        assert (
            self.video_length >= 0 and self.fps > 0
        ), "Video length and FPS should be positive."
        assert not set(supported_save_formats[:-1]).isdisjoint(
            self.format
        ), "At least one saving format should be selected."


    def apply_xyz(self):
        for k, v in xyz_attrs.items():
            setattr(self, k, v)


    def set_p(self, p: StableDiffusionProcessing):
        self._check()
        if self.video_length < self.batch_size:
            p.batch_size = self.batch_size
        else:
            p.batch_size = self.video_length
        if self.video_length == 0:
            self.video_length = p.batch_size
            self.video_default = True
        if self.overlap == -1:
            self.overlap = self.batch_size // 4
        if "PNG" not in self.format or shared.opts.data.get("animatediff_save_to_custom", True):
            p.do_not_save_samples = True

        cn_units = get_controlnet_units(p)
        min_batch_in_cn = -1
        for cn_unit in cn_units:
            # batch path broadcast
            if (cn_unit.input_mode.name == 'SIMPLE' and cn_unit.image is None) or \
               (cn_unit.input_mode.name == 'BATCH' and not cn_unit.batch_images) or \
               (cn_unit.input_mode.name == 'MERGE' and not cn_unit.batch_input_gallery):
                if not self.video_path:
                    extract_frames_from_video(self)
                cn_unit.input_mode = cn_unit.input_mode.__class__.BATCH
                cn_unit.batch_images = self.video_path

            # mask path broadcast
            if cn_unit.input_mode.name == 'BATCH' and self.mask_path and not getattr(cn_unit, 'batch_mask_dir', False):
                cn_unit.batch_mask_dir = self.mask_path

            # find minimun control images in CN batch
            cn_unit_batch_params = cn_unit.batch_images.split('\n')
            if cn_unit.input_mode.name == 'BATCH':
                cn_unit.animatediff_batch = True # for A1111 sd-webui-controlnet
                if not any([cn_param.startswith("keyframe:") for cn_param in cn_unit_batch_params[1:]]):
                    cn_unit_batch_num = len(shared.listfiles(cn_unit_batch_params[0]))
                    if min_batch_in_cn == -1 or cn_unit_batch_num < min_batch_in_cn:
                        min_batch_in_cn = cn_unit_batch_num

        if min_batch_in_cn != -1:
            self.fix_video_length(p, min_batch_in_cn)
            def cn_batch_modifler(batch_image_files: List[str], p: StableDiffusionProcessing):
                return batch_image_files[:self.video_length]
            for cn_unit in cn_units:
                if cn_unit.input_mode.name == 'BATCH':
                    cur_batch_modifier = getattr(cn_unit, "batch_modifiers", [])
                    cur_batch_modifier.append(cn_batch_modifler)
                    cn_unit.batch_modifiers = cur_batch_modifier
        self.post_setup_cn_for_i2i_batch(p)
        logger.info(f"AnimateDiff + ControlNet will generate {self.video_length} frames.")


    def fix_video_length(self, p: StableDiffusionProcessing, min_batch_in_cn: int):
        # ensure that params.video_length <= video_length and params.batch_size <= video_length
        if self.video_length > min_batch_in_cn:
            self.video_length = min_batch_in_cn
            p.batch_size = min_batch_in_cn
        if self.batch_size > min_batch_in_cn:
            self.batch_size = min_batch_in_cn
        if self.video_default:
            self.video_length = min_batch_in_cn
            p.batch_size = min_batch_in_cn


    def post_setup_cn_for_i2i_batch(self, p: StableDiffusionProcessing):
        if not (self.is_i2i_batch and isinstance(p, StableDiffusionProcessingImg2Img)):
            return

        if len(p.init_images) > self.video_length:
            p.init_images = p.init_images[:self.video_length]
            if p.image_mask and isinstance(p.image_mask, list) and len(p.image_mask) > self.video_length:
                p.image_mask = p.image_mask[:self.video_length]
        if len(p.init_images) < self.video_length:
            self.video_length = len(p.init_images)
            p.batch_size = len(p.init_images)
        if len(p.init_images) < self.batch_size:
            self.batch_size = len(p.init_images)


class AnimateDiffUiGroup:
    txt2img_submit_button = None
    img2img_submit_button = None
    setting_sd_model_checkpoint = None
    animatediff_ui_group = []

    def __init__(self):
        self.params = AnimateDiffProcess()
        AnimateDiffUiGroup.animatediff_ui_group.append(self)

        # Free-init
        self.filter_type_list = [
            "butterworth",
            "gaussian",
            "box",
            "ideal"
        ]


    def get_model_list(self):
        model_dir = motion_module.get_model_dir()
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        def get_sd_rm_tag():
            if shared.sd_model.is_sdxl:
                return ["sd1"]
            elif shared.sd_model.is_sd2:
                return ["sd1, xl"]
            elif shared.sd_model.is_sd1:
                return ["xl"]
            else:
                return []
        return [f for f in os.listdir(model_dir) if f != ".gitkeep" and not any(tag in f for tag in get_sd_rm_tag())]


    def refresh_models(self, *inputs):
        new_model_list = self.get_model_list()
        dd = inputs[0]
        if dd in new_model_list:
            selected = dd
        elif len(new_model_list) > 0:
            selected = new_model_list[0]
        else:
            selected = None
        return gr.Dropdown.update(choices=new_model_list, value=selected)


    def render(self, is_img2img: bool, infotext_fields, paste_field_names):
        elemid_prefix = "img2img-ad-" if is_img2img else "txt2img-ad-"
        with gr.Accordion("AnimateDiff", open=False):
            gr.Markdown(value="Please click [this link](https://github.com/continue-revolution/sd-webui-animatediff/blob/master/docs/how-to-use.md#parameters) to read the documentation of each parameter.")
            with gr.Row():
                with gr.Row():
                    model_list = self.get_model_list()
                    self.params.model = gr.Dropdown(
                        choices=model_list,
                        value=(self.params.model if self.params.model in model_list else (model_list[0] if len(model_list) > 0 else None)),
                        label="Motion module",
                        type="value",
                        elem_id=f"{elemid_prefix}motion-module",
                    )
                    refresh_model = ToolButton(value="\U0001f504")
                    refresh_model.click(self.refresh_models, self.params.model, self.params.model)

                self.params.format = gr.CheckboxGroup(
                    choices=supported_save_formats,
                    label="Save format",
                    type="value",
                    elem_id=f"{elemid_prefix}save-format",
                    value=self.params.format,
                )
            with gr.Row():
                self.params.enable = gr.Checkbox(
                    value=self.params.enable, label="Enable AnimateDiff", 
                    elem_id=f"{elemid_prefix}enable"
                )
                self.params.video_length = gr.Number(
                    minimum=0,
                    value=self.params.video_length,
                    label="Number of frames",
                    precision=0,
                    elem_id=f"{elemid_prefix}video-length",
                )
                self.params.fps = gr.Number(
                    value=self.params.fps, label="FPS", precision=0, 
                    elem_id=f"{elemid_prefix}fps"
                )
                self.params.loop_number = gr.Number(
                    minimum=0,
                    value=self.params.loop_number,
                    label="Display loop number",
                    precision=0,
                    elem_id=f"{elemid_prefix}loop-number",
                )
            with gr.Row():
                self.params.closed_loop = gr.Radio(
                    choices=["N", "R-P", "R+P", "A"],
                    value=self.params.closed_loop,
                    label="Closed loop",
                    elem_id=f"{elemid_prefix}closed-loop",
                )
                self.params.batch_size = gr.Slider(
                    minimum=1,
                    maximum=32,
                    value=self.params.batch_size,
                    label="Context batch size",
                    step=1,
                    precision=0,
                    elem_id=f"{elemid_prefix}batch-size",
                )
                self.params.stride = gr.Number(
                    minimum=1,
                    value=self.params.stride,
                    label="Stride",
                    precision=0,
                    elem_id=f"{elemid_prefix}stride",
                )
                self.params.overlap = gr.Number(
                    minimum=-1,
                    value=self.params.overlap,
                    label="Overlap",
                    precision=0,
                    elem_id=f"{elemid_prefix}overlap",
                )
            with gr.Row():
                self.params.interp = gr.Radio(
                    choices=["Off", "FILM"],
                    label="Frame Interpolation",
                    elem_id=f"{elemid_prefix}interp-choice",
                    value=self.params.interp
                )
                self.params.interp_x = gr.Number(
                    value=self.params.interp_x, label="Interp X", precision=0, 
                    elem_id=f"{elemid_prefix}interp-x"
                )
            with gr.Accordion("FreeInit Params", open=False):
                gr.Markdown(
                    """
                    Adjust to control the smoothness.
                    """
                )
                self.params.freeinit_enable = gr.Checkbox(
                    value=self.params.freeinit_enable, 
                    label="Enable FreeInit", 
                    elem_id=f"{elemid_prefix}freeinit-enable"
                )
                self.params.freeinit_filter = gr.Dropdown(
                    value=self.params.freeinit_filter, 
                    label="Filter Type", 
                    info="Default as Butterworth. To fix large inconsistencies, consider using Gaussian.",
                    choices=self.filter_type_list,
                    interactive=True, 
                    elem_id=f"{elemid_prefix}freeinit-filter"
                )
                self.params.freeinit_ds = gr.Slider( 
                    value=self.params.freeinit_ds, 
                    minimum=0, 
                    maximum=1, 
                    step=0.125, 
                    label="d_s", 
                    info="Stop frequency for spatial dimensions (0.0-1.0)", 
                    elem_id=f"{elemid_prefix}freeinit-ds"
                )
                self.params.freeinit_dt = gr.Slider(
                    value=self.params.freeinit_dt, 
                    minimum=0, 
                    maximum=1, 
                    step=0.125, 
                    label="d_t", 
                    info="Stop frequency for temporal dimension (0.0-1.0)", 
                    elem_id=f"{elemid_prefix}freeinit-dt"
                )
                self.params.freeinit_iters = gr.Slider(
                    value=self.params.freeinit_iters, 
                    minimum=2, 
                    maximum=5, 
                    step=1, 
                    label="FreeInit Iterations", 
                    info="Larger value leads to smoother results & longer inference time.", 
                    elem_id=f"{elemid_prefix}freeinit-dt",
                )
            self.params.video_source = gr.Video(
                value=self.params.video_source,
                label="Video source",
            )
            def update_fps(video_source):
                if video_source is not None and video_source != '':
                    cap = cv2.VideoCapture(video_source)
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    cap.release()
                    return fps
                else:
                    return int(self.params.fps.value)
            self.params.video_source.change(update_fps, inputs=self.params.video_source, outputs=self.params.fps)
            def update_frames(video_source):
                if video_source is not None and video_source != '':
                    cap = cv2.VideoCapture(video_source)
                    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    return frames
                else:
                    return int(self.params.video_length.value)
            self.params.video_source.change(update_frames, inputs=self.params.video_source, outputs=self.params.video_length)
            with gr.Row():
                self.params.video_path = gr.Textbox(
                    value=self.params.video_path,
                    label="Video path",
                    elem_id=f"{elemid_prefix}video-path"
                )
                self.params.mask_path = gr.Textbox(
                    value=self.params.mask_path,
                    label="Mask path",
                    visible=False,
                    elem_id=f"{elemid_prefix}mask-path"
                )
            if is_img2img:
                with gr.Accordion("I2V Traditional", open=False):
                    with gr.Row():
                        self.params.latent_power = gr.Slider(
                            minimum=0.1,
                            maximum=10,
                            value=self.params.latent_power,
                            step=0.1,
                            label="Latent power",
                            elem_id=f"{elemid_prefix}latent-power",
                        )
                        self.params.latent_scale = gr.Slider(
                            minimum=1,
                            maximum=128,
                            value=self.params.latent_scale,
                            label="Latent scale",
                            elem_id=f"{elemid_prefix}latent-scale"
                        )
                        self.params.latent_power_last = gr.Slider(
                            minimum=0.1,
                            maximum=10,
                            value=self.params.latent_power_last,
                            step=0.1,
                            label="Optional latent power for last frame",
                            elem_id=f"{elemid_prefix}latent-power-last",
                        )
                        self.params.latent_scale_last = gr.Slider(
                            minimum=1,
                            maximum=128,
                            value=self.params.latent_scale_last,
                            label="Optional latent scale for last frame",
                            elem_id=f"{elemid_prefix}latent-scale-last"
                        )
                    self.params.last_frame = gr.Image(
                        label="Optional last frame. Leave it blank if you do not need one.",
                        type="pil",
                    )
            with gr.Row():
                unload = gr.Button(value="Move motion module to CPU (default if lowvram)")
                remove = gr.Button(value="Remove motion module from any memory")
                unload.click(fn=motion_module.unload)
                remove.click(fn=motion_module.remove)

        # Set up controls to be copy-pasted using infotext
        fields = self.params.get_param_names(is_img2img)
        infotext_fields.extend((getattr(self.params, field), f"AnimateDiff {field}") for field in fields)
        paste_field_names.extend(f"AnimateDiff {field}" for field in fields)

        return self.register_unit(is_img2img)


    def register_unit(self, is_img2img: bool):
        unit = gr.State(value=AnimateDiffProcess)
        (
            AnimateDiffUiGroup.img2img_submit_button
            if is_img2img
            else AnimateDiffUiGroup.txt2img_submit_button
        ).click(
            fn=AnimateDiffProcess,
            inputs=self.params.get_list(is_img2img),
            outputs=unit,
            queue=False,
        )
        return unit


    @staticmethod
    def on_after_component(component, **_kwargs):
        elem_id = getattr(component, "elem_id", None)

        if elem_id == "txt2img_generate":
            AnimateDiffUiGroup.txt2img_submit_button = component
            return

        if elem_id == "img2img_generate":
            AnimateDiffUiGroup.img2img_submit_button = component
            return

        if elem_id == "setting_sd_model_checkpoint":
            for group in AnimateDiffUiGroup.animatediff_ui_group:
                component.change( # this step cannot success. I don't know why.
                    fn=group.refresh_models,
                    inputs=[group.params.model],
                    outputs=[group.params.model],
                    queue=False,
                )
            return

