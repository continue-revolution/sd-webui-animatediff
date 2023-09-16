import os
import gradio as gr

from scripts.animatediff_mm import mm_animatediff as motion_module


class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)

    def get_block_name(self):
        return "button"


class AnimateDiffProcess:
    def __init__(
            self, 
            enable=False, 
            loop_number=0, 
            video_length=16, 
            fps=8, 
            model="mm_sd_v15.ckpt"):
        self.enable = enable
        self.loop_number = loop_number
        self.video_length = video_length
        self.fps = fps
        self.model = model
    
    def get_list(self):
        return [
            self.enable,
            self.loop_number,
            self.video_length,
            self.fps,
            self.model,
        ]


class AnimateDiffUiGroup:
    txt2img_submit_button = None
    img2img_submit_button = None

    def __init__(self):
        self.params = AnimateDiffProcess()

    def render(self, is_img2img: bool, model_dir: str):
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        model_list = [f for f in os.listdir(model_dir) if f != ".gitkeep"]
        with gr.Accordion('AnimateDiff', open=False):
            with gr.Row():
                def refresh_models(*inputs):
                    new_model_list = [f for f in os.listdir(model_dir) if f != ".gitkeep"]
                    dd = inputs[0]
                    if dd in new_model_list:
                        selected = dd
                    elif len(new_model_list) > 0:
                        selected = new_model_list[0]
                    else:
                        selected = None
                    return gr.Dropdown.update(choices=new_model_list, value=selected)
                self.params.model = gr.Dropdown(choices=model_list, value=(model_list[0] if len(model_list) > 0 else None), label="Motion module", type="value")
                refresh_model = ToolButton(value='\U0001f504')
                refresh_model.click(refresh_models, self.params.model, self.params.model)
            with gr.Row():
                self.params.enable = gr.Checkbox(value=False, label='Enable AnimateDiff')
                self.params.video_length = gr.Slider(minimum=1, maximum=32, value=16, step=1, label="Number of frames", precision=0)
                self.params.fps = gr.Number(value=8, label="Frames per second (FPS)", precision=0)
                self.params.loop_number = gr.Number(minimum=0, value=0, label="Display loop number (0 = infinite loop)", precision=0)
            with gr.Row():
                unload = gr.Button(value="Move motion module to CPU (default if lowvram)")
                remove = gr.Button(value="Remove motion module from any memory")
                unload.click(fn=motion_module.unload)
                remove.click(fn=motion_module.remove)
        return self.register_unit(is_img2img)
    
    def register_unit(self, is_img2img: bool):
        unit = gr.State()
        unit_args = self.params.get_list()
        (
            AnimateDiffUiGroup.img2img_submit_button
            if is_img2img
            else AnimateDiffUiGroup.txt2img_submit_button
        ).click(
            fn=AnimateDiffProcess,
            inputs=unit_args,
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
