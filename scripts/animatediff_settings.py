import gradio as gr

from modules import shared
from scripts.animatediff_ui import supported_save_formats


def on_ui_settings():
    section = ("animatediff", "AnimateDiff")
    s3_selection =("animatediff", "AnimateDiff AWS") 
    shared.opts.add_option(
        "animatediff_model_path",
        shared.OptionInfo(
            None,
            "Path to save AnimateDiff motion modules",
            gr.Textbox,
            {"placeholder": "Leave empty to use default path: extensions/sd-webui-animatediff/model"},
            section=section,
        ),
    )
    shared.opts.add_option(
        "animatediff_default_save_formats",
        shared.OptionInfo(
            ["GIF", "PNG"],
            "Default Save Formats",
            gr.CheckboxGroup,
            {"choices": supported_save_formats},
            section=section
        ).needs_restart()
    )
    shared.opts.add_option(
        "animatediff_save_to_custom",
        shared.OptionInfo(
            True,
            "Save frames to stable-diffusion-webui/outputs/{ txt|img }2img-images/AnimateDiff/{gif filename}/{date} "
            "instead of stable-diffusion-webui/outputs/{ txt|img }2img-images/{date}/.",
            gr.Checkbox,
            section=section
        )
    )
    shared.opts.add_option(
        "animatediff_frame_extract_path",
        shared.OptionInfo(
            None,
            "Path to save extracted frames",
            gr.Textbox,
            {"placeholder": "Leave empty to use default path: tmp/animatediff-frames"},
            section=section
        )
    )
    shared.opts.add_option(
        "animatediff_frame_extract_remove",
        shared.OptionInfo(
            False,
            "Always remove extracted frames after processing",
            gr.Checkbox,
            section=section
        )
    )
    shared.opts.add_option(
        "animatediff_default_frame_extract_method",
        shared.OptionInfo(
            "ffmpeg",
            "Default frame extraction method",
            gr.Radio,
            {"choices": ["ffmpeg", "opencv"]},
            section=section
        )
    )

    # traditional video optimization specification
    shared.opts.add_option(
        "animatediff_optimize_gif_palette",
        shared.OptionInfo(
            False,
            "Calculate the optimal GIF palette, improves quality significantly, removes banding",
            gr.Checkbox,
            section=section
        )
    )
    shared.opts.add_option(
        "animatediff_optimize_gif_gifsicle",
        shared.OptionInfo(
            False,
            "Optimize GIFs with gifsicle, reduces file size",
            gr.Checkbox,
            section=section
        )
    )
    shared.opts.add_option(
        key="animatediff_mp4_crf",
        info=shared.OptionInfo(
            default=23,
            label="MP4 Quality (CRF)",
            component=gr.Slider,
            component_args={
                "minimum": 0,
                "maximum": 51,
                "step": 1},
            section=section
        )
        .link("docs", "https://trac.ffmpeg.org/wiki/Encode/H.264#crf")
        .info("17 for best quality, up to 28 for smaller size")
    )
    shared.opts.add_option(
        key="animatediff_mp4_preset",
        info=shared.OptionInfo(
            default="",
            label="MP4 Encoding Preset",
            component=gr.Dropdown,
            component_args={"choices": ["", 'veryslow', 'slower', 'slow', 'medium', 'fast', 'faster', 'veryfast', 'superfast', 'ultrafast']},
            section=section,
        )
        .link("docs", "https://trac.ffmpeg.org/wiki/Encode/H.264#Preset")
        .info("encoding speed, use the slowest you can tolerate")
    )
    shared.opts.add_option(
        key="animatediff_mp4_tune",
        info=shared.OptionInfo(
            default="",
            label="MP4 Tune encoding for content type",
            component=gr.Dropdown,
            component_args={"choices": ["", "film", "animation", "grain"]},
            section=section
        )
        .link("docs", "https://trac.ffmpeg.org/wiki/Encode/H.264#Tune")
        .info("optimize for specific content types")
    )
    shared.opts.add_option(
        "animatediff_webp_quality",
        shared.OptionInfo(
            80,
            "WebP Quality (if lossless=True, increases compression and CPU usage)",
            gr.Slider,
            {
                "minimum": 1,
                "maximum": 100,
                "step": 1},
            section=section
        )
    )
    shared.opts.add_option(
        "animatediff_webp_lossless",
        shared.OptionInfo(
            False,
            "Save WebP in lossless format (highest quality, largest file size)",
            gr.Checkbox,
            section=section
        )
    )

    # s3 storage specification, most likely for some startup
    shared.opts.add_option(
        "animatediff_s3_enable",
        shared.OptionInfo(
            False,
            "Enable to Store file in object storage that supports the s3 protocol",
            gr.Checkbox,
            section=s3_selection
        )
    )
    shared.opts.add_option(
        "animatediff_s3_host",
        shared.OptionInfo(
            None,
            "S3 protocol host",
            gr.Textbox,
            section=s3_selection,
        ),
    )
    shared.opts.add_option(
        "animatediff_s3_port",
        shared.OptionInfo(
            None,
            "S3 protocol port",
            gr.Textbox,
            section=s3_selection,
        ),
    )
    shared.opts.add_option(
        "animatediff_s3_access_key",
        shared.OptionInfo(
            None,
            "S3 protocol access_key",
            gr.Textbox,
            section=s3_selection,
        ),
    )
    shared.opts.add_option(
        "animatediff_s3_secret_key",
        shared.OptionInfo(
            None,
            "S3 protocol secret_key",
            gr.Textbox,
            section=s3_selection,
        ),
    )
    shared.opts.add_option(
        "animatediff_s3_storge_bucket",
        shared.OptionInfo(
            None,
            "Bucket for file storage",
            gr.Textbox,
            section=s3_selection,
        ),
    )