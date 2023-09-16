import os
import gc
import gradio as gr

from modules import scripts, script_callbacks, shared

from scripts.logging_animatediff import logger_animatediff

EXTENSION_DIRECTORY = scripts.basedir()

ANIMATEDIFF_SETTINGS_SECTION = ('animatediff', "AnimateDiff")

# Function to create an option with title and info
def make_option(option_name, option_info):
    option = shared.OptionInfo(
        option_info["default"], option_info["title"], 
        option_info["component_type"], option_info.get("component_args", {}), 
        section=ANIMATEDIFF_SETTINGS_SECTION)
    
    if option_info.get("info") is not None:
        option.info(option_info.get("info"))
    if option_info.get("wiki_link") is not None:
        option.link("wiki", option_info.get("wiki_link"))
    
    shared.opts.add_option(option_name, option)
     
def get_file_format_info_text():
    return "gif: Compatible with most browsers and image viewers, generally high filesize, generally low quality. \
    webp: Medium compatibility with browsers and image viewers. \
    How an image viewer implements to webp library can affect how the image quality appears in that image viewer. \
    Best viewed in a modern browser. Generally lower file size with great quality."
    
def get_video_quality_info_text():
    return "Only applies when animatediff_use_lossless_quality is False. Higher quality results in higher file sizes. Does not apply to gif."
    
def get_name_pattern_wiki_link():
    return "https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Custom-Images-Filename-Name-and-Subdirectory"

def get_hack_groupnorm_info_text():
    return "Check if you want to hack GroupNorm. By default, V1 hacks GroupNorm, which avoids a performance degradation. \
        If you choose not to hack GroupNorm for V1, you will be able to use this extension in img2img in all cases, but the generated video will flicker. \
        V2 does not hack GroupNorm, so that this option will not influence v2 inference."

def on_ui_settings():

    # Define options in a dictionary
    animatediff_options = {
        "animatediff_model_path": {
            "default": os.path.join(EXTENSION_DIRECTORY, "model"),
            "title": "Path to save AnimateDiff motion modules",
            "component_type": gr.Textbox,
            "info": None,
            "wiki_link": None
        },
        "animatediff_always_save_videos": {
            "default": True,
            "title": "Always save generated videos",
            "component_type": gr.Checkbox,
            "info": None,
            "wiki_link": None
        },
        "animatediff_file_format": {
            "default": "gif",
            "title": "File format for videos",
            "component_type": gr.Radio,
            "component_args": {"choices": ["gif", "webp"]},
            "info": get_file_format_info_text(),
            "wiki_link": None
        },
        "animatediff_use_lossless_quality": {
            "default": False,
            "title": "Use lossless quality",
            "component_type": gr.Checkbox,
            "info": "Great quality, very high file size. Not recommended.",
            "wiki_link": None
        },
        "animatediff_video_quality": {
            "default": 95,
            "title": "Video Quality for webp",
            "component_type": gr.Slider,
            "component_args": {"minimum": 1, "maximum": 100, "step": 1},
            "info":  get_video_quality_info_text(),
            "wiki_link": None
        },
        "animatediff_filename_pattern": {
            "default": "[seed]",
            "title": "Videos filename pattern",
            "component_type": gr.Textbox,
            "info": None,
            "wiki_link": get_name_pattern_wiki_link()
        },
        "animatediff_outdir_videos": {
            "default": "",
            "title": "Output directory for videos",
            "component_type": gr.Textbox,
            "info": "Leave blank to place in a subfolder named 'AnimateDiff' inside of txt2img-images or img2img-images folders.",
            "wiki_link": None
        },
        "animatediff_save_to_subdirectory": {
            "default": True,
            "title": "Save videos to a subdirectory",
            "component_type": gr.Checkbox,
            "info": "If True, put videos in a subfolder named based on animatediff_subdirectories_filename_pattern",
            "wiki_link": None
        },
        "animatediff_subdirectories_filename_pattern": {
            "default": "[date]",
            "title": "Subdirectory name pattern",
            "component_type": gr.Textbox,
            "info": None,
            "wiki_link": get_name_pattern_wiki_link()
        },
        "animatediff_copy_paste_infotext": {
            "default": True,
            "title": "Copy-paste from infotext",
            "component_type": gr.Checkbox,
            "info": "If True, AnimateDiff data saved to infotext and pnginfo will be applied when using buttons like 'Send to txt2img'.",
            "wiki_link": None
        },
        "animatediff_hack_gn": {
            "default": True,
            "title": "Hack GroupNorm",
            "component_type": gr.Checkbox,
            "info": get_hack_groupnorm_info_text(),
            "wiki_link": None
        }
    }
    
    # Create options
    for option_name, option_info in animatediff_options.items():
        make_option(option_name, option_info)

script_callbacks.on_ui_settings(on_ui_settings)
