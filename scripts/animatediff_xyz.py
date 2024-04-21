from types import ModuleType
from typing import Optional

from modules import scripts

from scripts.animatediff_logger import logger_animatediff as logger

xyz_attrs: dict = {}

def patch_xyz():
    xyz_module = find_xyz_module()
    if xyz_module is None:
        logger.warning("XYZ module not found.")
        return
    MODULE = "[AnimateDiff]"
    xyz_module.axis_options.extend([
        xyz_module.AxisOption(
            label=f"{MODULE} Enabled",
            type=str_to_bool,
            apply=apply_state("enable"),
            choices=choices_bool),
        xyz_module.AxisOption(
            label=f"{MODULE} Motion Module",
            type=str,
            apply=apply_state("model")),
        xyz_module.AxisOption(
            label=f"{MODULE} Video length",
            type=int_or_float,
            apply=apply_state("video_length")),
        xyz_module.AxisOption(
            label=f"{MODULE} FPS",
            type=int_or_float,
            apply=apply_state("fps")),
        xyz_module.AxisOption(
            label=f"{MODULE} Use main seed",
            type=str_to_bool,
            apply=apply_state("use_main_seed"),
            choices=choices_bool),
        xyz_module.AxisOption(
            label=f"{MODULE} Closed loop",
            type=str,
            apply=apply_state("closed_loop"),
            choices=lambda: ["N", "R-P", "R+P", "A"]),
        xyz_module.AxisOption(
            label=f"{MODULE} Batch size",
            type=int_or_float,
            apply=apply_state("batch_size")),
        xyz_module.AxisOption(
            label=f"{MODULE} Stride",
            type=int_or_float,
            apply=apply_state("stride")),
        xyz_module.AxisOption(
            label=f"{MODULE} Overlap",
            type=int_or_float,
            apply=apply_state("overlap")),
        xyz_module.AxisOption(
            label=f"{MODULE} Interp",
            type=str_to_bool,
                apply=apply_state("interp"),
            choices=choices_bool),
        xyz_module.AxisOption(
            label=f"{MODULE} Interp X",
            type=int_or_float,
            apply=apply_state("interp_x")),
        xyz_module.AxisOption(
            label=f"{MODULE} Video path",
            type=str,
            apply=apply_state("video_path")),
        xyz_module.AxisOptionImg2Img(
            label=f"{MODULE} Latent power",
            type=int_or_float,
            apply=apply_state("latent_power")),
        xyz_module.AxisOptionImg2Img(
            label=f"{MODULE} Latent scale",
            type=int_or_float,
            apply=apply_state("latent_scale")),
        xyz_module.AxisOptionImg2Img(
            label=f"{MODULE} Latent power last",
            type=int_or_float,
            apply=apply_state("latent_power_last")),
        xyz_module.AxisOptionImg2Img(
            label=f"{MODULE} Latent scale last",
            type=int_or_float,
            apply=apply_state("latent_scale_last")),
        ])


def apply_state(k, key_map=None):
    def callback(_p, v, _vs):
        if key_map is not None:
            v = key_map[v]
        xyz_attrs[k] = v

    return callback


def str_to_bool(string):
    string = str(string)
    if string in ["None", ""]:
        return None
    elif string.lower() in ["true", "1"]:
        return True
    elif string.lower() in ["false", "0"]:
        return False
    else:
        raise ValueError(f"Could not convert string to boolean: {string}")


def int_or_float(string):
    try:
        return int(string)
    except ValueError:
        return float(string)


def choices_bool():
    return ["False", "True"]


def find_xyz_module() -> Optional[ModuleType]:
    for data in scripts.scripts_data:
        if data.script_class.__module__ in {"xyz_grid.py", "xy_grid.py", "scripts.xyz_grid", "scripts.xy_grid"} and hasattr(data, "module"):
            return data.module

    return None
