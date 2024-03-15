import os
import cv2
import subprocess
from pathlib import Path

from modules import shared
from modules.paths import data_path
from modules.processing import StableDiffusionProcessing

from scripts.animatediff_logger import logger_animatediff as logger

def generate_random_hash(length=8):
    import hashlib
    import secrets

    # Generate a random number or string
    random_data = secrets.token_bytes(32)  # 32 bytes of random data

    # Create a SHA-256 hash of the random data
    hash_object = hashlib.sha256(random_data)
    hash_hex = hash_object.hexdigest()

    # Get the first 10 characters
    if length > len(hash_hex):
        length = len(hash_hex)
    return hash_hex[:length]


def get_animatediff_arg(p: StableDiffusionProcessing):
    """
    Get AnimateDiff argument from `p`. If it's a dict, convert it to AnimateDiffProcess.
    """
    if not p.scripts:
        return None

    for script in p.scripts.alwayson_scripts:
        if script.title().lower() == "animatediff":
            animatediff_arg = p.script_args[script.args_from]
            if isinstance(animatediff_arg, dict):
                from scripts.animatediff_ui import AnimateDiffProcess
                animatediff_arg = AnimateDiffProcess(**animatediff_arg)
                p.script_args = list(p.script_args)
                p.script_args[script.args_from] = animatediff_arg
            return animatediff_arg

    return None

def get_controlnet_units(p: StableDiffusionProcessing):
    """
    Get controlnet arguments from `p`.
    """
    if not p.scripts:
        return []

    for script in p.scripts.alwayson_scripts:
        if script.title().lower() == "controlnet":
            cn_units = p.script_args[script.args_from:script.args_to]

            if p.is_api and len(cn_units) > 0 and isinstance(cn_units[0], dict):
               from scripts import external_code
               from scripts.batch_hijack import InputMode
               cn_units_dataclass = external_code.get_all_units_in_processing(p)
               for cn_unit_dict, cn_unit_dataclass in zip(cn_units, cn_units_dataclass):
                    if cn_unit_dataclass.image is None:
                        cn_unit_dataclass.input_mode = InputMode.BATCH
                        cn_unit_dataclass.batch_images = cn_unit_dict.get("batch_images", None)
               p.script_args[script.args_from:script.args_to] = cn_units_dataclass

            return [x for x in cn_units if x.enabled] if not p.is_api else cn_units

    return []


def ffmpeg_extract_frames(source_video: str, output_dir: str, extract_key: bool = False):
    from modules.devices import device
    command = ["ffmpeg"]
    if "cuda" in str(device):
        command.extend(["-hwaccel", "cuda"])
    command.extend(["-i", source_video])
    if extract_key:
        command.extend(["-vf", "select='eq(pict_type,I)'", "-vsync", "vfr"])
    else:
        command.extend(["-filter:v", "mpdecimate=hi=64*200:lo=64*50:frac=0.33,setpts=N/FRAME_RATE/TB"])
    tmp_frame_dir = Path(output_dir)
    tmp_frame_dir.mkdir(parents=True, exist_ok=True)
    command.extend(["-qscale:v", "1", "-qmin", "1", "-c:a", "copy", str(tmp_frame_dir / '%09d.jpg')])
    logger.info(f"Attempting to extract frames via ffmpeg from {source_video} to {output_dir}")
    subprocess.run(command, check=True)


def cv2_extract_frames(source_video: str, output_dir: str):
    logger.info(f"Attempting to extract frames via OpenCV from {source_video} to {output_dir}")
    cap = cv2.VideoCapture(source_video)
    frame_count = 0
    tmp_frame_dir = Path(output_dir)
    tmp_frame_dir.mkdir(parents=True, exist_ok=True)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f"{tmp_frame_dir}/{frame_count}.png", frame)
        frame_count += 1
    cap.release()



def extract_frames_from_video(params):
    assert params.video_source, "You need to specify cond hint for ControlNet."
    params.video_path = shared.opts.data.get(
        "animatediff_frame_extract_path",
        f"{data_path}/tmp/animatediff-frames")
    if not params.video_path:
        params.video_path = f"{data_path}/tmp/animatediff-frames"
    params.video_path = os.path.join(params.video_path, f"{Path(params.video_source).stem}-{generate_random_hash()}")
    try:
        if shared.opts.data.get("animatediff_default_frame_extract_method", "ffmpeg") == "opencv":
            cv2_extract_frames(params.video_source, params.video_path)
        else:
            ffmpeg_extract_frames(params.video_source, params.video_path)
    except Exception as e:
        logger.error(f"[AnimateDiff] Error extracting frames via ffmpeg: {e}, fall back to OpenCV.")
        cv2_extract_frames(params.video_source, params.video_path)
