from pathlib import Path

import imageio.v3 as imageio
import numpy as np
from modules import images, shared
from modules.processing import Processed, StableDiffusionProcessing

from scripts.animatediff_logger import logger_animatediff as logger
from scripts.animatediff_ui import AnimateDiffProcess


class AnimateDiffOutput:
    def output(
        self, p: StableDiffusionProcessing, res: Processed, params: AnimateDiffProcess
    ):
        video_paths = []
        logger.info("Merging images into GIF.")
        Path(f"{p.outpath_samples}/AnimateDiff").mkdir(exist_ok=True, parents=True)
        step = params.video_length if params.video_length > params.batch_size else params.batch_size
        for i in range(res.index_of_first_image, len(res.images), step):
            video_list = res.images[i : i + params.video_length]

            seq = images.get_next_sequence_number(
                f"{p.outpath_samples}/AnimateDiff", ""
            )
            filename = f"{seq:05}-{res.seed}"
            video_path_prefix = f"{p.outpath_samples}/AnimateDiff/{filename}."

            video_list = self._add_reverse(params, video_list)
            video_paths += self._save(params, video_list, video_path_prefix, res, i)
        if len(video_paths) > 0 and not p.is_api:
            res.images = video_paths

    def _add_reverse(self, params: AnimateDiffProcess, video_list: list):
        if 0 in params.reverse:
            video_list_reverse = video_list[::-1]
            if 1 in params.reverse:
                video_list_reverse.pop(0)
            if 2 in params.reverse:
                video_list_reverse.pop(-1)
            return video_list + video_list_reverse
        return video_list

    def _save(
        self,
        params: AnimateDiffProcess,
        video_list: list,
        video_path_prefix: str,
        res: Processed,
        index: int,
    ):
        video_paths = []
        video_array = [np.array(v) for v in video_list]
        if "GIF" in params.format:
            video_path_gif = video_path_prefix + "gif"
            video_paths.append(video_path_gif)
            if shared.opts.data.get("animatediff_optimize_gif_palette", False):
                try:
                    import av
                except ImportError:
                    from launch import run_pip
                    run_pip(
                        "install imageio[pyav]",
                        "sd-webui-animatediff GIF palette optimization requirement: imageio[pyav]",
                    )
                imageio.imwrite(
                    video_path_gif, video_array, plugin='pyav', fps=params.fps, 
                    codec='gif', out_pixel_format='pal8',
                    filter_graph=(
                        {
                            "split": ("split", ""),
                            "palgen": ("palettegen", ""),
                            "paluse": ("paletteuse", ""),
                            "scale": ("scale", f"{video_list[0].width}:{video_list[0].height}")
                        },
                        [
                            ("video_in", "scale", 0, 0),
                            ("scale", "split", 0, 0),
                            ("split", "palgen", 1, 0),
                            ("split", "paluse", 0, 0),
                            ("palgen", "paluse", 0, 1),
                            ("paluse", "video_out", 0, 0),
                        ]
                    )
                )
            else:
                imageio.imwrite(
                    video_path_gif,
                    video_array,
                    duration=(1000 / params.fps),
                    loop=params.loop_number,
                )
            if shared.opts.data.get("animatediff_optimize_gif_gifsicle", False):
                self._optimize_gif(video_path_gif)
        if "MP4" in params.format:
            video_path_mp4 = video_path_prefix + "mp4"
            video_paths.append(video_path_mp4)
            imageio.imwrite(video_path_mp4, video_array, fps=params.fps, codec="h264")
        if "TXT" in params.format and res.images[index].info is not None:
            video_path_txt = video_path_prefix + "txt"
            self._save_txt(params, video_path_txt, res, index)
        return video_paths

    def _optimize_gif(self, video_path: str):
        try:
            import pygifsicle
        except ImportError:
            from launch import run_pip

            run_pip(
                "install pygifsicle",
                "sd-webui-animatediff GIF optimization requirement: pygifsicle",
            )
            import pygifsicle
        finally:
            try:
                pygifsicle.optimize(video_path)
            except FileNotFoundError:
                logger.warn(
                    "gifsicle not found, required for optimized GIFs, try: apt install gifsicle"
                )

    def _save_txt(
        self, params: AnimateDiffProcess, video_path: str, res: Processed, i: int
    ):
        res.images[i].info["motion_module"] = params.model
        res.images[i].info["video_length"] = params.video_length
        res.images[i].info["fps"] = params.fps
        res.images[i].info["loop_number"] = params.loop_number
        with open(video_path, "w", encoding="utf8") as file:
            file.write(f"{res.images[i].info}\n")
