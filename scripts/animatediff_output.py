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
        for i in range(res.index_of_first_image, len(res.images), params.video_length):
            video_list = res.images[i : i + params.video_length]

            seq = images.get_next_sequence_number(
                f"{p.outpath_samples}/AnimateDiff", ""
            )
            filename = f"{seq:05}-{res.seed}"
            video_path_prefix = f"{p.outpath_samples}/AnimateDiff/{filename}."

            video_list = self._add_reverse(params, video_list)
            video_paths += self._save(params, video_list, video_path_prefix, res, i)

            interp_x = 10 # interpolate x10
            interp_fps = 30 # interpolated video fps
            try:
                from deforum_helpers.frame_interpolation import calculate_frames_to_add, check_and_download_film_model
                from film_interpolation.film_inference import run_film_interp_infer
            except ImportError:
                self.logger("Deforum not found. Please install: https://github.com/deforum-art/deforum-for-automatic1111-webui.git")
            else:
                import os
                import glob
                import shutil
                import modules.paths as ph
                deforum_models_path = ph.models_path + '/Deforum'
                film_model_folder = os.path.join(deforum_models_path,'film_interpolation')
                film_model_name = 'film_net_fp16.pt'
                film_model_path = os.path.join(film_model_folder, film_model_name)
                check_and_download_film_model('film_net_fp16.pt', film_model_folder)
        
                film_in_between_frames_count = calculate_frames_to_add(len(video_list), interp_x) 

                # deforum takes a folder path and uses all frames in it
                # so we save the frames to a tmp folder and remove later
                tmp_folder = f"{p.outpath_samples}/AnimateDiff/tmp"
                input_folder = f"{tmp_folder}/input"
                os.makedirs(input_folder, exist_ok=True)
                for i, frame in enumerate(video_list):
                    imageio.imwrite(f"{input_folder}/{i:05}.png", frame)

                # location to save interpolated frames
                save_folder = f"{tmp_folder}/{filename}"
                os.makedirs(save_folder, exist_ok=True)

                run_film_interp_infer(
                    model_path = film_model_path,
                    input_folder = input_folder,
                    save_folder = save_folder,
                    inter_frames = film_in_between_frames_count)

                if "MP4" in params.format:
                    # load the interpolated frames
                    interp_frame_paths = sorted(glob.glob(os.path.join(save_folder, '*.png')))
                    interp_frames = [imageio.imread(f) for f in interp_frame_paths]

                    video_path_mp4_interp = video_path_prefix + "FILM.mp4"
                    imageio.imwrite(video_path_mp4_interp, interp_frames, fps=interp_fps, codec="h264")
                    
                    # XXX need to test
                    video_paths.append(video_path_mp4_interp)

                # save interpolated frames if saving PNG frames
                if "PNG" in params.format:
                    save_interp_path = f"{p.outpath_samples}/AnimateDiff/interp"
                    os.makedirs(save_interp_path, exist_ok=True)
                    shutil.move(save_folder, save_interp_path)

                # remove tmp folder
                try: shutil.rmtree(tmp_folder)
                except OSError as e: print(f"Error: {e}")

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
