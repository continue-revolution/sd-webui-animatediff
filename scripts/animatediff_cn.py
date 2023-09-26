import cv2
from pathlib import Path
import torch
import numpy as np
from modules import shared
from modules.paths import data_path
from modules.processing import (StableDiffusionProcessing,
                                StableDiffusionProcessingImg2Img,
                                StableDiffusionProcessingTxt2Img)
from scripts.animatediff_ui import AnimateDiffProcess


class AnimateDiffControl:
    """
    Everything AnimateDiff need to do to properly V2V via ControlNet.
    It has to the following tasks:
    1. generate all control maps in `process`, set CN control as the first, do not use CN batch mode.
    2. inject control maps into ControlNet Script.latest_network in `before_process_batch`
    3. optionally save all control maps in `postprocess`
    """

    def __init__(self):
        self.control = {}
        self.hr_control = {}
        self.detected_maps = []
        self.post_processors = []
        self.cn_post_processors = []


    def get_input_frames(self, params: AnimateDiffProcess):
        if params.video_source is not None and params.video_source != '':
            cap = cv2.VideoCapture(params.video_source)
            params.fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_count = 0
            tmp_frame_dir = Path(f'{data_path}/tmp/animatediff-frames/')
            tmp_frame_dir.mkdir(parents=True, exist_ok=True)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imwrite(tmp_frame_dir/f"{frame_count}.png", frame)
                frame_count += 1
            # Release the video capture object
            cap.release()
            return shared.listfiles(str(tmp_frame_dir))
        elif params.video_path is not None and params.video_path != '':
            return shared.listfiles(params.video_path)
        else:
            return None


    def gen_control_map(self, p: StableDiffusionProcessing, params: AnimateDiffProcess):
        # various types of input sources
        # 1. a video from AnimateDiff parameters
        # 2. a list of images from ControlNet parameters
        # 3. for single image control, let CN handle it
        # 4. do not support img2img batch mode

        # import ControlNet packages
        # TODO: what should I do for people who do not have CN?
        from scripts import external_code
        from scripts.controlnet import (Script, set_numpy_seed)
        from scripts.batch_hijack import InputMode
        from scripts.hook import ControlModelType
        from scripts.logging import logger
        from scripts.processor import model_free_preprocessors
        from scripts.adapter import Adapter, StyleAdapter, Adapter_light
        from scripts.controlnet_lllite import PlugableControlLLLite
        from scripts.controlmodel_ipadapter import PlugableIPAdapter

        input_frames = self.get_input_frames(params)

        for idx, unit in enumerate(Script.get_enabled_units(p)):
            Script.bound_check_params(unit)
            if 'inpaint_only' == unit.module and issubclass(type(p), StableDiffusionProcessingImg2Img) and p.image_mask is not None:
                logger.warning('A1111 inpaint and ControlNet inpaint duplicated. ControlNet support enabled.')
                unit.module = 'inpaint'

            if 'inpaint_only' in unit.module or ('recolor' in unit.module and ('luminance' in unit.module or 'intensity' in unit.module)):
                self.cn_post_processors.append(False) # all "False" should be removed

            batch_images = []
            # TODO: use first image as temporary control
            if getattr(unit, 'input_mode', InputMode.SIMPLE) == InputMode.BATCH:
                if isinstance(unit.batch_images, str):
                    batch_images = shared.listfiles(unit.batch_images)
                    unit.input_mode = InputMode.SIMPLE
            else:
                try:
                    Script.choose_input_image(p, unit, idx)
                    self.cn_post_processors[-1] = True
                    continue
                except Exception:
                    batch_images = input_frames
                    if batch_images is None:
                        logger.warn(f"No input images found for ControlNet module {idx}")
                        continue

            resize_mode = external_code.resize_mode_from_value(unit.resize_mode)
            
            if unit.module in model_free_preprocessors:
                model_net = None
            else:
                model_net = Script.load_control_model(p, p.sd_model.model.diffusion_model, unit.model)
                model_net.reset()
            control_model_type = ControlModelType.ControlNet
            if 'reference' in unit.module:
                control_model_type = ControlModelType.AttentionInjection
            elif 'revision' in unit.module:
                control_model_type = ControlModelType.ReVision
            elif hasattr(model_net, 'control_model') and (isinstance(model_net.control_model, Adapter) or isinstance(model_net.control_model, Adapter_light)):
                control_model_type = ControlModelType.T2I_Adapter
            elif hasattr(model_net, 'control_model') and isinstance(model_net.control_model, StyleAdapter):
                control_model_type = ControlModelType.T2I_StyleAdapter
            elif isinstance(model_net, PlugableIPAdapter):
                control_model_type = ControlModelType.IPAdapter
            elif isinstance(model_net, PlugableControlLLLite):
                control_model_type = ControlModelType.Controlllite
            del model_net

            # TODO: actually read in images, also support inpainting
            controls = []
            hr_controls = []
            for input_image in batch_images:
                # safe numpy
                logger.debug("Safe numpy convertion START")
                input_image = np.ascontiguousarray(input_image.copy()).copy()
                logger.debug("Safe numpy convertion END")

                logger.info(f"Loading preprocessor: {unit.module}")
                preprocessor = self.preprocessor[unit.module]

                high_res_fix = isinstance(p, StableDiffusionProcessingTxt2Img) and getattr(p, 'enable_hr', False)

                h = (p.height // 8) * 8
                w = (p.width // 8) * 8

                if high_res_fix:
                    if p.hr_resize_x == 0 and p.hr_resize_y == 0:
                        hr_y = int(p.height * p.hr_scale)
                        hr_x = int(p.width * p.hr_scale)
                    else:
                        hr_y, hr_x = p.hr_resize_y, p.hr_resize_x
                    hr_y = (hr_y // 8) * 8
                    hr_x = (hr_x // 8) * 8
                else:
                    hr_y = h
                    hr_x = w

                if unit.module == 'inpaint_only+lama' and resize_mode == external_code.ResizeMode.OUTER_FIT:
                    # inpaint_only+lama is special and required outpaint fix
                    _, input_image = Script.detectmap_proc(input_image, unit.module, resize_mode, hr_y, hr_x)

                preprocessor_resolution = unit.processor_res
                if unit.pixel_perfect:
                    preprocessor_resolution = external_code.pixel_perfect_resolution(
                        input_image,
                        target_H=h,
                        target_W=w,
                        resize_mode=resize_mode
                    )

                logger.info(f'preprocessor resolution = {preprocessor_resolution}')
                # Preprocessor result may depend on numpy random operations, use the
                # random seed in `StableDiffusionProcessing` to make the 
                # preprocessor result reproducable.
                # Currently following preprocessors use numpy random:
                # - shuffle
                seed = set_numpy_seed(p)
                logger.debug(f"Use numpy seed {seed}.")
                detected_map, is_image = preprocessor(
                    input_image, 
                    res=preprocessor_resolution, 
                    thr_a=unit.threshold_a,
                    thr_b=unit.threshold_b,
                )

                if high_res_fix:
                    if is_image:
                        hr_control, hr_detected_map = Script.detectmap_proc(detected_map, unit.module, resize_mode, hr_y, hr_x)
                        self.detected_maps.append((hr_detected_map, unit.module))
                    else:
                        hr_control = detected_map
                    hr_controls.append(hr_control)
                else:
                    hr_control = None

                if is_image:
                    control, detected_map = Script.detectmap_proc(detected_map, unit.module, resize_mode, h, w)
                    self.detected_maps.append((detected_map, unit.module))
                else:
                    control = detected_map
                    self.detected_maps.append((input_image, unit.module))

                if control_model_type == ControlModelType.T2I_StyleAdapter:
                    control = control['last_hidden_state']

                if control_model_type == ControlModelType.ReVision:
                    control = control['image_embeds']
                
                controls.append(control)
            
            self.control[unit.module] = torch.cat(controls, dim=0)
            if len(hr_controls) > 0:
                self.hr_control[unit.module] = torch.cat(hr_controls, dim=0)

            # TODO: in postprocess, execute and remove
            if 'inpaint_only' in unit.module:
                final_inpaint_feed = hr_controls if len(hr_controls) > 0 else controls
                final_inpaint_feed = final_inpaint_feed.detach().cpu().numpy()
                final_inpaint_feed = np.ascontiguousarray(final_inpaint_feed).copy()
                final_inpaint_mask = final_inpaint_feed[:, 3, :, :].astype(np.float32)
                final_inpaint_raw = final_inpaint_feed[:, :3].astype(np.float32)
                sigma = shared.opts.data.get("control_net_inpaint_blur_sigma", 7)
                for i in range(final_inpaint_mask.shape[0]):
                    final_inpaint_mask[i] = cv2.dilate(final_inpaint_mask, np.ones((sigma, sigma), dtype=np.uint8))
                    final_inpaint_mask[i] = cv2.blur(final_inpaint_mask, (sigma, sigma))[None]
                _, _, Hmask, Wmask = final_inpaint_mask.shape
                final_inpaint_raw = torch.from_numpy(np.ascontiguousarray(final_inpaint_raw).copy())
                final_inpaint_mask = torch.from_numpy(np.ascontiguousarray(final_inpaint_mask).copy())

                def inpaint_only_post_processing(x, i):
                    _, H, W = x.shape
                    if Hmask != H or Wmask != W:
                        logger.error('Error: ControlNet find post-processing resolution mismatch. This could be related to other extensions hacked processing.')
                        return x
                    r = final_inpaint_raw[i].to(x.dtype).to(x.device)
                    m = final_inpaint_mask[i].to(x.dtype).to(x.device)
                    y = m * x.clip(0, 1) + (1 - m) * r
                    y = y.clip(0, 1)
                    return y

                self.post_processors.append(inpaint_only_post_processing)

            if 'recolor' in unit.module:
                final_feed = hr_controls if len(hr_controls) > 0 else controls
                final_feed = final_feed.detach().cpu().numpy()
                final_feed = np.ascontiguousarray(final_feed).copy()
                final_feed = final_feed[:, 0, :, :].astype(np.float32)
                final_feed = (final_feed * 255).clip(0, 255).astype(np.uint8)
                _, Hfeed, Wfeed = final_feed.shape

                if 'luminance' in unit.module:

                    def recolor_luminance_post_processing(x, i):
                        C, H, W = x.shape
                        if Hfeed != H or Wfeed != W or C != 3:
                            logger.error('Error: ControlNet find post-processing resolution mismatch. This could be related to other extensions hacked processing.')
                            return x
                        h = x.detach().cpu().numpy().transpose((1, 2, 0))
                        h = (h * 255).clip(0, 255).astype(np.uint8)
                        h = cv2.cvtColor(h, cv2.COLOR_RGB2LAB)
                        h[:, :, 0] = final_feed[i]
                        h = cv2.cvtColor(h, cv2.COLOR_LAB2RGB)
                        h = (h.astype(np.float32) / 255.0).transpose((2, 0, 1))
                        y = torch.from_numpy(h).clip(0, 1).to(x)
                        return y

                    self.post_processors.append(recolor_luminance_post_processing)

                if 'intensity' in unit.module:

                    def recolor_intensity_post_processing(x, i):
                        C, H, W = x.shape
                        if Hfeed != H or Wfeed != W or C != 3:
                            logger.error('Error: ControlNet find post-processing resolution mismatch. This could be related to other extensions hacked processing.')
                            return x
                        h = x.detach().cpu().numpy().transpose((1, 2, 0))
                        h = (h * 255).clip(0, 255).astype(np.uint8)
                        h = cv2.cvtColor(h, cv2.COLOR_RGB2HSV)
                        h[:, :, 2] = final_feed[i]
                        h = cv2.cvtColor(h, cv2.COLOR_HSV2RGB)
                        h = (h.astype(np.float32) / 255.0).transpose((2, 0, 1))
                        y = torch.from_numpy(h).clip(0, 1).to(x)
                        return y

                    self.post_processors.append(recolor_intensity_post_processing)

            # TODO: change this in before_process_batch
            if '+lama' in unit.module:
                forward_param.used_hint_cond_latent = hook.UnetHook.call_vae_using_process(p, control)
                self.noise_modifier = forward_param.used_hint_cond_latent
            
        Path(f'{data_path}/tmp/animatediff-frames/').rmdir()
        # TODO: actually modify CN hook
