import cv2
from pathlib import Path
import torch
import numpy as np
from modules import shared, processing
from modules.paths import data_path
from modules.processing import (StableDiffusionProcessing,
                                StableDiffusionProcessingImg2Img,
                                StableDiffusionProcessingTxt2Img)
from scripts.animatediff_ui import AnimateDiffProcess
from scripts.animatediff_logger import logger_animatediff as logger


class AnimateDiffControl:

    def __init__(self, p: StableDiffusionProcessing):
        self.original_processing_process_images_hijack = None
        self.original_controlnet_main_entry = None
        try:
            from scripts.external_code import find_cn_script
            self.cn_script = find_cn_script(p.scripts)
        except:
            self.cn_script = None


    def hack_batchhijack(self, params: AnimateDiffProcess):
        logger.info('Hacking ControlNet BatchHijack')

        def get_input_frames():
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
                cap.release()
                return str(tmp_frame_dir)
            elif params.video_path is not None and params.video_path != '':
                return params.video_path
            return ''

        from scripts.batch_hijack import BatchHijack, InputMode
        from scripts import external_code

        def hacked_processing_process_images_hijack(self, p, *args, **kwargs):
            if self.is_batch:
                # we are in img2img batch tab, do a single batch iteration
                return self.process_images_cn_batch(p, *args, **kwargs)

            units = external_code.get_all_units_in_processing(p)
            units = [unit for unit in units if getattr(unit, 'enabled', False)]
            global_input_frames = get_input_frames()
            for unit in units:
                # if no input given for this unit, use global input
                # TODO: inpainting, SAM single image
                if getattr(unit, 'input_mode', InputMode.SIMPLE) == InputMode.BATCH:
                    if isinstance(unit.batch_images, str) and unit.batch_images == '':
                        unit.batch_images = shared.listfiles(unit.batch_images)
                    else:
                        assert global_input_frames != '', 'No input images found for ControlNet module'
                        unit.batch_images = global_input_frames
                elif unit.image is None:
                    assert global_input_frames != '', 'No input images found for ControlNet module'
                    unit.batch_images = global_input_frames
                    unit.input_mode = InputMode.BATCH

            video_length = min(len(getattr(unit, 'batch_images', []))
                            for unit in units
                            if getattr(unit, 'input_mode', InputMode.SIMPLE) == InputMode.BATCH)
            # ensure that params.video_length <= video_length and params.batch_size <= video_length
            if params.video_length > video_length:
                params.video_length = video_length
            if params.batch_size > video_length:
                params.batch_size = video_length

            return getattr(processing, '__controlnet_original_process_images_inner')(p, *args, **kwargs)
        
        self.original_processing_process_images_hijack = BatchHijack.processing_process_images_hijack
        BatchHijack.processing_process_images_hijack = hacked_processing_process_images_hijack
    

    def restore_batchhijack(self):
        logger.info('Restoring ControlNet BatchHijack')
        from scripts.batch_hijack import BatchHijack
        BatchHijack.processing_process_images_hijack = self.original_processing_process_images_hijack
        self.original_processing_process_images_hijack = None


    def hack_cn_main_entry(self):
        logger.info('Hacking ControlNet main entry')
        # TODO: hack this!
        # TODO: remove tmp_frame_dir
        def hacked_main_entry(self, p: StableDiffusionProcessing, params: AnimateDiffProcess):          
            sd_ldm = p.sd_model
            unet = sd_ldm.model.diffusion_model
            self.noise_modifier = None

            setattr(p, 'controlnet_control_loras', [])

            if self.latest_network is not None:
                # always restore (~0.05s)
                self.latest_network.restore()

            # always clear (~0.05s)
            clear_all_secondary_control_models()

            if not batch_hijack.instance.is_batch:
                self.enabled_units = Script.get_enabled_units(p)

            if len(self.enabled_units) == 0:
                self.latest_network = None
                return

            detected_maps = []
            forward_params = []
            post_processors = []

            # cache stuff
            if self.latest_model_hash != p.sd_model.sd_model_hash:
                Script.clear_control_model_cache()

            for idx, unit in enumerate(self.enabled_units):
                unit.module = global_state.get_module_basename(unit.module)

            # unload unused preproc
            module_list = [unit.module for unit in self.enabled_units]
            for key in self.unloadable:
                if key not in module_list:
                    self.unloadable.get(key, lambda:None)()

            self.latest_model_hash = p.sd_model.sd_model_hash
            for idx, unit in enumerate(self.enabled_units):
                Script.bound_check_params(unit)

                resize_mode = external_code.resize_mode_from_value(unit.resize_mode)
                control_mode = external_code.control_mode_from_value(unit.control_mode)

                if unit.module in model_free_preprocessors:
                    model_net = None
                else:
                    model_net = Script.load_control_model(p, unet, unit.model)
                    model_net.reset()

                    if getattr(model_net, 'is_control_lora', False):
                        control_lora = model_net.control_model
                        bind_control_lora(unet, control_lora)
                        p.controlnet_control_loras.append(control_lora)

                input_image, image_from_a1111 = Script.choose_input_image(p, unit, idx)
                if image_from_a1111:
                    a1111_i2i_resize_mode = getattr(p, "resize_mode", None)
                    if a1111_i2i_resize_mode is not None:
                        resize_mode = external_code.resize_mode_from_value(a1111_i2i_resize_mode)
                
                a1111_mask_image : Optional[Image.Image] = getattr(p, "image_mask", None)
                if 'inpaint' in unit.module and not image_has_mask(input_image) and a1111_mask_image is not None:
                    a1111_mask = np.array(prepare_mask(a1111_mask_image, p))
                    if a1111_mask.ndim == 2:
                        if a1111_mask.shape[0] == input_image.shape[0]:
                            if a1111_mask.shape[1] == input_image.shape[1]:
                                input_image = np.concatenate([input_image[:, :, 0:3], a1111_mask[:, :, None]], axis=2)
                                a1111_i2i_resize_mode = getattr(p, "resize_mode", None)
                                if a1111_i2i_resize_mode is not None:
                                    resize_mode = external_code.resize_mode_from_value(a1111_i2i_resize_mode)

                if 'reference' not in unit.module and issubclass(type(p), StableDiffusionProcessingImg2Img) \
                        and p.inpaint_full_res and a1111_mask_image is not None:
                    logger.debug("A1111 inpaint mask START")
                    input_image = [input_image[:, :, i] for i in range(input_image.shape[2])]
                    input_image = [Image.fromarray(x) for x in input_image]

                    mask = prepare_mask(a1111_mask_image, p)

                    crop_region = masking.get_crop_region(np.array(mask), p.inpaint_full_res_padding)
                    crop_region = masking.expand_crop_region(crop_region, p.width, p.height, mask.width, mask.height)

                    input_image = [
                        images.resize_image(resize_mode.int_value(), i, mask.width, mask.height) 
                        for i in input_image
                    ]

                    input_image = [x.crop(crop_region) for x in input_image]
                    input_image = [
                        images.resize_image(external_code.ResizeMode.OUTER_FIT.int_value(), x, p.width, p.height) 
                        for x in input_image
                    ]

                    input_image = [np.asarray(x)[:, :, 0] for x in input_image]
                    input_image = np.stack(input_image, axis=2)
                    logger.debug("A1111 inpaint mask END")

                if 'inpaint_only' == unit.module and issubclass(type(p), StableDiffusionProcessingImg2Img) and p.image_mask is not None:
                    logger.warning('A1111 inpaint and ControlNet inpaint duplicated. ControlNet support enabled.')
                    unit.module = 'inpaint'

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

                control_model_type = ControlModelType.ControlNet
                global_average_pooling = False

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

                if control_model_type is ControlModelType.ControlNet:
                    global_average_pooling = model_net.control_model.global_average_pooling

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
                        detected_maps.append((hr_detected_map, unit.module))
                    else:
                        hr_control = detected_map
                else:
                    hr_control = None

                if is_image:
                    control, detected_map = Script.detectmap_proc(detected_map, unit.module, resize_mode, h, w)
                    detected_maps.append((detected_map, unit.module))
                else:
                    control = detected_map
                    detected_maps.append((input_image, unit.module))

                if control_model_type == ControlModelType.T2I_StyleAdapter:
                    control = control['last_hidden_state']

                if control_model_type == ControlModelType.ReVision:
                    control = control['image_embeds']

                preprocessor_dict = dict(
                    name=unit.module,
                    preprocessor_resolution=preprocessor_resolution,
                    threshold_a=unit.threshold_a,
                    threshold_b=unit.threshold_b
                )

                forward_param = ControlParams(
                    control_model=model_net,
                    preprocessor=preprocessor_dict,
                    hint_cond=control,
                    weight=unit.weight,
                    guidance_stopped=False,
                    start_guidance_percent=unit.guidance_start,
                    stop_guidance_percent=unit.guidance_end,
                    advanced_weighting=None,
                    control_model_type=control_model_type,
                    global_average_pooling=global_average_pooling,
                    hr_hint_cond=hr_control,
                    soft_injection=control_mode != external_code.ControlMode.BALANCED,
                    cfg_injection=control_mode == external_code.ControlMode.CONTROL,
                )
                forward_params.append(forward_param)

                if 'inpaint_only' in unit.module:
                    final_inpaint_feed = hr_control if hr_control is not None else control
                    final_inpaint_feed = final_inpaint_feed.detach().cpu().numpy()
                    final_inpaint_feed = np.ascontiguousarray(final_inpaint_feed).copy()
                    final_inpaint_mask = final_inpaint_feed[0, 3, :, :].astype(np.float32)
                    final_inpaint_raw = final_inpaint_feed[0, :3].astype(np.float32)
                    sigma = shared.opts.data.get("control_net_inpaint_blur_sigma", 7)
                    final_inpaint_mask = cv2.dilate(final_inpaint_mask, np.ones((sigma, sigma), dtype=np.uint8))
                    final_inpaint_mask = cv2.blur(final_inpaint_mask, (sigma, sigma))[None]
                    _, Hmask, Wmask = final_inpaint_mask.shape
                    final_inpaint_raw = torch.from_numpy(np.ascontiguousarray(final_inpaint_raw).copy())
                    final_inpaint_mask = torch.from_numpy(np.ascontiguousarray(final_inpaint_mask).copy())

                    def inpaint_only_post_processing(x):
                        _, H, W = x.shape
                        if Hmask != H or Wmask != W:
                            logger.error('Error: ControlNet find post-processing resolution mismatch. This could be related to other extensions hacked processing.')
                            return x
                        r = final_inpaint_raw.to(x.dtype).to(x.device)
                        m = final_inpaint_mask.to(x.dtype).to(x.device)
                        y = m * x.clip(0, 1) + (1 - m) * r
                        y = y.clip(0, 1)
                        return y

                    post_processors.append(inpaint_only_post_processing)

                if 'recolor' in unit.module:
                    final_feed = hr_control if hr_control is not None else control
                    final_feed = final_feed.detach().cpu().numpy()
                    final_feed = np.ascontiguousarray(final_feed).copy()
                    final_feed = final_feed[0, 0, :, :].astype(np.float32)
                    final_feed = (final_feed * 255).clip(0, 255).astype(np.uint8)
                    Hfeed, Wfeed = final_feed.shape

                    if 'luminance' in unit.module:

                        def recolor_luminance_post_processing(x):
                            C, H, W = x.shape
                            if Hfeed != H or Wfeed != W or C != 3:
                                logger.error('Error: ControlNet find post-processing resolution mismatch. This could be related to other extensions hacked processing.')
                                return x
                            h = x.detach().cpu().numpy().transpose((1, 2, 0))
                            h = (h * 255).clip(0, 255).astype(np.uint8)
                            h = cv2.cvtColor(h, cv2.COLOR_RGB2LAB)
                            h[:, :, 0] = final_feed
                            h = cv2.cvtColor(h, cv2.COLOR_LAB2RGB)
                            h = (h.astype(np.float32) / 255.0).transpose((2, 0, 1))
                            y = torch.from_numpy(h).clip(0, 1).to(x)
                            return y

                        post_processors.append(recolor_luminance_post_processing)

                    if 'intensity' in unit.module:

                        def recolor_intensity_post_processing(x):
                            C, H, W = x.shape
                            if Hfeed != H or Wfeed != W or C != 3:
                                logger.error('Error: ControlNet find post-processing resolution mismatch. This could be related to other extensions hacked processing.')
                                return x
                            h = x.detach().cpu().numpy().transpose((1, 2, 0))
                            h = (h * 255).clip(0, 255).astype(np.uint8)
                            h = cv2.cvtColor(h, cv2.COLOR_RGB2HSV)
                            h[:, :, 2] = final_feed
                            h = cv2.cvtColor(h, cv2.COLOR_HSV2RGB)
                            h = (h.astype(np.float32) / 255.0).transpose((2, 0, 1))
                            y = torch.from_numpy(h).clip(0, 1).to(x)
                            return y

                        post_processors.append(recolor_intensity_post_processing)

                if '+lama' in unit.module:
                    forward_param.used_hint_cond_latent = hook.UnetHook.call_vae_using_process(p, control)
                    self.noise_modifier = forward_param.used_hint_cond_latent

                del model_net

            is_low_vram = any(unit.low_vram for unit in self.enabled_units)

            self.latest_network = UnetHook(lowvram=is_low_vram)
            self.latest_network.hook(model=unet, sd_ldm=sd_ldm, control_params=forward_params, process=p)

            for param in forward_params:
                if param.control_model_type == ControlModelType.IPAdapter:
                    param.control_model.hook(
                        model=unet,
                        clip_vision_output=param.hint_cond,
                        weight=param.weight,
                        dtype=torch.float32,
                        start=param.start_guidance_percent,
                        end=param.stop_guidance_percent
                    )
                if param.control_model_type == ControlModelType.Controlllite:
                    param.control_model.hook(
                        model=unet,
                        cond=param.hint_cond,
                        weight=param.weight,
                        start=param.start_guidance_percent,
                        end=param.stop_guidance_percent
                    )

            self.detected_map = detected_maps
            self.post_processors = post_processors

        self.original_controlnet_main_entry = self.cn_script.controlnet_main_entry
        self.cn_script.controlnet_main_entry = hacked_main_entry


    def restore_cn_main_entry(self):
        logger.info('Restoring ControlNet main entry')
        self.cn_script.controlnet_main_entry = self.original_controlnet_main_entry
        self.original_controlnet_main_entry = None


    def hack(self, params: AnimateDiffProcess):
        if self.cn_script is not None:
            self.hack_batchhijack(params)
            self.hack_cn_main_entry()


    def restore(self):
        if self.cn_script is not None:
            self.restore_batchhijack()
            self.restore_cn_main_entry()
