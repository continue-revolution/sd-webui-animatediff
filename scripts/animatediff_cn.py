from pathlib import Path
from types import MethodType
from typing import Optional

import os
import shutil
import cv2
import numpy as np
import torch
import hashlib
from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError
from modules import processing, shared, scripts, img2img, devices, masking, sd_samplers, images
from modules.paths import data_path
from modules.processing import (StableDiffusionProcessing,
                                StableDiffusionProcessingImg2Img,
                                StableDiffusionProcessingTxt2Img,
                                process_images,
                                create_binary_mask,
                                create_random_tensors,
                                images_tensor_to_samples,
                                setup_color_correction,
                                opt_f)
from modules.shared import opts
from modules.sd_samplers_common import images_tensor_to_samples, approximation_indexes

from scripts.animatediff_logger import logger_animatediff as logger
from scripts.animatediff_ui import AnimateDiffProcess
from scripts.animatediff_prompt import AnimateDiffPromptSchedule
from scripts.animatediff_infotext import update_infotext


class AnimateDiffControl:

    def __init__(self, p: StableDiffusionProcessing, prompt_scheduler: AnimateDiffPromptSchedule):
        self.original_processing_process_images_hijack = None
        self.original_img2img_process_batch_hijack = None
        self.original_controlnet_main_entry = None
        self.original_postprocess_batch = None
        try:
            from scripts.external_code import find_cn_script
            self.cn_script = find_cn_script(p.scripts)
        except:
            self.cn_script = None
        self.prompt_scheduler = prompt_scheduler


    def hack_batchhijack(self, params: AnimateDiffProcess):
        cn_script = self.cn_script
        prompt_scheduler = self.prompt_scheduler

        def get_input_frames():
            if params.video_source is not None and params.video_source != '':
                cap = cv2.VideoCapture(params.video_source)
                frame_count = 0
                tmp_frame_dir = Path(f'{data_path}/tmp/animatediff-frames/')
                tmp_frame_dir.mkdir(parents=True, exist_ok=True)
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    cv2.imwrite(f"{tmp_frame_dir}/{frame_count}.png", frame)
                    frame_count += 1
                cap.release()
                return str(tmp_frame_dir)
            elif params.video_path is not None and params.video_path != '':
                return params.video_path
            return ''

        from scripts.batch_hijack import BatchHijack, instance
        def hacked_processing_process_images_hijack(self, p: StableDiffusionProcessing, *args, **kwargs):
            from scripts import external_code
            from scripts.batch_hijack import InputMode

            units = external_code.get_all_units_in_processing(p)
            units = [unit for unit in units if getattr(unit, 'enabled', False)]

            if len(units) > 0:
                global_input_frames = get_input_frames()
                for idx, unit in enumerate(units):
                    # i2i-batch mode
                    if getattr(p, '_animatediff_i2i_batch', None) and not unit.image:
                        unit.input_mode = InputMode.BATCH
                    # if no input given for this unit, use global input
                    if getattr(unit, 'input_mode', InputMode.SIMPLE) == InputMode.BATCH:
                        if not unit.batch_images:
                            assert global_input_frames, 'No input images found for ControlNet module'
                            unit.batch_images = global_input_frames
                    elif not unit.image:
                        try:
                            cn_script.choose_input_image(p, unit, idx)
                        except:
                            assert global_input_frames != '', 'No input images found for ControlNet module'
                            unit.batch_images = global_input_frames
                            unit.input_mode = InputMode.BATCH

                    if getattr(unit, 'input_mode', InputMode.SIMPLE) == InputMode.BATCH:
                        if 'inpaint' in unit.module:
                            images = shared.listfiles(f'{unit.batch_images}/image')
                            masks = shared.listfiles(f'{unit.batch_images}/mask')
                            assert len(images) == len(masks), 'Inpainting image mask count mismatch'
                            unit.batch_images = [{'image': images[i], 'mask': masks[i]} for i in range(len(images))]
                        else:
                            unit.batch_images = shared.listfiles(unit.batch_images)

                unit_batch_list = [len(unit.batch_images) for unit in units
                                   if getattr(unit, 'input_mode', InputMode.SIMPLE) == InputMode.BATCH]
                if getattr(p, '_animatediff_i2i_batch', None):
                    unit_batch_list.append(len(p.init_images))

                if len(unit_batch_list) > 0:
                    video_length = min(unit_batch_list)
                    # ensure that params.video_length <= video_length and params.batch_size <= video_length
                    if params.video_length > video_length:
                        params.video_length = video_length
                    if params.batch_size > video_length:
                        params.batch_size = video_length
                    if params.video_default:
                        params.video_length = video_length
                        p.batch_size = video_length
                    for unit in units:
                        if getattr(unit, 'input_mode', InputMode.SIMPLE) == InputMode.BATCH:
                            unit.batch_images = unit.batch_images[:params.video_length]

            prompt_scheduler.parse_prompt(p)
            update_infotext(p, params)
            return getattr(processing, '__controlnet_original_process_images_inner')(p, *args, **kwargs)
        
        self.original_processing_process_images_hijack = BatchHijack.processing_process_images_hijack
        BatchHijack.processing_process_images_hijack = hacked_processing_process_images_hijack
        processing.process_images_inner = instance.processing_process_images_hijack

        def hacked_i2i_init(self, all_prompts, all_seeds, all_subseeds): # only hack this when i2i-batch with batch mask
            # TODO: hack this!
            self.image_cfg_scale: float = self.image_cfg_scale if shared.sd_model.cond_stage_key == "edit" else None

            self.sampler = sd_samplers.create_sampler(self.sampler_name, self.sd_model)
            crop_region = None

            image_mask = self.image_mask

            if image_mask is not None:
                # image_mask is passed in as RGBA by Gradio to support alpha masks,
                # but we still want to support binary masks.
                image_mask = create_binary_mask(image_mask)

                if self.inpainting_mask_invert:
                    image_mask = ImageOps.invert(image_mask)

                if self.mask_blur_x > 0:
                    np_mask = np.array(image_mask)
                    kernel_size = 2 * int(2.5 * self.mask_blur_x + 0.5) + 1
                    np_mask = cv2.GaussianBlur(np_mask, (kernel_size, 1), self.mask_blur_x)
                    image_mask = Image.fromarray(np_mask)

                if self.mask_blur_y > 0:
                    np_mask = np.array(image_mask)
                    kernel_size = 2 * int(2.5 * self.mask_blur_y + 0.5) + 1
                    np_mask = cv2.GaussianBlur(np_mask, (1, kernel_size), self.mask_blur_y)
                    image_mask = Image.fromarray(np_mask)

                if self.inpaint_full_res:
                    self.mask_for_overlay = image_mask
                    mask = image_mask.convert('L')
                    crop_region = masking.get_crop_region(np.array(mask), self.inpaint_full_res_padding)
                    crop_region = masking.expand_crop_region(crop_region, self.width, self.height, mask.width, mask.height)
                    x1, y1, x2, y2 = crop_region

                    mask = mask.crop(crop_region)
                    image_mask = images.resize_image(2, mask, self.width, self.height)
                    self.paste_to = (x1, y1, x2-x1, y2-y1)
                else:
                    image_mask = images.resize_image(self.resize_mode, image_mask, self.width, self.height)
                    np_mask = np.array(image_mask)
                    np_mask = np.clip((np_mask.astype(np.float32)) * 2, 0, 255).astype(np.uint8)
                    self.mask_for_overlay = Image.fromarray(np_mask)

                self.overlay_images = []

            latent_mask = self.latent_mask if self.latent_mask is not None else image_mask

            add_color_corrections = opts.img2img_color_correction and self.color_corrections is None
            if add_color_corrections:
                self.color_corrections = []
            imgs = []
            for img in self.init_images:

                # Save init image
                if opts.save_init_img:
                    self.init_img_hash = hashlib.md5(img.tobytes()).hexdigest()
                    images.save_image(img, path=opts.outdir_init_images, basename=None, forced_filename=self.init_img_hash, save_to_dirs=False)

                image = images.flatten(img, opts.img2img_background_color)

                if crop_region is None and self.resize_mode != 3:
                    image = images.resize_image(self.resize_mode, image, self.width, self.height)

                if image_mask is not None:
                    image_masked = Image.new('RGBa', (image.width, image.height))
                    image_masked.paste(image.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(self.mask_for_overlay.convert('L')))

                    self.overlay_images.append(image_masked.convert('RGBA'))

                # crop_region is not None if we are doing inpaint full res
                if crop_region is not None:
                    image = image.crop(crop_region)
                    image = images.resize_image(2, image, self.width, self.height)

                if image_mask is not None:
                    if self.inpainting_fill != 1:
                        image = masking.fill(image, latent_mask)

                if add_color_corrections:
                    self.color_corrections.append(setup_color_correction(image))

                image = np.array(image).astype(np.float32) / 255.0
                image = np.moveaxis(image, 2, 0)

                imgs.append(image)

            if len(imgs) == 1:
                batch_images = np.expand_dims(imgs[0], axis=0).repeat(self.batch_size, axis=0)
                if self.overlay_images is not None:
                    self.overlay_images = self.overlay_images * self.batch_size

                if self.color_corrections is not None and len(self.color_corrections) == 1:
                    self.color_corrections = self.color_corrections * self.batch_size

            elif len(imgs) <= self.batch_size:
                self.batch_size = len(imgs)
                batch_images = np.array(imgs)
            else:
                raise RuntimeError(f"bad number of images passed: {len(imgs)}; expecting {self.batch_size} or less")

            image = torch.from_numpy(batch_images)
            image = image.to(shared.device, dtype=devices.dtype_vae)

            if opts.sd_vae_encode_method != 'Full':
                self.extra_generation_params['VAE Encoder'] = opts.sd_vae_encode_method

            self.init_latent = images_tensor_to_samples(image, approximation_indexes.get(opts.sd_vae_encode_method), self.sd_model)
            devices.torch_gc()

            if self.resize_mode == 3:
                self.init_latent = torch.nn.functional.interpolate(self.init_latent, size=(self.height // opt_f, self.width // opt_f), mode="bilinear")

            if image_mask is not None:
                init_mask = latent_mask
                latmask = init_mask.convert('RGB').resize((self.init_latent.shape[3], self.init_latent.shape[2]))
                latmask = np.moveaxis(np.array(latmask, dtype=np.float32), 2, 0) / 255
                latmask = latmask[0]
                latmask = np.around(latmask)
                latmask = np.tile(latmask[None], (4, 1, 1))

                self.mask = torch.asarray(1.0 - latmask).to(shared.device).type(self.sd_model.dtype)
                self.nmask = torch.asarray(latmask).to(shared.device).type(self.sd_model.dtype)

                # this needs to be fixed to be done in sample() using actual seeds for batches
                if self.inpainting_fill == 2:
                    self.init_latent = self.init_latent * self.mask + create_random_tensors(self.init_latent.shape[1:], all_seeds[0:self.init_latent.shape[0]]) * self.nmask
                elif self.inpainting_fill == 3:
                    self.init_latent = self.init_latent * self.mask

            self.image_conditioning = self.img2img_image_conditioning(image * 2 - 1, self.init_latent, image_mask)

        def hacked_img2img_process_batch_hijack(
                self, p: StableDiffusionProcessingImg2Img, input_dir: str, output_dir: str, inpaint_mask_dir: str,
                args, to_scale=False, scale_by=1.0, use_png_info=False, png_info_props=None, png_info_dir=None):
            p._animatediff_i2i_batch = 1 # i2i-batch mode, ordinary
            output_dir = output_dir.strip()
            processing.fix_seed(p)

            images = list(shared.walk_files(input_dir, allowed_extensions=(".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff")))

            is_inpaint_batch = False
            if inpaint_mask_dir:
                inpaint_masks = shared.listfiles(inpaint_mask_dir)
                is_inpaint_batch = bool(inpaint_masks)

                if is_inpaint_batch:
                    assert len(inpaint_masks) == 1 or len(inpaint_masks) == len(images), 'The number of masks must be 1 or equal to the number of images.'
                    logger.info(f"\n[i2i batch] Inpaint batch is enabled. {len(inpaint_masks)} masks found.")
                    if len(inpaint_masks) > 1: # batch mask
                        p.init = MethodType(hacked_i2i_init, p)

            logger.info(f"[i2i batch] Will process {len(images)} images, creating {p.n_iter} new videos.")

            # extract "default" params to use in case getting png info fails
            prompt = p.prompt
            negative_prompt = p.negative_prompt
            seed = p.seed
            cfg_scale = p.cfg_scale
            sampler_name = p.sampler_name
            steps = p.steps
            frame_images = []
            frame_masks = []

            for i, image in enumerate(images):

                try:
                    img = Image.open(image)
                except UnidentifiedImageError as e:
                    print(e)
                    continue
                # Use the EXIF orientation of photos taken by smartphones.
                img = ImageOps.exif_transpose(img)

                if to_scale:
                    p.width = int(img.width * scale_by)
                    p.height = int(img.height * scale_by)

                frame_images.append(img)

                image_path = Path(image)
                if is_inpaint_batch:
                    if len(inpaint_masks) == 1:
                        mask_image_path = inpaint_masks[0]
                        p.image_mask = Image.open(mask_image_path)
                    else:
                        # try to find corresponding mask for an image using index matching
                        mask_image_path = inpaint_masks[i]
                        frame_masks.append(Image.open(mask_image_path))

                    mask_image = Image.open(mask_image_path)
                    p.image_mask = mask_image

            if use_png_info:
                try:
                    info_img = frame_images[0]
                    if png_info_dir:
                        info_img_path = os.path.join(png_info_dir, os.path.basename(image))
                        info_img = Image.open(info_img_path)
                    from modules import images as imgutil
                    from modules.generation_parameters_copypaste import parse_generation_parameters
                    geninfo, _ = imgutil.read_info_from_image(info_img)
                    parsed_parameters = parse_generation_parameters(geninfo)
                    parsed_parameters = {k: v for k, v in parsed_parameters.items() if k in (png_info_props or {})}
                except Exception:
                    parsed_parameters = {}

                p.prompt = prompt + (" " + parsed_parameters["Prompt"] if "Prompt" in parsed_parameters else "")
                p.negative_prompt = negative_prompt + (" " + parsed_parameters["Negative prompt"] if "Negative prompt" in parsed_parameters else "")
                p.seed = int(parsed_parameters.get("Seed", seed))
                p.cfg_scale = float(parsed_parameters.get("CFG scale", cfg_scale))
                p.sampler_name = parsed_parameters.get("Sampler", sampler_name)
                p.steps = int(parsed_parameters.get("Steps", steps))
            
            p.init_images = frame_images
            if len(frame_masks) > 0:
                p.image_mask = frame_masks

            proc = scripts.scripts_img2img.run(p, *args) # we should not support this, but just leave it here
            if proc is None:
                if output_dir:
                    p.outpath_samples = output_dir
                    p.override_settings['save_to_dirs'] = False
                    if p.n_iter > 1 or p.batch_size > 1:
                        p.override_settings['samples_filename_pattern'] = f'{image_path.stem}-[generation_number]'
                    else:
                        p.override_settings['samples_filename_pattern'] = f'{image_path.stem}'
                process_images(p)
            else:
                logger.warn("Warning: you are using an unsupported external script. AnimateDiff may not work properly.")

        self.original_img2img_process_batch_hijack = BatchHijack.img2img_process_batch_hijack
        BatchHijack.img2img_process_batch_hijack = hacked_img2img_process_batch_hijack
        img2img.process_batch = instance.img2img_process_batch_hijack


    def restore_batchhijack(self):
        from scripts.batch_hijack import BatchHijack, instance
        BatchHijack.processing_process_images_hijack = self.original_processing_process_images_hijack
        self.original_processing_process_images_hijack = None
        processing.process_images_inner = instance.processing_process_images_hijack
        BatchHijack.img2img_process_batch_hijack = self.original_img2img_process_batch_hijack
        self.original_img2img_process_batch_hijack = None
        img2img.process_batch = instance.img2img_process_batch_hijack


    def hack_cn(self):
        cn_script = self.cn_script


        def hacked_main_entry(self, p: StableDiffusionProcessing):
            from scripts import external_code, global_state, hook
            # from scripts.controlnet_lora import bind_control_lora # do not support control lora for sdxl
            from scripts.adapter import Adapter, Adapter_light, StyleAdapter
            from scripts.batch_hijack import InputMode
            # from scripts.controlnet_lllite import PlugableControlLLLite, clear_all_lllite # do not support controlllite for sdxl
            from scripts.controlmodel_ipadapter import (PlugableIPAdapter,
                                                        clear_all_ip_adapter)
            from scripts.hook import ControlModelType, ControlParams, UnetHook
            from scripts.logging import logger
            from scripts.processor import model_free_preprocessors

            # TODO: i2i-batch mode, what should I change?
            def image_has_mask(input_image: np.ndarray) -> bool:
                return (
                    input_image.ndim == 3 and 
                    input_image.shape[2] == 4 and 
                    np.max(input_image[:, :, 3]) > 127
                )


            def prepare_mask(
                mask: Image.Image, p: processing.StableDiffusionProcessing
            ) -> Image.Image:
                mask = mask.convert("L")
                if getattr(p, "inpainting_mask_invert", False):
                    mask = ImageOps.invert(mask)
                
                if hasattr(p, 'mask_blur_x'):
                    if getattr(p, "mask_blur_x", 0) > 0:
                        np_mask = np.array(mask)
                        kernel_size = 2 * int(2.5 * p.mask_blur_x + 0.5) + 1
                        np_mask = cv2.GaussianBlur(np_mask, (kernel_size, 1), p.mask_blur_x)
                        mask = Image.fromarray(np_mask)
                    if getattr(p, "mask_blur_y", 0) > 0:
                        np_mask = np.array(mask)
                        kernel_size = 2 * int(2.5 * p.mask_blur_y + 0.5) + 1
                        np_mask = cv2.GaussianBlur(np_mask, (1, kernel_size), p.mask_blur_y)
                        mask = Image.fromarray(np_mask)
                else:
                    if getattr(p, "mask_blur", 0) > 0:
                        mask = mask.filter(ImageFilter.GaussianBlur(p.mask_blur))
                
                return mask


            def set_numpy_seed(p: processing.StableDiffusionProcessing) -> Optional[int]:
                try:
                    tmp_seed = int(p.all_seeds[0] if p.seed == -1 else max(int(p.seed), 0))
                    tmp_subseed = int(p.all_seeds[0] if p.subseed == -1 else max(int(p.subseed), 0))
                    seed = (tmp_seed + tmp_subseed) & 0xFFFFFFFF
                    np.random.seed(seed)
                    return seed
                except Exception as e:
                    logger.warning(e)
                    logger.warning('Warning: Failed to use consistent random seed.')
                    return None

            sd_ldm = p.sd_model
            unet = sd_ldm.model.diffusion_model
            self.noise_modifier = None

            # setattr(p, 'controlnet_control_loras', []) # do not support control lora for sdxl

            if self.latest_network is not None:
                # always restore (~0.05s)
                self.latest_network.restore()

            # always clear (~0.05s)
            # clear_all_lllite() # do not support controlllite for sdxl
            clear_all_ip_adapter()

            self.enabled_units = cn_script.get_enabled_units(p)

            if len(self.enabled_units) == 0:
                self.latest_network = None
                return

            detected_maps = []
            forward_params = []
            post_processors = []

            # cache stuff
            if self.latest_model_hash != p.sd_model.sd_model_hash:
                cn_script.clear_control_model_cache()

            for idx, unit in enumerate(self.enabled_units):
                unit.module = global_state.get_module_basename(unit.module)

            # unload unused preproc
            module_list = [unit.module for unit in self.enabled_units]
            for key in self.unloadable:
                if key not in module_list:
                    self.unloadable.get(key, lambda:None)()

            self.latest_model_hash = p.sd_model.sd_model_hash
            for idx, unit in enumerate(self.enabled_units):
                cn_script.bound_check_params(unit)

                resize_mode = external_code.resize_mode_from_value(unit.resize_mode)
                control_mode = external_code.control_mode_from_value(unit.control_mode)

                if unit.module in model_free_preprocessors:
                    model_net = None
                else:
                    model_net = cn_script.load_control_model(p, unet, unit.model)
                    model_net.reset()

                    # if getattr(model_net, 'is_control_lora', False): # do not support control lora for sdxl
                    #     control_lora = model_net.control_model
                    #     bind_control_lora(unet, control_lora)
                    #     p.controlnet_control_loras.append(control_lora)

                if getattr(unit, 'input_mode', InputMode.SIMPLE) == InputMode.BATCH:
                    input_images = []
                    for img in unit.batch_images:
                        unit.image = img # TODO: SAM extension should use new API
                        input_image, _ = cn_script.choose_input_image(p, unit, idx)
                        input_images.append(input_image)
                else:
                    input_image, image_from_a1111 = cn_script.choose_input_image(p, unit, idx)
                    input_images = [input_image]

                    if image_from_a1111:
                        a1111_i2i_resize_mode = getattr(p, "resize_mode", None)
                        if a1111_i2i_resize_mode is not None:
                            resize_mode = external_code.resize_mode_from_value(a1111_i2i_resize_mode)

                for idx, input_image in enumerate(input_images):
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

                    # safe numpy
                    logger.debug("Safe numpy convertion START")
                    input_image = np.ascontiguousarray(input_image.copy()).copy()
                    logger.debug("Safe numpy convertion END")

                    input_images[idx] = input_image

                if 'inpaint_only' == unit.module and issubclass(type(p), StableDiffusionProcessingImg2Img) and p.image_mask is not None:
                    logger.warning('A1111 inpaint and ControlNet inpaint duplicated. ControlNet support enabled.')
                    unit.module = 'inpaint'

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
                    for idx, input_image in enumerate(input_images):
                        _, input_image = cn_script.detectmap_proc(input_image, unit.module, resize_mode, hr_y, hr_x)
                        input_images[idx] = input_image

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
                # elif isinstance(model_net, PlugableControlLLLite): # do not support controlllite for sdxl
                #     control_model_type = ControlModelType.Controlllite

                if control_model_type is ControlModelType.ControlNet:
                    global_average_pooling = model_net.control_model.global_average_pooling

                preprocessor_resolution = unit.processor_res
                if unit.pixel_perfect:
                    preprocessor_resolution = external_code.pixel_perfect_resolution(
                        input_images[0],
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

                controls = []
                hr_controls = []
                controls_ipadapter = {'hidden_states': [], 'image_embeds': []}
                hr_controls_ipadapter = {'hidden_states': [], 'image_embeds': []}
                for idx, input_image in enumerate(input_images):
                    detected_map, is_image = preprocessor(
                        input_image, 
                        res=preprocessor_resolution, 
                        thr_a=unit.threshold_a,
                        thr_b=unit.threshold_b,
                    )

                    if high_res_fix:
                        if is_image:
                            hr_control, hr_detected_map = cn_script.detectmap_proc(detected_map, unit.module, resize_mode, hr_y, hr_x)
                            detected_maps.append((hr_detected_map, unit.module))
                        else:
                            hr_control = detected_map
                    else:
                        hr_control = None

                    if is_image:
                        control, detected_map = cn_script.detectmap_proc(detected_map, unit.module, resize_mode, h, w)
                        detected_maps.append((detected_map, unit.module))
                    else:
                        control = detected_map
                        detected_maps.append((input_image, unit.module))

                    if control_model_type == ControlModelType.T2I_StyleAdapter:
                        control = control['last_hidden_state']

                    if control_model_type == ControlModelType.ReVision:
                        control = control['image_embeds']

                    if control_model_type == ControlModelType.IPAdapter:
                        if model_net.is_plus:
                            controls_ipadapter['hidden_states'].append(control['hidden_states'][-2])
                        else:
                            controls_ipadapter['image_embeds'].append(control['image_embeds'])
                        if hr_control is not None:
                            if model_net.is_plus:
                                hr_controls_ipadapter['hidden_states'].append(hr_control['hidden_states'][-2])
                            else:
                                hr_controls_ipadapter['image_embeds'].append(hr_control['image_embeds'])
                        else:
                            hr_controls_ipadapter = None
                            hr_controls = None
                    else:
                        controls.append(control)
                        if hr_control is not None:
                            hr_controls.append(hr_control)
                        else:
                            hr_controls = None
                
                if control_model_type == ControlModelType.IPAdapter:
                    ipadapter_key = 'hidden_states' if model_net.is_plus else 'image_embeds'
                    controls = {ipadapter_key: torch.cat(controls_ipadapter[ipadapter_key], dim=0)}
                    if controls[ipadapter_key].shape[0] > 1:
                        controls[ipadapter_key] = torch.cat([controls[ipadapter_key], controls[ipadapter_key]], dim=0)
                    if model_net.is_plus:
                        controls[ipadapter_key] = [controls[ipadapter_key], None]
                    if hr_controls_ipadapter is not None:
                        hr_controls = {ipadapter_key: torch.cat(hr_controls_ipadapter[ipadapter_key], dim=0)}
                        if hr_controls[ipadapter_key].shape[0] > 1:
                            hr_controls[ipadapter_key] = torch.cat([hr_controls[ipadapter_key], hr_controls[ipadapter_key]], dim=0)
                        if model_net.is_plus:
                            hr_controls[ipadapter_key] = [hr_controls[ipadapter_key], None]
                else:
                    controls = torch.cat(controls, dim=0)
                    if controls.shape[0] > 1:
                        controls = torch.cat([controls, controls], dim=0)
                    if hr_controls is not None:
                        hr_controls = torch.cat(hr_controls, dim=0)
                        if hr_controls.shape[0] > 1:
                            hr_controls = torch.cat([hr_controls, hr_controls], dim=0)

                preprocessor_dict = dict(
                    name=unit.module,
                    preprocessor_resolution=preprocessor_resolution,
                    threshold_a=unit.threshold_a,
                    threshold_b=unit.threshold_b
                )

                forward_param = ControlParams(
                    control_model=model_net,
                    preprocessor=preprocessor_dict,
                    hint_cond=controls,
                    weight=unit.weight,
                    guidance_stopped=False,
                    start_guidance_percent=unit.guidance_start,
                    stop_guidance_percent=unit.guidance_end,
                    advanced_weighting=None,
                    control_model_type=control_model_type,
                    global_average_pooling=global_average_pooling,
                    hr_hint_cond=hr_controls,
                    soft_injection=control_mode != external_code.ControlMode.BALANCED,
                    cfg_injection=control_mode == external_code.ControlMode.CONTROL,
                )
                forward_params.append(forward_param)

                unit_is_batch = getattr(unit, 'input_mode', InputMode.SIMPLE) == InputMode.BATCH
                if 'inpaint_only' in unit.module:
                    final_inpaint_raws = []
                    final_inpaint_masks = []
                    for i in range(len(controls)):
                        final_inpaint_feed = hr_controls[i] if hr_controls is not None else controls[i]
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
                        final_inpaint_raws.append(final_inpaint_raw)
                        final_inpaint_masks.append(final_inpaint_mask)

                    def inpaint_only_post_processing(x, i):
                        _, H, W = x.shape
                        if Hmask != H or Wmask != W:
                            logger.error('Error: ControlNet find post-processing resolution mismatch. This could be related to other extensions hacked processing.')
                            return x
                        idx = i if unit_is_batch else 0
                        r = final_inpaint_raw[idx].to(x.dtype).to(x.device)
                        m = final_inpaint_mask[idx].to(x.dtype).to(x.device)
                        y = m * x.clip(0, 1) + (1 - m) * r
                        y = y.clip(0, 1)
                        return y

                    post_processors.append(inpaint_only_post_processing)

                if 'recolor' in unit.module:
                    final_feeds = []
                    for i in range(len(controls)):
                        final_feed = hr_control if hr_control is not None else control
                        final_feed = final_feed.detach().cpu().numpy()
                        final_feed = np.ascontiguousarray(final_feed).copy()
                        final_feed = final_feed[0, 0, :, :].astype(np.float32)
                        final_feed = (final_feed * 255).clip(0, 255).astype(np.uint8)
                        Hfeed, Wfeed = final_feed.shape
                        final_feeds.append(final_feed)

                    if 'luminance' in unit.module:

                        def recolor_luminance_post_processing(x, i):
                            C, H, W = x.shape
                            if Hfeed != H or Wfeed != W or C != 3:
                                logger.error('Error: ControlNet find post-processing resolution mismatch. This could be related to other extensions hacked processing.')
                                return x
                            h = x.detach().cpu().numpy().transpose((1, 2, 0))
                            h = (h * 255).clip(0, 255).astype(np.uint8)
                            h = cv2.cvtColor(h, cv2.COLOR_RGB2LAB)
                            h[:, :, 0] = final_feed[i if unit_is_batch else 0]
                            h = cv2.cvtColor(h, cv2.COLOR_LAB2RGB)
                            h = (h.astype(np.float32) / 255.0).transpose((2, 0, 1))
                            y = torch.from_numpy(h).clip(0, 1).to(x)
                            return y

                        post_processors.append(recolor_luminance_post_processing)

                    if 'intensity' in unit.module:

                        def recolor_intensity_post_processing(x, i):
                            C, H, W = x.shape
                            if Hfeed != H or Wfeed != W or C != 3:
                                logger.error('Error: ControlNet find post-processing resolution mismatch. This could be related to other extensions hacked processing.')
                                return x
                            h = x.detach().cpu().numpy().transpose((1, 2, 0))
                            h = (h * 255).clip(0, 255).astype(np.uint8)
                            h = cv2.cvtColor(h, cv2.COLOR_RGB2HSV)
                            h[:, :, 2] = final_feed[i if unit_is_batch else 0]
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
                # Do not support controlllite for sdxl
                # if param.control_model_type == ControlModelType.Controlllite:
                #     param.control_model.hook(
                #         model=unet,
                #         cond=param.hint_cond,
                #         weight=param.weight,
                #         start=param.start_guidance_percent,
                #         end=param.stop_guidance_percent
                #     )

            self.detected_map = detected_maps
            self.post_processors = post_processors

            if os.path.exists(f'{data_path}/tmp/animatediff-frames/'):
                shutil.rmtree(f'{data_path}/tmp/animatediff-frames/')

        def hacked_postprocess_batch(self, p, *args, **kwargs):
            images = kwargs.get('images', [])
            for post_processor in self.post_processors:
                for i in range(len(images)):
                    images[i] = post_processor(images[i], i)
            return

        self.original_controlnet_main_entry = self.cn_script.controlnet_main_entry
        self.original_postprocess_batch = self.cn_script.postprocess_batch
        self.cn_script.controlnet_main_entry = MethodType(hacked_main_entry, self.cn_script)
        self.cn_script.postprocess_batch = MethodType(hacked_postprocess_batch, self.cn_script)


    def restore_cn(self):
        self.cn_script.controlnet_main_entry = self.original_controlnet_main_entry
        self.original_controlnet_main_entry = None
        self.cn_script.postprocess_batch = self.original_postprocess_batch
        self.original_postprocess_batch = None


    def hack(self, params: AnimateDiffProcess):
        if self.cn_script is not None:
            logger.info(f"Hacking ControlNet.")
            self.hack_batchhijack(params)
            self.hack_cn()


    def restore(self):
        if self.cn_script is not None:
            logger.info(f"Restoring ControlNet.")
            self.restore_batchhijack()
            self.restore_cn()
