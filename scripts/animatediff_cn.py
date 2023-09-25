from typing import Optional

import numpy as np
from modules import images, masking
from modules.processing import (StableDiffusionProcessing,
                                StableDiffusionProcessingImg2Img,
                                StableDiffusionProcessingTxt2Img)
from PIL import Image


class AnimateDiffControl:
    """
    Everything AnimateDiff need to do to properly V2V via ControlNet.
    It has to the following tasks:
    1. generate all control maps in `before_process`, set CN control as the first, do not use CN batch mode.
    2. inject control maps into ControlNet Script.latest_network in `before_process_batch`
    3. optionally save all control maps in `postprocess`
    """
    def __init__(self):
        self.control = None
        self.detected_maps = []
    
    def gen_control_map(self, p: StableDiffusionProcessing):
        # various types of input sources
        # 1. a video from AnimateDiff parameters
        # 2. a list of images from ControlNet parameters
        # 3. for single image control, let CN handle it
        # 4. do not support img2img batch mode

        # import ControlNet packages
        from scripts import external_code
        from scripts.controlnet import (Script, image_has_mask, prepare_mask,
                                        set_numpy_seed)
        from scripts.logging import logger

        for idx, unit in enumerate(self.enabled_units):

            resize_mode = external_code.resize_mode_from_value(unit.resize_mode)
            control_mode = external_code.control_mode_from_value(unit.control_mode)

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
            else:
                hr_control = None

            if is_image:
                control, detected_map = Script.detectmap_proc(detected_map, unit.module, resize_mode, h, w)
                self.detected_maps.append((detected_map, unit.module))
            else:
                control = detected_map
                self.detected_maps.append((input_image, unit.module))
