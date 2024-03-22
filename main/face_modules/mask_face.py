import cv2
import numpy as np

import main.globals as globals
import main.instances as instances
from main.type import Frame, Mask, MaskFaceModel
from main.utils.filesystem import resolve_relative_path
from main.face_modules.model_zoo.face_occluder import FaceOccluder
from main.face_modules.model_zoo.face_parser import FaceParser


def mask_face(frame: Frame, model_name: MaskFaceModel) -> Mask:
    if model_name == 'face_occluder':
        model = FaceOccluder(
            model_path=resolve_relative_path('../../models/face_occluder.onnx'),
            device=globals.device
        )
        occlude_mask = model.predict(frame)
        return occlude_mask
    if model_name == 'face_parser':
        model = FaceParser(
            model_path=resolve_relative_path('../../models/face_parser.onnx'),
            device=globals.device
        )
        region_mask = model.predict(frame, globals.mask_face_regions)
        return region_mask
    if model_name == 'box':
        crop_size = frame.shape[:2][::-1]
        mask_padding = (0, 0, 0, 0)
        mask_blur = 0.3
        blur_amount = int(crop_size[0] * 0.5 * mask_blur)
        blur_area = max(blur_amount // 2, 1)
        box_mask = np.ones(crop_size, np.float32)
        box_mask[:max(blur_area, int(crop_size[1] * mask_padding[0] / 100)), :] = 0
        box_mask[-max(blur_area, int(crop_size[1] * mask_padding[2] / 100)):, :] = 0
        box_mask[:, :max(blur_area, int(crop_size[0] * mask_padding[3] / 100))] = 0
        box_mask[:, -max(blur_area, int(crop_size[0] * mask_padding[1] / 100)):] = 0
        box_mask = cv2.GaussianBlur(box_mask, (0, 0), blur_amount * 0.25)
        return box_mask