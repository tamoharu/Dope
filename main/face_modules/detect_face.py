from typing import List, Tuple

import main.globals as globals
import main.instances as instances
from main.type import Frame, Bbox, Kps, Score
from main.utils.filesystem import resolve_relative_path
from main.face_modules.model_zoo.yolov8 import Yolov8


def model_router():
    if globals.detect_face_model == 'yolov8':
        if instances.yolov8_instance is None:
            instances.yolov8_instance = Yolov8(
                model_path=resolve_relative_path('../../models/yolov8n-face.onnx'), 
                device=globals.device, 
                score_threshold=globals.score_threshold, 
                iou_threshold=globals.iou_threshold
                )
        return instances.yolov8_instance
    else:
        raise NotImplementedError(f"Model {globals.detect_face_model} not implemented.")


def detect_face(frame: Frame) -> Tuple[List[Bbox], List[Kps], List[Score]]:
    model = model_router()
    return model.predict(frame)