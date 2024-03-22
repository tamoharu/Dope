from typing import Tuple

import main.globals as globals
import main.instances as instances
from main.type import Frame, Kps, Matrix
from main.utils.filesystem import resolve_relative_path
from main.face_modules.model_zoo.codeformer import Codeformer


def model_router():
    if globals.enhance_face_model == 'codeformer':
        return Codeformer(
            model_path=resolve_relative_path('../../models/codeformer.onnx'),
            device=globals.device
        )
    else:
        raise NotImplementedError(f"Model {globals.enhance_face_model} not implemented.")
    

def enhance_face(target_frame: Frame, kps: Kps) -> Tuple[Frame, Matrix]:
    model = model_router()
    enhanced_frame = model.predict(target_frame, kps)
    affine_matrix = model.get_affine_matrix(target_frame, kps)
    return enhanced_frame, affine_matrix