from typing import Tuple

import main.globals as globals
import main.instances as instances
from main.types import Frame, Kps, Matrix
from main.utils.filesystem import resolve_relative_path
from main.face_modules.model_zoo.codeformer import Codeformer


def enhance_face(target_frame: Frame, kps: Kps) -> Tuple[Frame, Matrix]:
    if globals.enhance_face_model == 'codeformer':
        if instances.codeformer_instance is None:
            instances.codeformer_instance = Codeformer(
                model_path=resolve_relative_path('../../models/enhance_face/codeformer.onnx'),
                device=globals.device
            )
        enhanced_frame = instances.codeformer_instance.predict(target_frame, kps)
        affine_matrix = instances.codeformer_instance.get_affine_matrix(target_frame, kps)
        return enhanced_frame, affine_matrix
    else:
        raise NotImplementedError(f"Model {globals.enhance_face_model} not implemented.")