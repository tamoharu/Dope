import main.globals as globals
import main.instances as instances
from main.type import Frame, Kps, Embedding
from main.utils.filesystem import resolve_relative_path
from main.face_modules.model_zoo.arcface_inswapper import ArcfaceInswapper


def embed_face(frame: Frame, kps: Kps) -> Embedding:
    if globals.swap_face_model == 'inswapper':
        if instances.arcface_inswapper_instance is None:
            instances.arcface_inswapper_instance = ArcfaceInswapper(
                model_path=resolve_relative_path('../../models/arcface_w600k_r50.onnx'),
                device=globals.device
            )
        return instances.arcface_inswapper_instance.predict(frame, kps)
    else:
        raise NotImplementedError(f"Model {globals.swap_face_model} not implemented.")