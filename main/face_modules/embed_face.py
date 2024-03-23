import main.globals as globals
from main.type import Frame, Kps, Embedding
from main.utils.filesystem import resolve_relative_path
from main.face_modules.model_zoo.arcface_inswapper import ArcfaceInswapper


def model_router():
    if globals.swap_face_model == 'inswapper':
        return ArcfaceInswapper(
            model_path=resolve_relative_path('../../models/arcface_w600k_r50.onnx'),
            device=globals.device
        )
    else:
        raise NotImplementedError(f"Model {globals.swap_face_model} not implemented.")
    

def embed_face(frame: Frame, kps: Kps) -> Embedding:
    model = model_router()
    return model.predict(frame, kps)
