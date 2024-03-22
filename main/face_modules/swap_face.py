from typing import List, Tuple
import threading
import time

import cv2
import numpy as np

import main.globals as globals
import main.instances as instances
import main.face_store as face_store
from main.type import Frame, Kps, Mask, Matrix, Embedding, Template, Size
from main.utils.filesystem import resolve_relative_path
from main.face_modules.detect_face import detect_face
from main.face_modules.embed_face import embed_face
from main.face_modules.mask_face import mask_face
from main.face_modules.enhance_face import enhance_face
from main.face_modules.model_zoo.inswapper import Inswapper


def model_router():
    if globals.swap_face_model == 'inswapper':
        return Inswapper(
            model_path=resolve_relative_path('../../models/inswapper_128.onnx'),
            device=globals.device
        )
    else:
        raise NotImplementedError(f"Model {globals.swap_face_model} not implemented.")


def create_source_embedding(source_frames: List[Frame]):
    with threading.Lock():
        if face_store.source_embedding is None:
            embedding_list = []
            for source_frame in source_frames:
                _, source_kps_list, _ = detect_face(frame=source_frame)
                source_kps = source_kps_list[0]
                embedding = embed_face(frame=source_frame, kps=source_kps)
                embedding_list.append(embedding)
            face_store.source_embedding = np.mean(embedding_list, axis=0)


def swap_face(source_embedding: Embedding, target_frame: Frame) -> Frame:
    start_time = time.time()  # detect_faceの開始時間
    _, target_kps_list, _ = detect_face(frame=target_frame)
    print(f"detect_face took {time.time() - start_time:.2f} seconds")

    temp_frame = target_frame
    for i, target_kps in enumerate(target_kps_list):
        start_time = time.time()  # apply_swapの開始時間
        temp_frame = apply_swap(temp_frame, target_frame, target_kps, source_embedding)
        print(f"apply_swap {i} took {time.time() - start_time:.2f} seconds")
        
        # start_time = time.time()  # apply_enhanceの開始時間
        # temp_frame = apply_enhance(temp_frame, target_kps)
        # print(f"apply_enhance {i} took {time.time() - start_time:.2f} seconds")
    
    return temp_frame


# def apply_swap(temp_frame, target_frame: Frame, target_kps: Kps, embedding: Embedding) -> Frame:
#     model = model_router()
#     crop_frame, affine_matrix = warp_face_kps(target_frame, target_kps, model.model_template, model.model_size)
#     mask_list = []
#     if 'face_occluder' in globals.mask_face_model:
#         crop_mask = mask_face(frame=crop_frame, model_name='face_occluder')
#         mask_list.append(crop_mask)
#     if 'box' in globals.mask_face_model:
#         crop_mask = mask_face(frame=crop_frame, model_name='box')
#         mask_list.append(crop_mask)
#     crop_frame = model.predict(target_crop_frame=crop_frame, source_embedding=embedding)
#     if 'face_parser' in globals.mask_face_model:
#         crop_mask = mask_face(frame=crop_frame, model_name='face_parser')
#         mask_list.append(crop_mask)
#     crop_mask = np.minimum.reduce(mask_list).clip(0, 1)
#     return paste_back(temp_frame, crop_frame, crop_mask, affine_matrix)


def apply_swap(temp_frame, target_frame: Frame, target_kps: Kps, embedding: Embedding) -> Frame:
    start_time = time.time()  # apply_swapの開始時間
    model = model_router()
    print(f"model_router took {time.time() - start_time:.2f} seconds")
    
    start_time = time.time()  # warp_face_kpsの開始時間
    crop_frame, affine_matrix = warp_face_kps(target_frame, target_kps, model.model_template, model.model_size)
    print(f"warp_face_kps took {time.time() - start_time:.2f} seconds")
    
    mask_list = []
    if 'face_occluder' in globals.mask_face_model:
        start_time = time.time()  # mask_face (face_occluder)の開始時間
        crop_mask = mask_face(frame=crop_frame, model_name='face_occluder')
        mask_list.append(crop_mask)
        print(f"mask_face (face_occluder) took {time.time() - start_time:.2f} seconds")
    
    if 'box' in globals.mask_face_model:
        start_time = time.time()  # mask_face (box)の開始時間
        crop_mask = mask_face(frame=crop_frame, model_name='box')
        mask_list.append(crop_mask)
        print(f"mask_face (box) took {time.time() - start_time:.2f} seconds")
    
    start_time = time.time()  # model.predictの開始時間
    crop_frame = model.predict(target_crop_frame=crop_frame, source_embedding=embedding)
    print(f"model.predict took {time.time() - start_time:.2f} seconds")
    
    if 'face_parser' in globals.mask_face_model:
        start_time = time.time()  # mask_face (face_parser)の開始時間
        crop_mask = mask_face(frame=crop_frame, model_name='face_parser')
        mask_list.append(crop_mask)
        print(f"mask_face (face_parser) took {time.time() - start_time:.2f} seconds")
    
    start_time = time.time()  # np.minimum.reduceの開始時間
    crop_mask = np.minimum.reduce(mask_list).clip(0, 1)
    print(f"np.minimum.reduce took {time.time() - start_time:.2f} seconds")
    
    start_time = time.time()  # paste_backの開始時間
    result_frame = paste_back(temp_frame, crop_frame, crop_mask, affine_matrix)
    print(f"paste_back took {time.time() - start_time:.2f} seconds")
    
    return result_frame


def warp_face_kps(temp_frame: Frame, kps: Kps, model_template: Template, model_size: Size) -> Tuple[Frame, Matrix]:
    normed_template = model_template * model_size
    affine_matrix = cv2.estimateAffinePartial2D(kps, normed_template, method = cv2.RANSAC, ransacReprojThreshold = 100)[0]
    crop_frame = cv2.warpAffine(temp_frame, affine_matrix, model_size, borderMode = cv2.BORDER_REPLICATE, flags = cv2.INTER_AREA)
    return crop_frame, affine_matrix


def paste_back(target_frame: Frame, crop_frame: Frame, crop_mask: Frame, affine_matrix: Matrix) -> Frame:
    inverse_matrix = cv2.invertAffineTransform(affine_matrix)
    temp_frame_size = target_frame.shape[:2][::-1]
    inverse_crop_mask = cv2.warpAffine(crop_mask, inverse_matrix, temp_frame_size).clip(0, 1)
    inverse_crop_frame = cv2.warpAffine(crop_frame, inverse_matrix, temp_frame_size, borderMode = cv2.BORDER_REPLICATE)
    paste_frame = target_frame.copy()
    paste_frame[:, :, 0] = inverse_crop_mask * inverse_crop_frame[:, :, 0] + (1 - inverse_crop_mask) * target_frame[:, :, 0]
    paste_frame[:, :, 1] = inverse_crop_mask * inverse_crop_frame[:, :, 1] + (1 - inverse_crop_mask) * target_frame[:, :, 1]
    paste_frame[:, :, 2] = inverse_crop_mask * inverse_crop_frame[:, :, 2] + (1 - inverse_crop_mask) * target_frame[:, :, 2]
    return paste_frame


def apply_enhance(temp_frame, kps: Kps) -> Frame:
    crop_frame, affine_matrix = enhance_face(temp_frame, kps)
    crop_mask = mask_face(frame=crop_frame, model_name='face_occluder')
    paste_frame = paste_back(temp_frame, crop_frame, crop_mask, affine_matrix)
    face_enhancer_blend = 1 - (80 / 100)
    return cv2.addWeighted(temp_frame, face_enhancer_blend, paste_frame, 1 - face_enhancer_blend, 0)