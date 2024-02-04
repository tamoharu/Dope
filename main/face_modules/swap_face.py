from typing import List, Tuple

import cv2
import numpy as np

import main.globals as globals
import main.instances as instances
import main.face_store as face_store
from main.type import Frame, Kps, Mask, Matrix, Embedding
from main.utils.filesystem import resolve_relative_path
from main.face_modules.detect_face import detect_face
from main.face_modules.embed_face import embed_face
from main.face_modules.mask_face import mask_face
from main.face_modules.enhance_face import enhance_face
from main.face_modules.model_zoo.inswapper import Inswapper


class SwapFace:
    def __init__(self):
        if instances.inswapper_instance is None:
            instances.inswapper_instance = Inswapper(
                model_path=resolve_relative_path('../../models/swap_face/inswapper_128.onnx'),
                device=globals.device
            )
        self.model = instances.inswapper_instance
        self.model_template, self.model_size = self.model.get_model_info()


    def swap(self, source_frames: List[Frame], target_frame: Frame) -> Frame:
        self.create_source_embedding(source_frames)
        _, target_kps_list, _ = detect_face(frame=target_frame)
        source_embedding_list = self.prepare_source_embedding_list(target_frame, target_kps_list)
        self.temp_frame = target_frame
        for i, target_kps in enumerate(target_kps_list):
            temp_frame = self.apply_swap(target_frame, target_kps, source_embedding_list[i])
            temp_frame = self.apply_enhance(temp_frame, target_kps)
            self.temp_frame = temp_frame
        return self.temp_frame


    def apply_enhance(self, temp_frame, kps: Kps) -> Frame:
        crop_frame, affine_matrix = enhance_face(temp_frame, kps)
        crop_mask = mask_face(frame=crop_frame, model_name='face_occluder')
        paste_frame = self.paste_back(temp_frame, crop_frame, crop_mask, affine_matrix)
        face_enhancer_blend = 1 - (80 / 100)
        return cv2.addWeighted(temp_frame, face_enhancer_blend, paste_frame, 1 - face_enhancer_blend, 0)


    def create_source_embedding(self, source_frames: List[Frame]):
        if face_store.source_embedding is None:
            embedding_list = []
            for source_frame in source_frames:
                _, source_kps_list, _ = detect_face(frame=source_frame)
                source_kps = source_kps_list[0]
                embedding = embed_face(frame=source_frame, kps=source_kps)
                embedding_list.append(embedding)
            face_store.source_embedding = np.mean(embedding_list, axis=0)


    def prepare_source_embedding_list(self, target_frame: Frame, target_kps_list: List[Kps]) -> List[Embedding]:
        source_embedding_list = []
        if globals.blend_strangth < 100:
            strength = globals.blend_strangth / 100
            for target_kps in target_kps_list:
                embedding = embed_face(frame=target_frame, kps=target_kps)
                source_embedding = face_store.source_embedding * strength + embedding * (1 - strength)
                source_embedding_list.append(source_embedding)
        else:
            for _ in target_kps_list:
                source_embedding_list.append(face_store.source_embedding)
        return source_embedding_list
        
    
    def apply_swap(self, target_frame: Frame, target_kps: Kps, embedding: Embedding) -> Frame:
        crop_frame, affine_matrix = self.warp_face_kps(target_frame, target_kps)
        mask_list = []
        if 'face_occluder' in globals.mask_face_model:
            crop_mask = mask_face(frame=crop_frame, model_name='face_occluder')
            mask_list.append(crop_mask)
        if 'box' in globals.mask_face_model:
            crop_mask = mask_face(frame=crop_frame, model_name='box')
            mask_list.append(crop_mask)
        crop_frame = self.model.predict(target_crop_frame=crop_frame, source_embedding=embedding)
        if 'face_parser' in globals.mask_face_model:
            crop_mask = mask_face(frame=crop_frame, model_name='face_parser')
            mask_list.append(crop_mask)
        crop_mask = np.minimum.reduce(mask_list).clip(0, 1)
        return self.paste_back(self.temp_frame, crop_frame, crop_mask, affine_matrix)


    def warp_face_kps(self, temp_frame: Frame, kps: Kps) -> Tuple[Frame, Matrix]:
        normed_template = self.model_template * self.model_size
        affine_matrix = cv2.estimateAffinePartial2D(kps, normed_template, method = cv2.RANSAC, ransacReprojThreshold = 100)[0]
        crop_frame = cv2.warpAffine(temp_frame, affine_matrix, self.model_size, borderMode = cv2.BORDER_REPLICATE, flags = cv2.INTER_AREA)
        return crop_frame, affine_matrix


    def paste_back(self, target_frame: Frame, crop_frame: Frame, crop_mask: Frame, affine_matrix: Matrix) -> Frame:
        inverse_matrix = cv2.invertAffineTransform(affine_matrix)
        temp_frame_size = target_frame.shape[:2][::-1]
        inverse_crop_mask = cv2.warpAffine(crop_mask, inverse_matrix, temp_frame_size).clip(0, 1)
        inverse_crop_frame = cv2.warpAffine(crop_frame, inverse_matrix, temp_frame_size, borderMode = cv2.BORDER_REPLICATE)
        paste_frame = target_frame.copy()
        paste_frame[:, :, 0] = inverse_crop_mask * inverse_crop_frame[:, :, 0] + (1 - inverse_crop_mask) * target_frame[:, :, 0]
        paste_frame[:, :, 1] = inverse_crop_mask * inverse_crop_frame[:, :, 1] + (1 - inverse_crop_mask) * target_frame[:, :, 1]
        paste_frame[:, :, 2] = inverse_crop_mask * inverse_crop_frame[:, :, 2] + (1 - inverse_crop_mask) * target_frame[:, :, 2]
        return paste_frame