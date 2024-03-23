from typing import List, Tuple
import threading

import cv2
import numpy as np

import main.instances as instances
from main.type import Frame, Kps, Output, Matrix
from main.face_modules.model_zoo._base_model import OnnxBaseModel


class Codeformer(OnnxBaseModel):
    '''
    input
    input: [1, 3, 512, 512]
    weight: []

    output
    output: [1, 3, 512, 512]
    logits: [1, 256, 1024]
    style_feat: [1, 256, 16, 16]
    '''
    _lock = threading.Lock()
    def __init__(self, model_path: str, device: List[str]):
        with Codeformer._lock:
            if instances.codeformer_instance is None:
                super().__init__(model_path, device)
                self.model_size = (512, 512)
                self.model_template = np.array(
                [
                    [ 0.37691676, 0.46864664 ],
                    [ 0.62285697, 0.46912813 ],
                    [ 0.50123859, 0.61331904 ],
                    [ 0.39308822, 0.72541100 ],
                    [ 0.61150205, 0.72490465 ]
                ])
                instances.codeformer_instance = self
            else:
                self.__dict__ = instances.codeformer_instance.__dict__


    def predict(self, frame: Frame, kps: Kps) -> Frame:
        crop_frame = self.pre_process(frame, kps)
        output = self.forward(crop_frame)
        crop_frame = self.post_process(output)
        return crop_frame
    
    
    def forward(self, crop_frame : Frame) -> Output:
        weight = np.array([ 1 ], dtype = np.double)
        with self.semaphore:
            output = self.session.run(None, {
                self.input_names[0]: crop_frame,
                self.input_names[1]: weight
            })
        return output


    def pre_process(self, frame : Frame, kps: Kps) -> Frame:
        crop_frame, _ = self.warp_face_kps(frame, kps)
        crop_frame = crop_frame.astype(np.float32)[:, :, ::-1] / 255.0
        crop_frame = (crop_frame - 0.5) / 0.5
        crop_frame = np.expand_dims(crop_frame.transpose(2, 0, 1), axis = 0).astype(np.float32)
        return crop_frame


    def post_process(self, output: Output) -> Frame:
        crop_frame = output[0][0]
        crop_frame = np.clip(crop_frame, -1, 1)
        crop_frame = (crop_frame + 1) / 2
        crop_frame = crop_frame.transpose(1, 2, 0)
        crop_frame = (crop_frame * 255.0).round()
        crop_frame = crop_frame.astype(np.uint8)[:, :, ::-1]
        return crop_frame


    def warp_face_kps(self, frame: Frame, kps: Kps) -> Tuple[Frame, Matrix]:
        normed_template = np.array(self.model_template) * np.array(self.model_size)
        affine_matrix = cv2.estimateAffinePartial2D(np.array(kps), normed_template, method = cv2.RANSAC, ransacReprojThreshold = 100)[0]
        crop_frame = cv2.warpAffine(frame, affine_matrix, self.model_size, borderMode = cv2.BORDER_REPLICATE, flags = cv2.INTER_AREA)
        return crop_frame, affine_matrix
    

    def get_affine_matrix(self, frame: Frame, kps: Kps) -> Matrix:
        _, affine_matrix = self.warp_face_kps(frame, kps)
        return affine_matrix