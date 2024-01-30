from typing import List

import cv2
import numpy as np

from main.types import Frame, Output, Mask
from main.face_modules.model_zoo._base_model import OnnxBaseModel

class FaceOccluder(OnnxBaseModel):
    '''
    input
    in_face:0: ['unk__359', 256, 256, 3]

    output
    out_mask:0: ['unk__360', 256, 256, 1]
    '''
    def __init__(self, model_path: str, device: List[str]):
        print('FaceOccluder init')
        super().__init__(model_path, device)
        self.model_size = (256, 256)

    
    def predict(self, frame: Frame) -> Mask:
        prepare_frame = self.pre_process(frame)
        output = self.forward(prepare_frame)
        mask = self.post_process(output, frame)
        return mask


    def pre_process(self, frame: Frame) -> Frame:
        frame = cv2.resize(frame, self.model_size, interpolation = cv2.INTER_LINEAR)
        frame = np.expand_dims(frame, axis = 0).astype(np.float32) / 255
        return frame


    def forward(self, frame: Frame) -> Output:
        with self.semaphore:
            output = self.session.run(None,
            {
                self.input_names[0]: frame,
            })
        return output
    

    def post_process(self, output: Output, frame: Frame) -> Mask:
        mask = output[0][0]
        occlusion_mask = mask.transpose(0, 1, 2).clip(0, 1).astype(np.float32)
        occlusion_mask = cv2.resize(occlusion_mask, frame.shape[:2][::-1])
        return occlusion_mask