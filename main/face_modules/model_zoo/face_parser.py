from typing import List

import cv2
import numpy as np

from main.types import Frame, Output, Mask, MaskFaceRegion
from main.face_modules.model_zoo._base_model import OnnxBaseModel


class FaceParser(OnnxBaseModel):
    '''
    input
    input: [1, 3, 512, 512]

    output
    out: [1, 19, 512, 512]
    392: [1, 19, 512, 512]
    402: [1, 19, 512, 512]
    '''
    def __init__(self, model_path: str, device: List[str]):
        print('FaceParser init')
        super().__init__(model_path, device)
        self.mask_regions =\
        {
        'skin': 1,
        'left-eyebrow': 2,
        'right-eyebrow': 3,
        'left-eye': 4,
        'right-eye': 5,
        'eye-glasses': 6,
        'nose': 10,
        'mouth': 11,
        'upper-lip': 12,
        'lower-lip': 13
        }


    def predict(self, frame: Frame, face_mask_regions: List[MaskFaceRegion]) -> Mask:
        prepare_frame = self.pre_process(frame)
        output = self.forward(prepare_frame)
        mask = self.post_process(output, frame, face_mask_regions)
        return mask
    

    def pre_process(self, frame: Frame) -> Frame:
        frame = cv2.flip(cv2.resize(frame, (512, 512)), 1)
        frame = np.expand_dims(frame, axis = 0).astype(np.float32)[:, :, ::-1] / 127.5 - 1
        frame = frame.transpose(0, 3, 1, 2)
        return frame
    

    def forward(self, frame: Frame) -> Output:
        with self.semaphore:
            output = self.session.run(None,
            {
                self.input_names[0]: frame,
            })
        return output
    

    def post_process(self, output: Output, frame: Frame, face_mask_regions: List[MaskFaceRegion]) -> Mask:
        mask = output[0][0]
        mask = np.isin(mask.argmax(0), [ self.mask_regions[region] for region in face_mask_regions ])
        mask = cv2.resize(mask.astype(np.float32), frame.shape[:2][::-1])
        return mask
