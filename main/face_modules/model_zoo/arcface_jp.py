from typing import List

import numpy as np
import cv2
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
import torch

from main.type import Frame, Kps, Embedding, Output
from main.face_modules.model_zoo._base_model import OnnxBaseModel


class ArcfaceJP(OnnxBaseModel):
    '''
    input
    x.1: [1, 3, 224, 224]

    output
    1170: [1, 512]
    '''
    def __init__(self, model_path: str, device: List[str]):
        print('ArcfaceJP init')
        super().__init__(model_path, device)
        self.model_size = (224, 224)
        self.model_template = np.array(
        [
            [ 0.36167656, 0.40387734 ],
            [ 0.63696719, 0.40235469 ],
            [ 0.50019687, 0.56044219 ],
            [ 0.38710391, 0.72160547 ],
            [ 0.61507734, 0.72034453 ]
        ]),
    

    def predict(self, frame: Frame, kps: Kps) -> Embedding:
        crop_frame = self.pre_process(frame, kps)
        output = self.forward(crop_frame)
        embedding = self.post_process(output)
        return embedding
    

    def pre_process(self, frame: Frame, kps: Kps) -> Frame:
        crop_frame = self.transform(frame)
        crop_frame = crop_frame.astype(np.float32) / 127.5 - 1
        crop_frame = crop_frame[:, :, ::-1]
        crop_frame = np.expand_dims(crop_frame, axis = 0)
        return crop_frame


    def forward(self, frame: Frame) -> Output:
        with self.semaphore:
            output = self.session.run(None,
            {
                self.input_names[0]: frame,
            })
        return output
    

    def post_process(self, output: Output) -> Embedding:
        return output[0].ravel()

    
    def transform(self, frame) -> Frame:
        mean_value = [0.485, 0.456, 0.406]
        std_value = [0.229, 0.224, 0.225]
        # frame を PIL Image に変換
        frame = to_pil_image(frame)
        transform = transforms.Compose([
            transforms.Resize(self.model_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_value, std=std_value)
        ])

        frame = transform(frame)
        # Tensor を numpy 配列に変換する必要がある場合
        frame = frame.numpy()
        return frame