import cv2
import sys
import numpy as np
sys.path.append('../')
from main.face_modules.embed_face import embed_face

def test_embed_face():
    frame = cv2.imread('./images/width.jpg')
    kps = [[363.99844, 278.33057],
        [489.2065 , 273.31516],
        [415.13513, 358.81454],
        [380.81882, 424.8471 ],
        [485.6311 , 420.6566 ]]
    embedding = embed_face(frame=frame, kps=np.array(kps))
    print(embedding)

test_embed_face()