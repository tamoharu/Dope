import cv2
import sys

sys.path.append('../')
from main.face_modules.swap_face import SwapFace

def test_swap_face():
    source_list = []
    source1 = cv2.imread('./images/elon-1.jpg')
    source_list.append(source1)
    source2 = cv2.imread('./images/elon-2.jpg')
    source_list.append(source2)
    source_3 = cv2.imread('./images/elon-3.jpg')
    source_list.append(source_3)


    target = cv2.imread('./images/mark.jpg')
    swap_face = SwapFace()
    result = swap_face.swap(source_list, target)
    cv2.imshow("Face Swap", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

test_swap_face()