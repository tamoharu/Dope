import cv2
import sys

sys.path.append('../')
from main.face_modules.swap_face import SwapFace

def test_swap_face():
    source_list = []
    source1 = cv2.imread('./images/hou-1.JPG')
    source_list.append(source1)
    source2 = cv2.imread('./images/hou-2.JPG')
    source_list.append(source2)
    source_3 = cv2.imread('./images/hou-3.JPG')
    source_list.append(source_3)
    source_4 = cv2.imread('./images/hou-4.JPG')
    source_list.append(source_4)
    # source_5 = cv2.imread('./images/hou-5.JPG')
    # source_list.append(source_5)
    # source_6 = cv2.imread('./images/hou-6.JPG')
    # source_list.append(source_6)
    # source_7 = cv2.imread('./images/hou-7.JPG')
    # source_list.append(source_7)
    # source_8 = cv2.imread('./images/hou-8.JPG')
    # source_list.append(source_8)


    target = cv2.imread('./images/sample.jpg')
    swap_face = SwapFace()
    result = swap_face.swap(source_list, target)
    cv2.imshow("Face Swap", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

test_swap_face()