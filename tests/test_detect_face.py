import cv2
import sys

sys.path.append('../')
from main.face_modules.detect_face import detect_face

def test_detect_face():
    temp_frame = cv2.imread('./images/width.jpg')
    bbox, kps, score = detect_face(frame=temp_frame)
    print(kps)
    for box, keypoints, sc in zip(bbox, kps, score):
        start_point = (int(box[0]), int(box[1]))
        end_point = (int(box[2]), int(box[3]))
        color = (255, 0, 0)
        thickness = 1
        temp_frame = cv2.rectangle(temp_frame, start_point, end_point, color, thickness)

        score_text = f"{sc:.2f}"
        cv2.putText(temp_frame, score_text, (int(box[0]), int(box[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 1)

        for kp in keypoints:
            kp_x, kp_y = int(kp[0]), int(kp[1])
            cv2.circle(temp_frame, (kp_x, kp_y), 2, (0, 255, 0), -1)

    cv2.imshow("Face Detection", temp_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

test_detect_face()
