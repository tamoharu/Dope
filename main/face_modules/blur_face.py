from main.face_modules.detect_face import detect_face

    
def blur_face(temp_frame):
    bbox, kps, _ = detect_face(frame=temp_frame)
    