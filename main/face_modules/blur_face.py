from main.face_modules.detect_face import detect_face

class FaceBlur:
    def __init__(self, blur_strength: int = 100):
        self.blur_strength = blur_strength

    
    def blur(self):
        bbox, kps, _ = detect_face(frame=temp_frame)
        