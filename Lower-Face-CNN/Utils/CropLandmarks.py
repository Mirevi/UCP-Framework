import numpy as np
class CropLandmarks:
    def __init__(self, landmarks):
        self.landmarks = landmarks

    def crop_landmarks_face_bottom_half(self):
        # https://pyimagesearch.com/wp-content/uploads/2017/04/facial_landmarks_68markup.jpg
        left_face_landmarks = list(range(0,6)) # index of landmarks 1-3
        right_face_landmarks = list(range(28,34)) # index of landmarks 15-17
        eye_landmarks = list(range(72, 96)) # index of landmarks 37-48
        nose_landmarks = list(range(54,62)) # index of landmarks 28-31
        eye_brows_landmarks = list(range(34,54)) # index of landmarks 18-27
        removable_landmarks = left_face_landmarks + right_face_landmarks + eye_landmarks + nose_landmarks + eye_brows_landmarks

        croped_landmarks = np.delete(self.landmarks, removable_landmarks)
        return croped_landmarks
