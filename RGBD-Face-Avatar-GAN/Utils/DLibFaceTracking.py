import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("Utils/shape_predictor_68_face_landmarks.dat")

def create2DLandmarks(imageTensor, show=False):

    input = imageTensor.detach().numpy()

    if input.shape[0] < input.shape[2]:
        input = input.transpose(1, 2, 0)

    pred = np.zeros((68, 2))

    input = input.astype("uint8")
    faces = detector(input)


    for face in faces:

        landmarks = predictor(input, face)

        for n in range(0, 68):
            pred[n, 0] = landmarks.part(n).x
            pred[n, 1] = landmarks.part(n).y

    if show == True:
        h, w, c = input.shape

        landmarks_preview = np.zeros((h, w), dtype="float")

        for lm in pred:
            x = int(lm[1])
            y = int(lm[0])
            landmarks_preview[x - 4:x + 4, y - 4:y + 4] = 1

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Landmarks')
        ax1.imshow(input / 255)
        ax2.imshow(landmarks_preview)
        plt.show(block=False)
        plt.pause(3)
        plt.close()

    print(pred)
    return torch.Tensor(pred)


if __name__ == "__main__":
    #input = io.imread('testbild2.png')
    input = cv2.imread('C:/Users/Alexander Pech/Desktop/142207-phones-feature-what-is-apple-face-id-and-how-does-it-work-image1-5d72kjh6lq.jpg', cv2.IMREAD_UNCHANGED)
    input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
    testTensor = torch.Tensor(input)
    landmarks = create2DLandmarks(testTensor, show=True)
    print(landmarks.shape)