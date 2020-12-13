import matplotlib.pyplot as plt
import numpy as np
import cv2

def show(image,landmarks=None, h=1, w=1):

    image = image.detach().cpu().numpy().transpose(1, 2, 0)
    h, w, c = image.shape
    image = image.reshape(h, w)
    #cv2.cvtColor(image, cv2.COLOR_GRAY2RGB, image)

    if landmarks is not None:
        landmarks = landmarks.detach().cpu().numpy()
        for i in range(0, len(landmarks), 2):
            x = np.clip(int(landmarks[i] * w), 0, image.shape[1])
            y = np.clip(int(landmarks[i+1] * h), 0, image.shape[0])
            image[y-1:y+1, x-1:x+1 ] = 1 # np.array([0.0, 0.0, 1.0])
    plt.imshow(image)
    plt.show(block=False)
    plt.pause(3)
    plt.close()

    return image