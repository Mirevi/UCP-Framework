import numpy as np
import cv2

class Camera:

    def __init__(self, _camera):
        self.camera = _camera

        self.camera_matrix = {
            "azure_kinect_rgb": [[764.8213949097228, 0, 586.9704721996291],
                                 [0, 787.2213463953578, 469.7783408003594],
                                 [0, 0, 1]],
            "webcam": [[836.7084186420553, 0, 312.4524598264829],
                       [0, 839.2824786745844, 254.9663907756732],
                       [0, 0, 1]],
            "Weitwinkel": [[337.318228880027, 0, 289.2694151449546],
                           [0, 332.4867154736427, 250.6453350883297],
                           [0, 0, 1]]#[[329.5102304994368, 0, 350.3954627226036],
                          # [0, 324.3627558435883, 224.9580911282809],
                          # [0, 0, 1]]
        }
        self.distortion_coefficients = {
            "azure_kinect_rgb": [[0.2001623500255367, 0.1042384526966689, 0.07375628115376121, -0.02761273514622563, -0.1501943390678023]],
            "webcam": [[-0.04020993812637849, 0.514599635573169, -0.00163616260153157, -0.005393102963345314, -1.372108570040381]],
            "Weitwinkel": [[-0.09117239964970096, -0.04864146997358993, -0.0002963330366070298, -0.0004242255249618672, 0.03162596715988156]]#[[-0.1178943670415475, 0.002065241539559859, -1.1539759331937e-05, 0.0004326343932169886, -0.0004568805768271211]]
        }


    def __call__(self, img):

        h, w = img.shape[:2]

        mtx = np.array(self.camera_matrix[self.camera])
        dist = np.array(self.distortion_coefficients[self.camera])

        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        # crop the image
        x, y, w, h = roi
        return dst[y:y + h, x:x + w]

if __name__ == "__main__":
    camera = Camera("Weitwinkel")

    path = "Weitwinkel.png"
    img = cv2.imread(path)
    img = camera(img)
    img = cv2.imwrite("_" + path, img)


