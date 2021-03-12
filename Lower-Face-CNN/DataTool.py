import cv2
import face_alignment
import os
import uuid
import glob
import Utils.Camera as cam
import numpy as np
import pandas as pd

PATH2VID = "data/Jannik-28-10-20/Jannik-28-10-20.mp4"
ROTATE = False
SHOWFRAMES = True
scaleBoundingBox = 1.1

#subtract videoname from path and set it as dirname for further processing
dirname = PATH2VID[::-1]
notUsed, notUsed1, dirname = dirname.partition("/")
dirname = dirname[::-1]
dirname = dirname + "/"

class CSVGenerator:
    def __init__(self):
        self.csv = ""
        self.rows = []

    def add_data(self, id, landmark):
        row = np.concatenate((np.array([id]), np.array(landmark[0]).reshape(-1)), axis=0)
        self.rows.append(row)

    def export(self,name):
        all = np.array([self.rows]).reshape(-1,137)
        df = pd.DataFrame(all)
        df.to_csv(name)

    def read_csv(self, name ):
        self.csv = pd.read_csv(name, index_col=0)

        ids = []
        landmarks = []
        for i in range(len(self.csv)):
            ids.append([self.csv.iloc[i, 0]])
            landmarks_values = []
            for x in range(1, len(self.csv.columns)):
                landmarks_values.append(self.csv.iloc[i, x])
            landmarks.append(landmarks_values)

        row = np.concatenate((np.array(ids), np.array(landmarks)), axis=1)

        return row

    def get_shape(self):
        print(self.csv.shape)


def show(img, landmarks):
    for i in landmarks[0]:
        cv2.circle(img, (int(i[0]), int(i[1])), 3, (255, 0, 0))
    cv2.imshow('frame', img)
    cv2.waitKey(1)
    return img

if __name__ == "__main__":
    #if action == 1:
        # Select a camera type. Caddx EOS 2 is "Weitwinkel"
        cam = cam.Camera("Weitwinkel")
        # Create output dir and if files in it, delete them
        dirnameOutput = dirname + "output"
        if not os.path.exists(dirnameOutput):
            os.mkdir(dirnameOutput)
        else:
            files = glob.glob(dirnameOutput + '/*')
            for f in files:
                os.remove(f)

        csv = CSVGenerator()
        fa_v = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda:0', flip_input=True)
        vidcap = cv2.VideoCapture(PATH2VID, 0)
        frame_list = []
        allFacialLandmarks = []
        while (vidcap.isOpened()):
            hasFrames, image = vidcap.read()
            if hasFrames:  # Lets FAN detect landmarks on each frame
                image = cam(image)
                # Contrast adjustment (test for better landmarks results)
                # alpha = 0.5  # Contrast control (1.0-3.0)
                # beta = 0  # Brightness control (0-100)
                # image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
                if ROTATE:
                    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                preds = fa_v.get_landmarks_from_image(image)
                # better use get_landmarks_from_image() because get_landmarks is deprecated -> Todo: Test
                allFacialLandmarks.append(preds)
                if preds == None:
                    continue

                id = uuid.uuid4()
                cv2.imwrite(os.path.join(dirnameOutput + "/", str(id) + ".png"), image) # Frame speichern
                csv.add_data(id, preds)
                if SHOWFRAMES:
                    frame = show(image, preds)
            else:
                break

        vidcap.release()
        csv.export(dirname + "landmark.csv")

        # Calc mouth bounding box
        # For better debug prints
        # np.set_printoptions(threshold=sys.maxsize) # import sys if uncomment
        # convert in np.array
        allFacialLandmarks = np.array(allFacialLandmarks)
        # reduce dimensionaliyt from w, x, y, z to w, y, z (FAN creates a weired array)
        allFacialLandmarks = allFacialLandmarks[:, -1]
        # take only the relevant lower face landmarks
        onlyLowerFaceLandmarks = np.concatenate([allFacialLandmarks[:, 3:13], allFacialLandmarks[:, 31:35]], 1)
        onlyLowerFaceLandmarks = np.concatenate([onlyLowerFaceLandmarks , allFacialLandmarks[:, 48:67]], 1)
        # Min and max for bounding box
        landmarksMinX = onlyLowerFaceLandmarks[:, :, 0:1].min()
        landmarksMaxX = onlyLowerFaceLandmarks[:, :, 0:1].max()
        landmarksMinY = onlyLowerFaceLandmarks[:, :, 1:].min()
        landmarksMaxY = onlyLowerFaceLandmarks[:, :, 1:].max()

    #elif action == 2:
        # read and save array TODO: Could be avoided, since we run the process in one run, instead of adjusting manually the bounding box
        images = glob.glob(dirname + 'output/' +  '*.png')

        # Lower face bounding box
        width = [int(landmarksMinX - ((landmarksMinX * scaleBoundingBox) - landmarksMinX)), int(landmarksMaxX * scaleBoundingBox)]  # [220, 220+360] #crop nichts von rechts oder links (fullsize)
        height = [int(landmarksMinY - ((landmarksMinY * scaleBoundingBox) - landmarksMinY)), int(landmarksMaxY * scaleBoundingBox)]  # crop von 400 bis Bildende

        # Create output dir and if files in it, delete them
        dirnameCropped = dirname + 'cropped'
        if not os.path.exists(dirnameCropped):
            os.mkdir(dirnameCropped)
        else:
            files = glob.glob(dirnameCropped + '/*')
            for f in files:
                os.remove(f)

        # load old landmarks and append new only lower face landmarks
        csv = CSVGenerator()
        data = csv.read_csv(dirname + "landmark.csv")
        # Corrects x coordinates
        data[:, 1::2] = data[:, 1::2].astype(float) - width[0]
        # Corrects y coordinates
        data[:, 2::2] = data[:, 2::2].astype(float) - height[0]
        for i in range(data.shape[0]):
            csv.add_data(data[i, 0], data[i, 1:].reshape((1, 68, 2)))
        csv.export(dirname + "landmark_cropped.csv")

        frame_list = []
        # save crops
        img_ids = [x for x in glob.glob(dirname + "output/*.png")]
        for id in img_ids:
            image = cv2.imread(id, 1)  # gray=0, color=1, color_alpha= -1 #
            image = image[height[0]:height[1], width[0]:width[1]]
            cv2.imwrite(os.path.join(dirnameCropped + "/", os.path.basename(id)), image)  # save cropped image

            i = int(np.argwhere(data[:, 0] == os.path.basename(id)[0:-4]))
            frame = show(image, data[i, 1:].reshape((1, 68, 2)).astype(float))