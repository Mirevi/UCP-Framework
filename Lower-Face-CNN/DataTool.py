import cv2
import face_alignment
import os
import uuid
import glob
import Utils.Camera as cam
import numpy as np
import pandas as pd

PATH2VID = "Philipp-12-03-21-full-26sec.mp4"
ROTATE = True

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

    print("Analyse (1) or Crop (2):")
    action = int(input('Enter action:'))

    if action == 1:
        cam = cam.Camera("Weitwinkel")

        # Outputfolder erstellen
        dirname = 'output'
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        else:
            files = glob.glob(dirname + '/*')
            for f in files:
                os.remove(f)

        csv = CSVGenerator()
        fa_v = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda:0', flip_input=True)
        vidcap = cv2.VideoCapture(PATH2VID, 0)
        frame_list = []
        while (vidcap.isOpened()):
            hasFrames, image = vidcap.read()
            if hasFrames:  # Auf jeden Frame FAN anwenden
                image = cam(image)
                if ROTATE:
                    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                preds = fa_v.get_landmarks(image)
                if preds == None:
                    continue

                id = uuid.uuid4()
                cv2.imwrite(os.path.join(dirname, str(id) + ".png"), image)  # Frame speichern
                csv.add_data(id, preds)
                frame = show(image, preds)
            else:
                break

        vidcap.release()
        csv.export("landmark.csv")

    elif action == 2:

        # lies pngs und speicher in Array
        images = glob.glob("output/*.png")

        # Variablen Höhe / Weite / Y-Achsen Anfang
        width = [65, 65 + 300]  # [220, 220+360] #crop nichts von rechts oder links (fullsize)
        height = [340, 340 + 220]  # crop von 400 bis Bildende

        # Outputordner erstellen
        dirname = 'cropped'
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        else:
            files = glob.glob(dirname + '/*')
            for f in files:
                os.remove(f)

        csv = CSVGenerator()
        data = csv.read_csv("landmark.csv")
        # X-Korrigieren
        data[:, 1::2] = data[:, 1::2].astype(float) - width[0]
        # Y-Korrogieren
        data[:, 2::2] = data[:, 2::2].astype(float) - height[0]

        for i in range(data.shape[0]):
            csv.add_data(data[i, 0], data[i, 1:].reshape((1, 68, 2)))
        csv.export("landmark_cropped.csv")

        frame_list = []
        # crop speichern & neue landmarks in crop.cvs speichern
        img_ids = [x for x in glob.glob("output/*.png")]
        for id in img_ids:
            image = cv2.imread(id, 1)  # gray=0, color=1, color_alpha= -1 #
            image = image[height[0]:height[1], width[0]:width[1]]
            cv2.imwrite(os.path.join(dirname, os.path.basename(id)), image)  # in crop speichern

            i = int(np.argwhere(data[:, 0] == os.path.basename(id)[0:-4]))
            frame = show(image, data[i, 1:].reshape((1, 68, 2)).astype(float))