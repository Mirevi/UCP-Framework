import cv2
import face_alignment
import os
import uuid
import glob
import Utils.Camera as cam
import numpy as np
import pandas as pd
import sys

# PATH2VID = "data/Eva-07-03-21/_eva_shortened_to_10sec.mp4"
PATH2VID = "data/Mirco-16-10-21/WIN_20211016_20_31_04_Pro.mp4"
PATH2VIDS = "data/multiVids" # without slash or asterisk
USE_MULTIPLE_VIDS = True # when True PATH2VIDS will be used, otherwise PATH2VID (without S at the end)

ROTATE = True
SHOWFRAMES = True
DEBUG_SAVE_IMAGES_WITH_LM = True # show images must be turned ON for that option
scaleBoundingBox = 1.1 # e.g.: the cropping bounding box is 10% bigger (with a value of 1.1) than the bounding box that spans the min and max values of the landmarks
boundingBoxPadding = 10
targetCroppedImageSizeWidth = 340
targetCroppedImageSizeHeight = 260


#subtract videoname from path and set it as dirname for further processing
dirname = ""
if(USE_MULTIPLE_VIDS):
    dirname = PATH2VIDS
    dirname = dirname + "/"
else:
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
    print("DataTool.py started")
    # Select a camera type. Caddx EOS 2 is "Weitwinkel"
    cam = cam.Camera("Weitwinkel")

    dirnameOutput = dirname + "output"
    dirnameOutputWithLM = dirname + "outputWithLM"

    # Create output dir for cropped images (based on bounding box) and if files in it, delete them
    dirnameCropped = dirname + 'cropped'
    if not os.path.exists(dirnameCropped):
        os.mkdir(dirnameCropped)
    else:
        files = glob.glob(dirnameCropped + '/*')
        for f in files:
            os.remove(f)

    csv = CSVGenerator()
    fa_v = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda:0', flip_input=True)

    print("Start facial landmark detection")
    print("Processing Mode: Using multiple videos for training")

    allFacialLandmarksAsList = []
    data = []

    # Create output dir and if files in it, delete them
    if not os.path.exists(dirnameOutput):
        os.mkdir(dirnameOutput)
    else:
        files = glob.glob(dirnameOutput + '/*')
        for f in files:
            os.remove(f)

    if not os.path.exists(dirnameOutputWithLM):
        os.mkdir(dirnameOutputWithLM)
    else:
        files = glob.glob(dirnameOutputWithLM + '/*')
        for f in files:
            os.remove(f)

    csv = CSVGenerator()

    vid_filenames = [x for x in glob.glob(PATH2VIDS + "/*.mp4")]
    for filename in vid_filenames:
        print("Processing " + filename)
        data = []
        allFacialLandmarksAsList = []
        allFacialLandmarksAsNpArray = np.array()
        vidcap = cv2.VideoCapture(filename, 0)
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

                #if no landsmarks were found skip this loop
                if preds is None:
                    continue

                sizeOfPreds = len(preds)
                if sizeOfPreds != 1:
                    print("There were found more or less faces found in images. The number of faces is: ")
                    print(sizeOfPreds)
                    continue
                if preds == None:
                    print("No faces found in this frame")  # stops this loop iteration with continue, because there are no faces found
                    continue

                allFacialLandmarksAsList.append(preds)

                id = uuid.uuid4()
                cv2.imwrite(os.path.join(dirnameOutput + "/", str(id) + ".png"), image)  # save frame with unique id as png image
                csv.add_data(id, preds)

                row = np.concatenate((np.array([str(id)]), np.array(preds[0]).reshape(-1)), axis=0)
                data.append(row)

                if SHOWFRAMES:
                    frame = show(image, preds)
                    if DEBUG_SAVE_IMAGES_WITH_LM:
                        cv2.imwrite(os.path.join(dirnameOutputWithLM + "/", str(id) + ".png"), frame)  # save cropped image

            else:
                break

        vidcap.release()

        print("Start automatic bounding box calculation")
        # Calc mouth bounding box
        # convert in np.array
        allFacialLandmarksAsNpArray = np.array(allFacialLandmarksAsList)
        # reduce dimensionality from w, x, y, z to w, y, z (FAN creates a weired array)
        allFacialLandmarksAsNpArray = allFacialLandmarksAsNpArray[:, -1]
        # take only the relevant lower face landmarks
        onlyLowerFaceLandmarks = np.concatenate([allFacialLandmarksAsNpArray[:, 3:13], allFacialLandmarksAsNpArray[:, 31:35]], 1)
        onlyLowerFaceLandmarks = np.concatenate([onlyLowerFaceLandmarks, allFacialLandmarksAsNpArray[:, 48:67]], 1)
        # Min and max for bounding box
        landmarksMinX = onlyLowerFaceLandmarks[:, :, 0:1].min()
        landmarksMaxX = onlyLowerFaceLandmarks[:, :, 0:1].max()
        landmarksMinY = onlyLowerFaceLandmarks[:, :, 1:].min()
        landmarksMaxY = onlyLowerFaceLandmarks[:, :, 1:].max()

        images = glob.glob(dirname + 'output/' + '*.png')

        # Lower face bb (bounding box)
        minMaxX = [int(landmarksMinX - ((landmarksMinX * scaleBoundingBox) - landmarksMinX)), int(landmarksMaxX * scaleBoundingBox)]
        minMaxY = [int(landmarksMinY - ((landmarksMinY * scaleBoundingBox) - landmarksMinY)), int(landmarksMaxY * scaleBoundingBox)]

        bbWidth = minMaxX[1] - minMaxX[0]
        bbHeight = minMaxY[1] - minMaxY[0]

        # 340 ist "Soll"
        bbWidthDiffToCorrectRatio = targetCroppedImageSizeWidth - bbWidth
        bbHeightDiffToCorrectRatio = targetCroppedImageSizeHeight - bbHeight

        if bbWidthDiffToCorrectRatio % 2 == 1:
                bbWidthDiffToCorrectRatio += 1

        if bbHeightDiffToCorrectRatio % 2 == 1:
            bbHeightDiffToCorrectRatio += 1

        if minMaxX[0] % 2 == 1:
                minMaxX[0] -= 1

        if minMaxX[1] % 2 == 1:
            minMaxX[1] -= 1

        if minMaxY[0] % 2 == 1:
                minMaxY[0] -= 1

        if minMaxY[1] % 2 == 1:
            minMaxY[1] -= 1

        minMaxX[0] -= int(bbWidthDiffToCorrectRatio / 2)
        minMaxY[0] -= int(bbHeightDiffToCorrectRatio / 2)
        minMaxX[1] += int(bbWidthDiffToCorrectRatio / 2)
        minMaxY[1] += int(bbHeightDiffToCorrectRatio / 2)

        data = np.array(data)
        # print("Here comes the data before: ")
        # print(data)
        # Corrects x coordinates based on the calculated bounding box above
        data[:, 1::2] = data[:, 1::2].astype(float) - minMaxX[0] # x value
        # Corrects y coordinates based on the calculated bounding box above
        data[:, 2::2] = data[:, 2::2].astype(float) - minMaxY[0] # y value
        for i in range(data.shape[0]):
            csv.add_data(data[i, 0], data[i, 1:].reshape((1, 68, 2)))

        i = 0
        for id in data[:, 0]:
            filePathOutput = dirnameOutput + "/" + id +  ".png"
            image = cv2.imread(filePathOutput, 1)  # gray=0, color=1, color_alpha= -1 #

            # Crop image based on the calculated bounding box above
            image = image[minMaxY[0]:minMaxY[1], minMaxX[0]:minMaxX[1]]

            # Save cropped image
            filePathCropped = dirnameCropped + "/" + id +  ".png"
            cv2.imwrite(filePathCropped, image)  # save cropped image

            # Show image with landmarks
            frame = show(image, data[i, 1:].reshape((1, 68, 2)).astype(float))
            i += 1
    csv.export(dirname + "landmark_cropped.csv")
    print("Successful processing. Have a look at the results in the directory " + PATH2VIDS)