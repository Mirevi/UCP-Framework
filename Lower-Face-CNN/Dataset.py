import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd
import Utils.Visualization as vis
from Utils.CropLandmarks import CropLandmarks
import Utils.transformations as t
import numpy as np
from skimage import io, transform
import cv2

class FaceLandmarkDataset(torch.utils.data.Dataset):
    def __init__(self, path2data):

        self.path2data = path2data
        path2csv = os.path.join(path2data, "landmark_cropped.csv")
        print("CSV:", path2csv)
        dataframe = pd.read_csv(path2csv)
        self.data = dataframe.to_numpy()

        height = int(260/1.5)#206#192#/2  #188/300*256#96 #225
        width = int(340/1.5)#270#256#/2   #int(360/225*96)

        self.transform = transforms.Compose([t.Rescale((height, width)), #225 360
                                             #t.RandomFlip(),
                                             t.RandomCrop((int(height*0.90), int(width*0.90))),
                                             t.ToTensor(),
                                            ])

        self.h = int(height * 0.90)
        self.w = int(width * 0.90)

        print(self.__len__())

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):

        flip = False
        if idx >= self.__len__()/2:
            idx -= self.__len__()
            flip = True

        img_path = os.path.join(self.path2data, "cropped/" + self.data[idx, 1] + ".png")
        image = io.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        cv2.equalizeHist(image, image)

        h, w = image.shape
        image = image.reshape(h, w, 1)

        landmarks = self.data[idx, 2:138].astype(float)
        crop_landmarks = CropLandmarks(landmarks)
        landmarks = crop_landmarks.crop_landmarks_face_bottom_half()
        landmarks = landmarks.reshape(-1, 2)

        sample = {'image': image, 'landmarks': landmarks}
        #if flip == True:
        #    sample = self.transformFlip(sample)
        sample = self.transform(sample)

        sample["landmarks"] /= torch.Tensor([self.w, self.h])

        return sample


if __name__ == "__main__":

    completeDataset = FaceLandmarkDataset("../Dataset/Jannik")
    sample = completeDataset[0]

    vis.show(sample["image"], sample["landmarks"].reshape(-1), completeDataset.h, completeDataset.w)
    print(sample["landmarks"].reshape(-1))
