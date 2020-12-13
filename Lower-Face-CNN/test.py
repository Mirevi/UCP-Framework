import torch
import cv2
from CNN_PyTorch.models.example2 import Net
from CNN_PyTorch.models.VGG16 import create_vgg16
import CNN_PyTorch.Dataset as ds
import numpy as np
import glob
import time
from CNN_PyTorch.colorprocessing import color_prosessing
from CNN_PyTorch.models.ResNet import ResNet
import torchsummary


if __name__ == "__main__":

    DatasetName = "Jannik" #

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    completeDataset = ds.FaceLandmarkDataset("../Dataset/" + DatasetName)
    sample = completeDataset[0]

    model = Net(sample["image"].unsqueeze(0), 0.1)#create_vgg16()#
    model.load_state_dict(torch.load("Weights/14_model__bs=8_lr=0.0001_dr=0.2_Jannik.pth"))#("Weights/best_model__bs=8_lr=0.0001_dr=0.2_Weitwinkel3_OhneLampe3.pth"))
    model.to(device)
    model.eval()
    #torchsummary.summary(model, sample["image"].shape)
    #exit()

    """camID = 0
    cap = cv2.VideoCapture(camID)  # 0 for webcam or path to video
    while (cap.isOpened() == False):
        print("Error opening video stream or file")
        camID += 1
        cap = cv2.VideoCapture(camID)  # 0 for webcam or path to video
    print("Camera ID:", camID)
    """
    files = glob.glob("../Dataset/" + DatasetName + "/cropped/*.png")

    q = 0
    while (True):
        if q == len(files):
            break
        frame = cv2.imread(files[q])
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.equalizeHist(frame, frame)
        #frame = color_prosessing(frame)


        frame = cv2.resize(frame, (completeDataset.w, completeDataset.h))
        #cv2.flip(frame, 1, frame)
        #cv2.normalize(frame, frame, 255, 0,cv2.NORM_MINMAX);
        time.sleep(0.5)
        q += 1

        #_, frame = cap.read()
        #frame = cv2.flip(frame,1)
        h, w = frame.shape

        offset_X = int((w - completeDataset.w) / 2)
        offset_Y = int((h - completeDataset.h) / 2)
        frame = frame[offset_Y:offset_Y+completeDataset.h, offset_X:offset_X+completeDataset.w]

        input = frame#.transpose(2, 0, 1)
        input = torch.tensor(input/255, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        start = time.time()
        output = model(input.to(device))
        #print(output)
        # print(time.time()-start)

        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        x_prev = int(output[0][0] * completeDataset.w)
        y_prev = int(output[0][0 + 1] * completeDataset.h)
        for i in range(2, len(output[0]), 2):
            x = int(output[0][i]*completeDataset.w)
            y = int(output[0][i+1]*completeDataset.h)
            #cv2.circle(frame, (x, y), 1, (255, 0, 0))
            if i != 22 and i != 32:
                cv2.line(frame, (x_prev, y_prev), (x, y), (255, 0, 0))
            x_prev = x
            y_prev = y

        output = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        #output = cv2.resize(output,(256, 160))
        cv2.imshow('Window', output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #cap.release()
    cv2.destroyAllWindows()
