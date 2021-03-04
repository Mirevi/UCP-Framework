# last edit 16.06.20 Alpe6825
import math
import numpy as np

def NME(landmarks_groundTruth, landmarks_predicted, percentage = False, printNME=True):

    if len(landmarks_groundTruth) != len(landmarks_predicted):
        print("Unequal number of points in A and B set.")
        print("groundTruth:",len(landmarks_groundTruth))
        print("predicted:",len(landmarks_predicted))
        exit()

    #Boundingbox-Part
    x_min = np.min(landmarks_groundTruth[:, 0])
    x_max = np.max(landmarks_groundTruth[:, 0])
    y_min = np.min(landmarks_groundTruth[:, 1])
    y_max = np.max(landmarks_groundTruth[:, 1])

    d = math.sqrt((x_max-x_min)*(y_max-y_min))

    #Normelized Mean Error (vgl. [bulat2017far] 3.3 Metric)
    sum = 0
    for i in range(len(landmarks_groundTruth)-1):
        sum += math.sqrt(
            math.pow(landmarks_groundTruth[i][0] - landmarks_predicted[i][0], 2) +
            math.pow(landmarks_groundTruth[i][1] - landmarks_predicted[i][1], 2)
        )/d
    nme = sum / len(landmarks_groundTruth)

    if percentage == True:
        nme *= 100

    if printNME == True:
        print("NME:", nme, "%" if percentage else "")

    return nme

"""
@inproceedings{bulat2017far,
  title={How far are we from solving the 2D \& 3D Face Alignment problem? (and a dataset of 230,000 3D facial landmarks)},
  author={Bulat, Adrian and Tzimiropoulos, Georgios},
  booktitle={International Conference on Computer Vision},
  year={2017}
}
"""

if __name__ == '__main__':

    pred = np.array([ 15.8149,  70.8383,  23.1018, 112.0394,  45.0785, 147.2564,  67.1905,
        165.7947,  92.8273, 179.9299, 149.6204, 193.3956, 205.4972, 184.4597,
        246.8566, 171.6060, 280.9257, 154.2388, 311.1960, 124.3119, 330.6689,
         78.5818, 121.2333,  21.6538, 133.0869,  22.0615, 148.9704,  26.4093,
        165.5083,  22.6743, 183.2996,  21.3034,  94.0876,  89.2835, 110.9580,
         70.6713, 135.0317,  57.7471, 148.7875,  62.1021, 161.2734,  58.9836,
        191.2197,  71.2922, 212.0242,  89.6749, 190.8128, 104.2638, 168.5078,
        110.1904, 150.0220, 110.7261, 128.1230, 109.7175, 111.7346, 101.4493,
         98.7641,  87.8230, 131.7474,  76.6830, 148.5881,  75.3634, 167.6442,
         78.0436, 207.7502,  88.8824, 168.5469,  88.5169, 148.8464,  90.9131,
        131.9855,  87.0774])

    real = np.array([ 17.,  71.,  28., 117.,  45., 145.,  68., 162.,  97., 174., 148., 185.,
        205., 179., 244., 174., 279., 157., 307., 123., 330.,  77., 125.,  20.,
        136.,  20., 148.,  26., 165.,  20., 182.,  20.,  97.,  88., 114.,  71.,
        136.,  60., 148.,  66., 159.,  66., 188.,  77., 210.,  94., 188., 100.,
        165., 105., 148., 105., 131., 100., 114.,  94., 102.,  88., 131.,  83.,
        148.,  83., 165.,  83., 210.,  88., 165.,  83., 148.,  83., 136.,  83.])

    NME(real.reshape(-1,2), pred.reshape(-1,2))