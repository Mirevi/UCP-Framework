import os
import torch
import Dataset as ds
from tqdm import tqdm
import Utils.Visualization as vis
import torchsummary
from torch.utils.tensorboard import SummaryWriter
from NormalizedMeanError import NME
import numpy as np
import matplotlib.pyplot as plt
import statistics
from CNNModel import Net
import cv2

epochs = 15
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

bs_list = [8]
lr_list = [0.001]
drop_list = [0]

SHOW_FRAMES = True
WRITE_IMAGES_SORTED_BY_NME = False # if true SHOW_FRAMES must also be true

def show_and_save_with_NME(nmw, img, truth_landmarks, pred_landmarks):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    for i in truth_landmarks:
        cv2.circle(img, (int(i[0] * 340 / 1.5 * 0.9), int(i[1] * 260 / 1.5 * 0.9)), 3, (0, 255, 0))
    for i in pred_landmarks:
        cv2.circle(img, (int(i[0] * 340 / 1.5 * 0.9), int(i[1] * 260 / 1.5 * 0.9)), 3, (0, 0, 255))

    if WRITE_IMAGES_SORTED_BY_NME:
        cv2.imwrite("NME-Images-Sorted-By-NME/" + str(nmw) + ".jpg", img * 255)
    cv2.imshow('frame', img)
    cv2.waitKey(1)
    return img


if __name__ == "__main__":
    # Dataset
    completeDataset = ds.FaceLandmarkDataset("C:/devel/UCP-Framework/Lower-Face-CNN/data/multiVids/")
    #train_ds = completeDataset
    #val_ds = ds.FaceLandmarkDataset("../Dataset/Philipp3_1/")

    # Create output dir for the trained models, if it doesn't exist yet
    dirnameWeights = "Weights"
    if not os.path.exists(dirnameWeights):
        os.mkdir(dirnameWeights)

    len_complete = len(completeDataset)
    len_train = int(0.7 * len_complete)
    len_val = len_complete - len_train
    train_ds, val_ds = torch.utils.data.random_split(completeDataset, [len_train, len_val])
    print("train dataset length:", len(train_ds))
    print("validation dataset lengh:", len(val_ds))

    for batchsize in bs_list:
        trainLoader = torch.utils.data.DataLoader(train_ds, batchsize, shuffle=True, num_workers=4)
        valLoader = torch.utils.data.DataLoader(val_ds, 1, shuffle=True, num_workers=4)

        for learningrate in lr_list:
            for dropoutrate in drop_list:

                # Model
                sample = completeDataset[0]
                model = Net(sample["image"].unsqueeze(0), dropoutrate).to(device)
                model = model.to(device)
                # torchsummary.summary(model, exampleimage.shape)
                lossFunction = torch.nn.MSELoss() #torch.nn.L1Loss()
                optimizer = torch.optim.Adam(model.parameters(), lr=learningrate)
                lr_scheduler = None
                #torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer) #None#torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

                comment = f'_bs={batchsize}_lr={learningrate}_dr={dropoutrate}_BESSERERNAME'
                print(comment)
                tb = SummaryWriter(comment=comment)
                special_name = "_bs=" + str(batchsize) + "_lr=" + str(learningrate) + "_dr=" + str(dropoutrate) + "_BESSERERNAME"

                loss_hist = []
                loss_hist_eval = []
                best_loss = float('inf')

                nme_hist = []
                nme_hist_eval = []
                best_nme = float('inf')

                def loop(model, dataloader, lossFunction, optimizer = None, lr_scheduler = None ):
                    running_loss = []
                    running_nme = []

                    if optimizer == None:
                        model.eval()
                    else:
                        model.train()

                    for batch in dataloader:
                        # print("type(batch)")
                        # print(type(batch))
                        pred_landmarks = model(batch["image"].to(device))
                        loss = lossFunction(pred_landmarks, batch["landmarks"].reshape(batch["landmarks"].shape[0], -1).to(device))

                        if optimizer is not None:
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                        running_loss.append(loss.item())
                        for i in range(batch["landmarks"].shape[0]):
                            truth = batch["landmarks"][i].detach().cpu().numpy().reshape(-1, 2)
                            # print("truth.shape")
                            # print(truth.shape)
                            pred = pred_landmarks[i].detach().cpu().numpy().reshape(-1, 2)
                            running_nme.append(NME(truth, pred, printNME=False))

                            # print("pred.shape")
                            # print(pred.shape)
                            nme = NME(truth, pred, printNME=False)
                            running_nme.append(nme)
                            # print("type(batch[image][i])")
                            # print(type(batch["image"][i].cpu()))

                            # print(batch["image"].shape)
                            if SHOW_FRAMES:
                                show_and_save_with_NME(nme, batch["image"][i].cpu().numpy().reshape(155, 203, 1), truth, pred)

                    if lr_scheduler is not None:
                        lr_scheduler.step()

                    return model, statistics.mean(running_loss), np.mean(running_nme)

                # Train
                for e in tqdm(range(epochs)):

                    model, loss_train, nme_train = loop(model, trainLoader, lossFunction, optimizer, lr_scheduler)
                    loss_hist.append(loss_train)
                    nme_hist.append(nme_train)

                    model, loss_eval, nme_eval = loop(model, valLoader, lossFunction)
                    loss_hist_eval.append(loss_eval)
                    nme_hist_eval.append(nme_eval)

                    tb.add_scalars("Loss", {'Train': loss_train, 'Val': loss_eval}, e)
                    tb.add_scalars("NME", {'Train': nme_train, 'Val': nme_eval}, e)
                    tb.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], e)

                    if loss_eval < best_loss:
                        best_loss = loss_eval
                        torch.save(model.state_dict(), "Weights/best_model_" + special_name + ".pth")
                        print("Saved best model.")

                    torch.save(model.state_dict(), "Weights/" + str(e) +"_model_" + special_name + ".pth")

                tb.close()

