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

epochs = 15
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

bs_list = [8]
lr_list = [0.001]
drop_list = [0]

if __name__ == "__main__":

    # Dataset
    completeDataset = ds.FaceLandmarkDataset("")
    #train_ds = completeDataset
    #val_ds = ds.FaceLandmarkDataset("../Dataset/Philipp3_1/")

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

                comment = f'_bs={batchsize}_lr={learningrate}_dr={dropoutrate}_Jannik'
                print(comment)
                tb = SummaryWriter(comment=comment)
                special_name = "_bs=" + str(batchsize) + "_lr=" + str(learningrate) + "_dr=" + str(dropoutrate) + "_Jannik"

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

                        pred_landmarks = model(batch["image"].to(device))
                        loss = lossFunction(pred_landmarks, batch["landmarks"].reshape(batch["landmarks"].shape[0], -1).to(device))

                        if optimizer is not None:
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                        running_loss.append(loss.item())
                        for i in range(batch["landmarks"].shape[0]):
                            truth = batch["landmarks"][i].detach().cpu().numpy().reshape(-1, 2)
                            pred = pred_landmarks[i].detach().cpu().numpy().reshape(-1, 2)
                            running_nme.append(NME(truth, pred, printNME=True))

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

