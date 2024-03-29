# Last edit 06.07.2020
import torch
import torch.nn as nn
import Pix2PixGAN.Generator as pix2pixG
import Pix2PixGAN.Discriminator as pix2pixD
import Pix2PixGAN.Initialization as pix2pixInit
import Pix2PixGAN.GANLoss as pix2pixLoss
import functools
import RGBDFaceDataset as rgbdDataset
#import Data.MelSpecDataset as rgbdDataset
import Utils.Visualization as Vis
from tqdm import tqdm
from os import path
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import onnx
import os
import configFile as config
import statistics
import math

if __name__ == '__main__':

    if not os.path.exists("Data/" + config.DatasetName + "/Result/"):
        os.mkdir("Data/" + config.DatasetName + "/Result/")

    if not os.path.exists("Data/" + config.DatasetName + "/Result/" + str(config.IMAGE_SIZE)):
        os.mkdir("Data/" + config.DatasetName + "/Result/" + str(config.IMAGE_SIZE))
        os.mkdir("Data/" + config.DatasetName + "/Result/" + str(config.IMAGE_SIZE) + "/Snaps/")
        os.mkdir("Data/" + config.DatasetName + "/Result/" + str(config.IMAGE_SIZE) + "/Examples/")
        os.mkdir("Data/" + config.DatasetName + "/Result/" + str(config.IMAGE_SIZE) + "/Weights/")

    ### Define Networks ###

    netG = pix2pixG.UnetGenerator(input_nc= config.INPUT_CHANNEL, output_nc=4, num_downs=int(math.log2(config.IMAGE_SIZE)), ngf=64, norm_layer=functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True), use_dropout=True)
    netG = pix2pixInit.init_net(netG, gpu_ids=[0])
    summary(netG, (config.INPUT_CHANNEL, config.IMAGE_SIZE, config.IMAGE_SIZE))


    netD = pix2pixD.NLayerDiscriminator(input_nc=config.INPUT_CHANNEL + 4, ndf=64, n_layers=int(math.log2(config.IMAGE_SIZE) - 5), norm_layer=functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True))
    netD = pix2pixInit.init_net(netD, gpu_ids=[0])
    summary(netD, (config.INPUT_CHANNEL + 4, config.IMAGE_SIZE, config.IMAGE_SIZE))

    ### Load Exsting Model State ###

    if path.exists("Data/" + config.DatasetName + "/Result/" + str(config.IMAGE_SIZE) + "/trainedGenerator.pth") and \
            path.exists("Data/" + config.DatasetName + "/Result/" + str(config.IMAGE_SIZE) + "/trainedDiscriminator.pth"):

        netG.load_state_dict(torch.load("Data/" + config.DatasetName + "/Result/" + str(config.IMAGE_SIZE) + "/trainedGenerator.pth"))
        netD.load_state_dict(torch.load("Data/" + config.DatasetName + "/Result/" + str(config.IMAGE_SIZE) + "/trainedDiscriminator.pth"))

        startEpoch = int(input('Enter startEpoch:'))
        print('startEpoch:', startEpoch, type(startEpoch))

    else:
        startEpoch = 1

    ### Tensorboard
    comment = f'_dataset={config.DatasetName}'
    print("TensorBoard:" + comment)
    tb = SummaryWriter(comment=comment)

    ### Training settings ###

    learningRate = 0.0002
    lambda_L1 = 100

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterionGAN = pix2pixLoss.GANLoss("vanilla").to(device)
    criterionL1 = torch.nn.L1Loss()

    optimizer_G = torch.optim.Adam(netG.parameters(), lr=learningRate, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=learningRate, betas=(0.5, 0.999))

    lambda_rule = lambda epoch: 1.0 - max(0, epoch + startEpoch - 100) / float(100 + 1)
    scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_rule)
    scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lambda_rule)

    ### Training ###

    dataset = rgbdDataset.RGBDFaceDataset(imageSize=config.IMAGE_SIZE, path="Data/" + config.DatasetName + "/")
    dataset = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    for epoch in range(startEpoch, 200 + 1):
        print("Epoche: ", epoch, "(LearningRates:", scheduler_G.get_last_lr(), scheduler_D.get_last_lr(), ")")
        epoch_loss_D = []
        epoch_loss_G = []
        for i, data in enumerate(tqdm(dataset)):

            heatmap = data['Heatmap'].to(device)
            realRGBD = data['RGBD'].to(device)
            #Vis.showDatapair(realRGBD[0], heatmap[0])
            fakeRGBD = netG(heatmap)
            #Vis.showDatapair(fakeRGBD[0], heatmap[0])


            ### Update Discriminator ###  similar to https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/pix2pix_model.py

            optimizer_D.zero_grad()  # set D's gradients to zero
            """Calculate GAN loss for the discriminator"""
            # Fake; stop backprop to the generator by detaching fake_B
            fake_AB = torch.cat((heatmap, fakeRGBD),1)
            pred_fake = netD(fake_AB.detach())
            loss_D_fake = criterionGAN(pred_fake, False)
            # Real
            real_AB = torch.cat((heatmap, realRGBD), 1)
            pred_real = netD(real_AB)
            loss_D_real = criterionGAN(pred_real, True)
            # combine loss and calculate gradients
            loss_D = (loss_D_fake + loss_D_real) * 0.5
            loss_D.backward()
            optimizer_D.step()  # update D's weights
            ### Update Generator ###  similar to https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/pix2pix_model.py

            optimizer_G.zero_grad()  # set G's gradients to zero
            """Calculate GAN and L1 loss for the generator"""
            # First, G(A) should fake the discriminator
            fake_AB = torch.cat((heatmap, fakeRGBD), 1)
            #with torch.no_grad():     # %TODO Bug: exclude D for loss computation
            pred_fake = netD(fake_AB)
            loss_G_GAN = criterionGAN(pred_fake, True)
            # Second, G(A) = B
            loss_G_L1 = criterionL1(fakeRGBD, realRGBD) * lambda_L1
            # combine loss and calculate gradients
            loss_G = loss_G_GAN + loss_G_L1
            loss_G.backward()
            optimizer_G.step()  # udpate G's weights

            epoch_loss_D.append(loss_D.item())
            epoch_loss_G.append(loss_G.item())
        tb.add_scalars("Loss", {'Discriminator': statistics.mean(epoch_loss_D),
                                'Generator': statistics.mean(epoch_loss_G)}, epoch)

        ### Learning Rate Schedular

        scheduler_G.step()
        scheduler_D.step()

        ### Save Modell ###

        torch.save(netG.cpu().state_dict(), "Data/" + config.DatasetName + "/Result/" + str(config.IMAGE_SIZE) + "/trainedGenerator.pth")
        torch.save(netD.cpu().state_dict(), "Data/" + config.DatasetName + "/Result/" + str(config.IMAGE_SIZE) + "/trainedDiscriminator.pth")

        if epoch % 10 == 0:
            torch.save(netG.cpu().state_dict(), "Data/" + config.DatasetName + "/Result/" + str(config.IMAGE_SIZE) + "/Weights/trainedGenerator_epoch_" + str(epoch) + ".pth")
            torch.save(netD.cpu().state_dict(), "Data/" + config.DatasetName + "/Result/" + str(config.IMAGE_SIZE) + "/Weights/trainedDiscriminator_epoch_" + str(epoch) + ".pth")

        ### Trace Modell ###

        noise = torch.randn(heatmap.shape)
        traced = torch.jit.trace(netG.eval(), (noise,))
        netG.train().to(device)
        netD.train().to(device)
        traced.save("Data/" + config.DatasetName + "/Result/" + str(config.IMAGE_SIZE) + "/tracedGenerator.zip")
        #print("LoadModel")
        #loaded = torch.jit.load('trainedGenerator.zip')
        #print(loaded)
        #print(loaded.code)
        #temp = loaded.forward(heatmap)
        #Vis.showDatapair(temp[0],heatmap[0])

        ### Export Sample Image ###
        if fakeRGBD[0].shape[0] == 4 and heatmap[0].shape[0] == 1:
            Vis.exportExample(fakeRGBD[0], heatmap[0], "Data/" + config.DatasetName + "/Result/" + str(config.IMAGE_SIZE) + "/Examples/example_" + str(epoch) +".png")

    ### ONNX ####
    x = torch.randn(1, config.INPUT_CHANNEL, config.IMAGE_SIZE, config.IMAGE_SIZE, requires_grad=True)
    torch.onnx.export(netG.to("cpu"),  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      "Data/" + config.DatasetName + "/Result/" + str(config.IMAGE_SIZE) + "/tracedGenerator.onnx",  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}})
    onnx_model = onnx.load("Data/" + config.DatasetName + "/Result/" + str(config.IMAGE_SIZE) + "/tracedGenerator.onnx")
    print(onnx.checker.check_model(onnx_model))

