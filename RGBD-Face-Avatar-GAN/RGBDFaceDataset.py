# Last edit 28.10.2020
import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from tqdm import tqdm
import cv2
import configFile as config
import Utils.FaceAlignmentNetwork as fan
import Utils.DLibFaceTracking as dl
import Utils.HeatmapDrawing as hd
import Utils.CropAndResize as car
import Utils.Visualization as vis
import Utils.IR_EyeTracking as ir


class RGBDFaceDataset(Dataset):

    def __init__(self, path="", imageSize=256):

        self.path_rgb8 = path + "Color/"  # Pfad zu Ordner
        self.path_depth16 = path + "Depth/"
        self.path_ir8 = path + "IR/"
        self.path_rgbd = path + "RGBD/"

        self.rgb8_files = [i for i in os.listdir(self.path_rgb8) if
                           i.endswith('.png') or i.endswith('.jpg')]
        self.depth16_files = [i for i in os.listdir(self.path_depth16) if i.endswith('.png')]
        self.ir8_files = [i for i in os.listdir(self.path_ir8) if i.endswith('.png')]

        self.rgb8_files.sort(key=lambda f: os.path.splitext(f)[0])  # Sort by Namen
        self.depth16_files.sort(key=lambda f: os.path.splitext(f)[0])
        self.ir8_files.sort(key=lambda f: os.path.splitext(f)[0])

        if len(self.rgb8_files) != len(self.depth16_files) and len(self.rgb8_files) != len(self.ir8_files):  # Prüfen ob gleich viele RGB unf Tiefenbilder vorhanden sind
            print("Different number of rgb8- and depth16-files!")
            print("RGB: ", len(self.rgb8_files), " Depth: ", len(self.depth16_files), " IR: ", len(self.ir8_files))
            exit()

        self.imageSize = imageSize
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.RandomAffine((-5, 5), scale=(0.98, 1), shear=None, resample=False, fillcolor=0),
                                                          torchvision.transforms.ToTensor()])

        self.landmarks = np.ndarray([len(self.rgb8_files), 68 + 2, 2])
        self.crop_region = np.zeros(4)

        if os.path.exists(path + 'Landmarks_' + config.FACETRACKING + '_' + str(config.IMAGE_SIZE) + '.txt'):                    # LoadLandmarks
            print('Load Landmarks')
            test = np.loadtxt(path + 'Landmarks_' + config.FACETRACKING  + '_' + str(config.IMAGE_SIZE) + '.txt', dtype=float)
            self.landmarks = test.reshape((-1, 68 + 2, 2))

            self.crop_region = np.loadtxt(path + 'crop_region_' + config.FACETRACKING  + '_' + str(config.IMAGE_SIZE) + '.txt', dtype=float)
        else:
            print("Create Landmarks:")

            eye_tracking_ir = ir.IREyeTraking(config.IRET_Region["x"],
                                              config.IRET_Region["y"],
                                              config.IRET_Region["width"],
                                              config.IRET_Region["height"])

            for idx in tqdm(range(0, len(self.rgb8_files))):
                image = o3d.io.read_image(self.path_rgb8 + self.rgb8_files[idx])  # öffne Bilder
                image = np.asarray(image).astype(float)  # Convert to Numpy Array
                ir_image = cv2.imread(self.path_ir8 + self.ir8_files[idx], cv2.IMREAD_GRAYSCALE)

                if config.FlipYAxis:
                    image = cv2.flip(image, 0)  # Mirror -> Bug Claymore/AzureKinect
                    ir_image = cv2.flip(ir_image, 0)

                if config.FACETRACKING == "fan":
                    landmarks = fan.create2DLandmarks(torch.Tensor(image[:, :, 0:3]), show=False)
                else:
                    landmarks = dl.create2DLandmarks(torch.Tensor(image[:, :, 0:3]), show=False)
                    print(landmarks)

                x1, y1, x2, y2 = eye_tracking_ir(ir_image)
                eyesTensor = torch.Tensor([[x1, y1], [x2, y2]])
                landmarks = torch.cat((landmarks, eyesTensor), 0)

                self.landmarks[idx] = landmarks

            self.crop_region[0] = min(self.landmarks[:, :68, 0].reshape(-1))
            self.crop_region[1] = max(self.landmarks[:, :68, 0].reshape(-1))
            self.crop_region[2] = min(self.landmarks[:, :68, 1].reshape(-1))
            self.crop_region[3] = max(self.landmarks[:, :68, 1].reshape(-1))

            for idx in range(0, self.landmarks.shape[0]):
                self.landmarks[idx] = car.cropAndResizeLandmarksDatasetBased(self.landmarks[idx], self.imageSize, self.crop_region)

            np.savetxt(path + 'Landmarks_' + config.FACETRACKING  + '_' + str(config.IMAGE_SIZE) + '.txt', self.landmarks.reshape(-1))
            print("Landmarks saved as " + path + "Landmarks_" + config.FACETRACKING  + '_' + str(config.IMAGE_SIZE) + ".txt")

            np.savetxt(path + 'crop_region_' + config.FACETRACKING  + '_' + str(config.IMAGE_SIZE) + '.txt', self.crop_region)
            print("Crop-Region saved as " + path + "crop_region_" + config.FACETRACKING  + '_' + str(config.IMAGE_SIZE) + ".txt")

        if not os.path.exists(self.path_rgbd):
            os.mkdir(self.path_rgbd)

        if not os.path.exists(self.path_rgbd + str(config.IMAGE_SIZE)):
            os.mkdir(self.path_rgbd + str(config.IMAGE_SIZE))

            print('Crop images to {}px.'.format(config.IMAGE_SIZE))
            for idx in tqdm(range(0, len(self.rgb8_files))):
                imageRGB8 = o3d.io.read_image(self.path_rgb8 + self.rgb8_files[idx])
                imageDepth16 = o3d.io.read_image(self.path_depth16 + self.depth16_files[idx])

                imageRGB8 = np.asarray(imageRGB8).astype(float)  # Convert to Numpy Array
                imageDepth16 = np.asarray(imageDepth16).astype(float)

                if config.FlipYAxis:
                    imageRGB8 = cv2.flip(imageRGB8, 0)  # Kinect Capture Bug
                    imageDepth16 = cv2.flip(imageDepth16, 0)

                if imageRGB8.shape[0:2] != imageDepth16.shape:  # Prüfen ob RGB und D zusammenpassen
                    print("RGB8 and Depth16 have different sizes.")
                    exit()

                imageRGBD = np.ndarray([imageRGB8.shape[0], imageRGB8.shape[1], 4])
                imageRGBD[:, :, 0:3] = imageRGB8[:, :, 0:3]

                imageDepth16 = imageDepth16 - config.DEPTH_OFFSET
                for x in range(0, imageDepth16.shape[1]):
                    for y in range(0, imageDepth16.shape[0]):
                        temp = imageDepth16[y, x]
                        if temp <= 0:
                            temp = config.DEPTH_MAX
                        if temp > config.DEPTH_MAX:
                            temp = config.DEPTH_MAX
                        imageRGBD[y, x, 3] = temp

                imageRGBD = car.cropAndResizeImageDatasetBased(imageRGBD, self.imageSize, self.crop_region)
                imageRGBD[:, :, 3] = self.apply_depthmask(imageRGBD[:, :, 3])

                imageRGBD = cv2.cvtColor(imageRGBD.astype("uint8"), cv2.COLOR_RGBA2BGRA)
                cv2.imwrite(self.path_rgbd + str(config.IMAGE_SIZE) + '/{:05d}.png'.format(idx), imageRGBD)

    def __len__(self):
        return len(self.rgb8_files)


    def __getitem__(self, idx):
        """imageRGB8 = o3d.io.read_image(self.path_rgb8 + self.rgb8_files[idx])
        imageDepth16 = o3d.io.read_image(self.path_depth16 + self.depth16_files[idx])

        imageRGB8 = np.asarray(imageRGB8).astype(float)  # Convert to Numpy Array
        imageDepth16 = np.asarray(imageDepth16).astype(float)

        if config.FlipYAxis:
            imageRGB8 = cv2.flip(imageRGB8, 0)  # Kinect Capture Bug
            imageDepth16 = cv2.flip(imageDepth16, 0)

        if imageRGB8.shape[0:2] != imageDepth16.shape:  # Prüfen ob RGB und D zusammenpassen
            print("RGB8 and Depth16 have different sizes.")
            exit()

        imageRGBD = np.ndarray([imageRGB8.shape[0], imageRGB8.shape[1], 4])
        imageRGBD[:, :, 0:3] = imageRGB8[:, :, 0:3]

        imageDepth16 = imageDepth16 - config.DEPTH_OFFSET
        for x in range(0, imageDepth16.shape[1]):
            for y in range(0, imageDepth16.shape[0]):
                temp = imageDepth16[y, x]
                if temp <= 0:
                    temp = config.DEPTH_MAX
                if temp > config.DEPTH_MAX:
                    temp = config.DEPTH_MAX
                imageRGBD[y, x, 3] = temp

        imageRGBD = car.cropAndResizeImageDatasetBased(imageRGBD, self.imageSize, self.crop_region)
        imageRGBD[:, :, 3] = self.apply_depthmask(imageRGBD[:, :, 3])"""

        imageRGBD = cv2.imread(self.path_rgbd + str(config.IMAGE_SIZE) + '/{:05d}.png'.format(idx), cv2.IMREAD_UNCHANGED)
        imageRGBD = cv2.cvtColor(imageRGBD, cv2.COLOR_BGRA2RGBA)
        imageRGBD = (imageRGBD.astype("float") - 127.5) / 127.5

        imageRGBD = imageRGBD.transpose(2, 0, 1)
        imageRGBD = torch.Tensor(imageRGBD)

        fourChannelHeatmap_PIL = hd.drawHeatmap(self.landmarks[idx], self.imageSize, returnType="PIL")

        fourChannelHeatmap = self.transforms(fourChannelHeatmap_PIL)
        fourChannelHeatmap = fourChannelHeatmap*2-1

        sample = {'RGBD': imageRGBD, 'Heatmap': fourChannelHeatmap[0].unsqueeze(0)}
        return sample

    def apply_depthmask(self, img):
        ret, mask = cv2.threshold(img, config.DEPTH_MAX-5, 1, cv2.THRESH_BINARY_INV) #10
        # plt.imshow(mask)
        # plt.show()
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        # plt.imshow(img)
        # plt.show()
        img = img * mask + (1 - mask) * config.DEPTH_MAX # img = img * mask
        # plt.imshow(img)
        # plt.show()
        # print("applied mask", np.min(img[np.nonzero(img)]), img.max())
        return img


if __name__ == '__main__':

    rgbdFaceDataset = RGBDFaceDataset(imageSize=config.IMAGE_SIZE, path="Data/" + config.DatasetName + "/")

    if not os.path.exists("Data/" + config.DatasetName + "/Visualization/"):
        os.mkdir("Data/" + config.DatasetName + "/Visualization/")

    print("Visualization:")
    for i in tqdm(range(0, len(rgbdFaceDataset))):
        sample = rgbdFaceDataset[i]
        vis.exportExample(sample['RGBD'], sample['Heatmap'], "Data/" + config.DatasetName + "/Visualization/" + str(i) + ".png")

    vis.showPointCloud(sample["RGBD"])
