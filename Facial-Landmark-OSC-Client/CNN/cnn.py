import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I
#import torchsummary
#from CNN_PyTorch.models.SRINC import SRINC

class Net(nn.Module):

    def __init__(self, exampleimage, dropoutrate):

        super(Net, self).__init__()
        #self.SRINC = SRINC(aux_logits=True, num_classes=72)
        self.conv0 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5)
        self.conv1 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 5)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3)
        self.conv4 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=2)

        self.batchnorm0 = nn.BatchNorm2d(16)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.batchnorm4 = nn.BatchNorm2d(256)
        self.batchnorm5 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        if dropoutrate == -1:
            self.drop0 = nn.Dropout(p=0.05 / 2)
            self.drop1 = nn.Dropout(p=0.1 / 2)
            self.drop2 = nn.Dropout(p=0.2 / 2)
            self.drop3 = nn.Dropout(p=0.3 / 2)
            self.drop4 = nn.Dropout(p=0.4 / 2)
            self.drop5 = nn.Dropout(p=0.5 / 2)
            self.drop6 = nn.Dropout(p=0.6 / 2)
        else:
            dropoutrate2 = dropoutrate
            self.drop0 = nn.Dropout(p=dropoutrate2)
            self.drop1 = nn.Dropout(p=dropoutrate2)
            self.drop2 = nn.Dropout(p=dropoutrate2)
            self.drop3 = nn.Dropout(p=dropoutrate2)
            self.drop4 = nn.Dropout(p=dropoutrate2)
            self.drop5 = nn.Dropout(p=dropoutrate)
            self.drop6 = nn.Dropout(p=dropoutrate)

        self.forward(exampleimage, init=True)

        self.fc1 = nn.Linear(in_features = self.flat, out_features = 500, bias= False) # The number of input gained by "print("Flatten size: ", x.shape)" in below
        self.fc2 = nn.Linear(in_features = 500, out_features = 500, bias= False)
        self.fc3 = nn.Linear(in_features = 500, out_features = 72, bias= False) # the output 136 in order to having 2 for each of the 68 keypoint (x, y) pairs

    def forward(self, x, init=False):

        #x = self.SRINC(x)
        x = self.drop0(self.batchnorm0(self.pool(F.relu(self.conv0(x)))))
        x = self.drop1(self.batchnorm1(self.pool(F.relu(self.conv1(x)))))
        x = self.drop2(self.batchnorm2(self.pool(F.relu(self.conv2(x)))))
        x = self.drop3(self.batchnorm3(self.pool(F.relu(self.conv3(x)))))
        x = self.drop4(self.batchnorm4(self.pool(F.relu(self.conv4(x)))))
        x = self.drop5(self.batchnorm5(F.relu(self.conv5(x))))

        x = x.view(x.size(0), -1)

        if init == True:
            _ , temp = x.shape
            self.flat = temp
            return

        x = self.drop5(F.relu(self.fc1(x)))
        x = self.drop6(F.relu(self.fc2(x)))
        x = self.fc3(x)
        x = torch.sigmoid(x)

        return x

if __name__ == "__main__":

    t = torch.zeros([1, 1, int(160*0.95),int(256*0.95)], dtype=torch.float32).to("cpu")

    model = Net(t, 0.1).to("cpu")
    #torchsummary.summary(model, t[0].shape, device="cpu")
