import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torchvision.models import resnet18, resnet34, resnet50, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

# Just a regular reusable convblock
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batchnorm_momentum):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.batchnorm = nn.BatchNorm2d(out_channels, momentum=batchnorm_momentum)
        self.pool = nn.MaxPool2d(2, 2)
        self.act = nn.GELU()


    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.batchnorm(x)
        x = self.pool(x)
        return x

class ConvNet(nn.Module):
    def __init__(self, shape=(12, 25, 25), momentum=0.9):
        super(ConvNet, self).__init__()
        self.conv_block1 = ConvBlock(shape[0], 16, momentum)
        self.conv_block2 = ConvBlock(16, 32, momentum)
        self.global_pool = nn.AdaptiveMaxPool2d((1,1))

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.global_pool(x)
        return x

class CropModel(nn.Module):
    def __init__(self, shape=(30, 12, 25, 25), n_classes=7):
        super(CropModel, self).__init__()

        self.convnet = torchvision.models.resnet18(pretrained=True)

        self.convnet = ConvNet(shape[1:], momentum=0.9)
        self.gru = nn.GRU(32, 12, batch_first=True)
        self.fc1 = nn.Linear(12, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, n_classes)
        self.dropout = nn.Dropout(0.5)
        self.act = nn.GELU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        x = x.view(batch_size * timesteps, C, H, W)
        x = self.convnet(x)
        print(x.shape)
        x = x.view(batch_size, timesteps, -1)
        x, _ = self.gru(x)
        x = self.fc1(x[:, -1, :]) # Take the last time step
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        return x


class ResNetCropModel(nn.Module):
    def __init__(self, shape=(30, 12, 25, 25), n_classes=3, size="small"):
        super(ResNetCropModel, self).__init__()
        
        if size == "small":
            self.convnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.convnet.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.convnet.fc = nn.Identity()
        elif size == "medium":
            self.convnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            self.convnet.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.convnet.fc = nn.Identity()
        else:
            self.convnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.convnet.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.convnet.fc = nn.Identity()
        
        if size == "large":
            self.gru = nn.GRU(2048, 256, batch_first=True)
            self.fc1 = nn.Linear(256, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, n_classes)
        else:
            self.gru = nn.GRU(512, 32, batch_first=True)
            self.fc1 = nn.Linear(32, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, n_classes)
        self.dropout = nn.Dropout(0.2)
        self.act = nn.GELU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        x = x.view(batch_size * timesteps, C, H, W)
        x = self.convnet(x).unsqueeze(-1).unsqueeze(-1)
        x = x.view(batch_size, timesteps, -1)
        x, _ = self.gru(x)
        x = self.fc1(x[:, -1, :]) # Take the last time step
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        return x
    

def get_crop_model(model_name, shape, n_classes, size):
    
    if model_name == "crop":
        if size == "small":
            return CropModel(shape, n_classes, size)
        elif size == "medium":
            return CropModel(shape, n_classes, size)
        elif size == "large":
            return CropModel(shape, n_classes, size)
        else:
            raise ValueError("Invalid size")
    elif model_name == "resnet_crop":
        if size == "small":
            return ResNetCropModel(shape, n_classes, size)
        elif size == "medium":
            return ResNetCropModel(shape, n_classes, size)
        elif size == "large":
            return ResNetCropModel(shape, n_classes, size)
        else:
            raise ValueError("Invalid size")
    else:
        raise ValueError("Invalid model name")
    
    