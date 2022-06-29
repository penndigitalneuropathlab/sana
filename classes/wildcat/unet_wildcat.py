import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

import wildcat.models
from wildcat.pooling import WildcatPool2d, ClassWisePool

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_ch, in_skipch, out_ch):
        super(Upsample, self).__init__()
        # self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = double_conv(in_ch//2+in_skipch, out_ch)

    def forward(self, x1, x2):
        
        # Upsample the input
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # Pad so that the input and the skip connection are same size
        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # Creates the skip connection 
        x = torch.cat([x2, x1], dim=1)
        
        # Doubl convolution
        x = self.conv(x)
        return x

    
class ResNetWSLUpsample(nn.Module):

    def __init__(self, model, num_classes, pooling=WildcatPool2d(), dense=False):
        super(ResNetWSLUpsample, self).__init__()

        self.dense = dense
        
        self.model = model
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4)

        # classification layer
        num_features = model.layer4[1].conv1.in_channels

        # Create an upsampling layer
        self.up1 = Upsample(num_features*2, num_features//2, num_features//2)
        self.up2 = Upsample(num_features, num_features//4, num_features//4)
        self.up3 = Upsample(num_features//2, num_features//8, num_features//8)
        self.up4 = Upsample(num_features//4, num_features//32, num_features//32)
        
        # Create a classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(num_features//32, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

        self.spatial_pooling = pooling

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]
        
    def forward_to_classifier(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        xq = self.model.relu(x)
        x0 = self.model.maxpool(x)
        x1 = self.model.layer1(x0)
        x2 = self.model.layer2(x1)
        x3 = self.model.layer3(x2)
        x4 = self.model.layer4(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, xq)
        x = self.classifier(x)
        return x
        
    def forward(self, x):
        x = self.forward_to_classifier(x)
        if not self.dense:
            x = self.spatial_pooling(x)
        return x

    def get_config_optim(self, lr, lrp):
        return [{'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.up1.parameters()},
                {'params': self.up2.parameters()},
                {'params': self.up3.parameters()},
                {'params': self.up4.parameters()},
                {'params': self.classifier.parameters()},
                {'params': self.spatial_pooling.parameters()}]


class ResNet18WSLUpsample(nn.Module):

    def __init__(self, model, num_classes, pooling=WildcatPool2d(), dense=False):
        super(ResNet18WSLUpsample, self).__init__()

        self.dense = dense
        
        self.model = model
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4)

        # classification layer
        num_features = model.layer4[1].conv1.in_channels

        # Create an upsampling layer
        self.up1 = Upsample(num_features*2, num_features//2, num_features//2)
        self.up2 = Upsample(num_features, num_features//4, num_features//4)
        self.up3 = Upsample(num_features//2, num_features//8, num_features//8)
        self.up4 = Upsample(num_features//4, num_features//8, num_features//8)
        
        # Create a classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(num_features//8, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

        self.spatial_pooling = pooling

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]
        
    def forward_to_classifier(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        xq = self.model.relu(x)
        x0 = self.model.maxpool(x)
        x1 = self.model.layer1(x0)
        x2 = self.model.layer2(x1)
        x3 = self.model.layer3(x2)
        x4 = self.model.layer4(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, xq)
        x = self.classifier(x)
        return x
        
    def forward(self, x):
        x = self.forward_to_classifier(x)
        if not self.dense:
            x = self.spatial_pooling(x)
        return x

    def get_config_optim(self, lr, lrp):
        return [{'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.up1.parameters()},
                {'params': self.up2.parameters()},
                {'params': self.up3.parameters()},
                {'params': self.up4.parameters()},
                {'params': self.classifier.parameters()},
                {'params': self.spatial_pooling.parameters()}]


def resnet50_wildcat_upsample(num_classes, pretrained=True, kmax=1, kmin=None, alpha=1, num_maps=1):
    model = models.resnet50(pretrained)
    pooling = nn.Sequential()
    pooling.add_module('class_wise', ClassWisePool(num_maps))
    pooling.add_module('spatial', WildcatPool2d(kmax, kmin, alpha))
    return ResNetWSLUpsample(model, num_classes * num_maps, pooling=pooling)


def resnet18_wildcat_upsample(num_classes, pretrained=True, kmax=1, kmin=None, alpha=1, num_maps=1):
    model = models.resnet18(pretrained)
    pooling = nn.Sequential()
    pooling.add_module('class_wise', ClassWisePool(num_maps))
    pooling.add_module('spatial', WildcatPool2d(kmax, kmin, alpha))
    return ResNet18WSLUpsample(model, num_classes * num_maps, pooling=pooling)

