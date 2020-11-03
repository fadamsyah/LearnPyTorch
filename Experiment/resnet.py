# ResNet implementation for CIFAR 10 and CIFAR 100
# This code is inspired from https://github.com/kuangliu/pytorch-cifar
# This code is slightly different from the architecture described in the original paper

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels_in, channels_out, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels_out)
        self.conv2 = nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels_out)
        
        self.shortcut = None
        if channels_in != channels_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(channels_in, channels_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels_out),
            )
    
    def forward(self, x):
        shct = x
        if self.shortcut is not None: shct = self.shortcut(shct)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += shct
        x = F.relu(x)
        return x
        
class ResNetShallow(nn.Module):
    def __init__(self, first_channels, num_blocks, num_classes):
        super(ResNetShallow, self).__init__()
        self.channels = first_channels
        
        self.conv1 = nn.Conv2d(3, self.channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.channels)
        self.layer1 = self._make_layer(num_blocks[0], False)
        self.layer2 = self._make_layer(num_blocks[1], True)
        self.layer3 = self._make_layer(num_blocks[2], True)
        self.layer4 = self._make_layer(num_blocks[3], True)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(self.channels, num_classes)
        
    def _make_layer(self, num_blocks, downsample=True):
        layers = []
        for i in range(num_blocks):
            if i==0 and downsample:
                layers.append(ResBlock(self.channels, self.channels*2, 2))
                self.channels *= 2
            else:
                layers.append(ResBlock(self.channels, self.channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
    
def ResNet18(first_channels, num_classes):
    return ResNetShallow(first_channels, [2, 2, 2, 2], num_classes)

def ResNet34(first_channels, num_classes):
    return ResNetShallow(first_channels, [3, 4, 6, 3], num_classes)