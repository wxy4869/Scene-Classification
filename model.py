import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = self.layers(x) + self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, dropout_rate, num_classes=15):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),  # image channel = 1
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_x = self._make_layer(block, 64, layers[0], stride=1)
        self.conv3_x = self._make_layer(block, 128, layers[1], stride=2)
        self.conv4_x = self._make_layer(block, 256, layers[2], stride=2)
        self.conv5_x = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride=stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)  # [N, 1, 224, 224] -> [N, 64, 112, 112]
        x = self.maxpool(x)  # -> [N, 64, 56, 56]
        
        x = self.conv2_x(x)  # -> [N, 64, 56, 56]
        x = self.conv3_x(x)  # -> [N, 128, 28, 28]
        x = self.conv4_x(x)  # -> [N, 256, 14, 14]
        x = self.conv5_x(x)  # -> [N, 512, 7, 7]
        
        x = self.avgpool(x)  # -> [N, 512, 1, 1]
        x = torch.flatten(x, 1)  # -> [N, 512]
        x = self.dropout(x)
        x = self.fc(x)  # -> [N, num_classes]
        return x


def build_model(layers=18, dropout_rate=0.1, num_classes=15):
    if layers == 18:
        return ResNet(BasicBlock, [2, 2, 2, 2], dropout_rate=dropout_rate, num_classes=num_classes)
    elif layers == 34:
        return ResNet(BasicBlock, [3, 4, 6, 3], dropout_rate=dropout_rate, num_classes=num_classes)
