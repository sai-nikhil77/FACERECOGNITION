import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

class ResUNet(nn.Module):
    def __init__(self):
        super(ResUNet, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.res_block1 = ResidualBlock(64, 128, stride=2)
        self.res_block2 = ResidualBlock(128, 256, stride=2)

        # Decoder (U-Net-like structure)
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_res_block1 = ResidualBlock(128, 64, stride=1)
        self.deconv2 = nn.ConvTranspose2d(192, 64, kernel_size=2, stride=2)
        self.dec_res_block2 = ResidualBlock(64, 32, stride=1)
        self.conv3 = nn.Conv2d(96, 7, kernel_size=1, stride=2)
        self.fc = nn.Linear(4032, 7)


    def forward(self, x):
        # Encoder
        x = F.relu(self.conv1(x))
        x = self.res_block1(x)
        x = self.res_block2(x)

        # Decoder
        x = F.relu(self.deconv1(x))
        x = torch.cat([x, self.dec_res_block1(x)], dim=1)
        x = F.relu(self.deconv2(x))
        x = torch.cat([x, self.dec_res_block2(x)], dim=1)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return F.log_softmax(x, dim=1)