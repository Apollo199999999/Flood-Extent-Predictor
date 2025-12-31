import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        down = self.conv(x)
        x = self.pool(down)
        return down, x

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x2 = x2[:, :, 
                diffY//2: diffY//2 + x1.size()[2],
                diffX//2: diffX//2 + x1.size()[3]]

        x = torch.cat([x1, x2], 1)
        return self.conv(x)

class BaseUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.down_conv1 = DownSample(in_channels, 32)
        self.down_conv2 = DownSample(32, 64)
        self.down_conv3 = DownSample(64, 128)
        self.down_conv4 = DownSample(128, 256)

        self.extra_conv = DoubleConv(256, 512)

        self.up_conv1 = UpSample(512, 256)
        self.up_conv2 = UpSample(256, 128)
        self.up_conv3 = UpSample(128, 64)
        self.up_conv4 = UpSample(64, 32)

        self.final_conv = nn.Conv2d(32, out_channels, 1, 1)
        self.pool = nn.AvgPool2d(10, 10)

    def forward(self, x):
        x = F.pad(x, (6, 6, 6, 6))

        down1, x = self.down_conv1(x)
        down2, x = self.down_conv2(x)
        down3, x = self.down_conv3(x)
        down4, x = self.down_conv4(x)

        x = self.extra_conv(x)

        x = self.up_conv1(x, down4)
        x = self.up_conv2(x, down3)
        x = self.up_conv3(x, down2)
        x = self.up_conv4(x, down1)

        x = self.final_conv(x)
        final = x[:, :, 6:-6, 6:-6]
        out = self.pool(final)
        out = torch.squeeze(out, 1)

        return out

