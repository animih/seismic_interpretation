import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.down_sample = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        skip_out = self.double_conv(x)
        down_out = self.down_sample(skip_out)
        return (down_out, skip_out)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_sample_mode, size=None):
        super(UpBlock, self).__init__()
        if up_sample_mode == 'conv_transpose':
            self.up_sample = nn.ConvTranspose2d(in_channels - out_channels, in_channels - out_channels, kernel_size=2,
                                                stride=2)
        elif up_sample_mode == 'bilinear':
            if size == None:
                self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            else:
                self.up_sample = nn.Upsample(size=size, mode='bilinear', align_corners=True)
        else:
            raise ValueError("Unsupported `up_sample_mode` (can take one of `conv_transpose` or `bilinear`)")
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, down_input, skip_input):
        x = self.up_sample(down_input)
        x = torch.cat([x, skip_input], dim=1)
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_classes=1, up_sample_mode='bilinear'):
        super(UNet, self).__init__()
        self.up_sample_mode = up_sample_mode
        # Downsampling Path
        self.down_conv1 = DownBlock(in_channels, 16)
        self.down_conv2 = DownBlock(16, 32)
        self.down_conv3 = DownBlock(32, 64)
        # self.down_conv4 = DownBlock(64, 128)
        # Bottleneck
        self.double_conv = DoubleConv(64, 128)
        # Upsampling Path
        # self.up_conv4 = UpBlock(128 + 256, 128, self.up_sample_mode, size = (43, 42))
        self.up_conv3 = UpBlock(64 + 128, 64, self.up_sample_mode, size=(87, 85))
        self.up_conv2 = UpBlock(64 + 32, 32, self.up_sample_mode, size=(175, 170))
        self.up_conv1 = UpBlock(32 + 16, 16, self.up_sample_mode, size=(351, 341))
        # Final Convolution
        self.conv_last = nn.Conv2d(16, out_classes, kernel_size=1)

    def forward(self, x):
        x, skip1_out = self.down_conv1(x)
        x, skip2_out = self.down_conv2(x)
        x, skip3_out = self.down_conv3(x)
        # x, skip4_out = self.down_conv4(x)
        x = self.double_conv(x)
        # x = self.up_conv4(x, skip4_out)
        x = self.up_conv3(x, skip3_out)
        x = self.up_conv2(x, skip2_out)
        x = self.up_conv1(x, skip1_out)
        x = self.conv_last(x)
        return x

class CNN_st(nn.Module):
    def __init__(self, in_channels=2):
        super(CNN_st, self).__init__()
        # Downsampling Path
        self.down_conv1 = DownBlock(in_channels, 16)
        self.down_conv2 = DownBlock(16, 32)
        self.down_conv3 = DownBlock(32, 64)
        self.down_conv4 = DownBlock(64, 64) # 43 x 42 out
        self.down_conv5 = DownBlock(64, 32) # 21 x 12 out
        self.down_conv6 = DownBlock(32, 16) # 10 x 10 out
        self.down_conv7 = DownBlock(16, 1) # 5 x 5 out
        # Final FC
        self.FC = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(4, 1),
        )

    def forward(self, x):
        x, _ = self.down_conv1(x)
        x, _ = self.down_conv2(x)
        x, _ = self.down_conv3(x)
        x, _ = self.down_conv4(x)
        x, _ = self.down_conv5(x)
        x, _ = self.down_conv6(x)
        x, _ = self.down_conv7(x)

        x = self.FC(x)
        return x


class CS_model(nn.Module):
    def __init__(self, in_channels=1, out_classes=1, up_sample_mode='bilinear'):
        super(CS_model, self).__init__()

        # U-net part
        self.up_sample_mode = up_sample_mode
        # Downsampling Path
        self.down_conv1 = DownBlock(in_channels, 16)
        self.down_conv2 = DownBlock(16, 32)
        self.down_conv3 = DownBlock(32, 64)
        # Bottleneck
        self.double_conv = DoubleConv(64, 128)
        # Upsampling Path
        self.up_conv3 = UpBlock(64 + 128, 64, self.up_sample_mode, size=(87, 85))
        self.up_conv2 = UpBlock(64 + 32, 32, self.up_sample_mode, size=(175, 170))
        self.up_conv1 = UpBlock(32 + 16, 16, self.up_sample_mode, size=(351, 341))
        # Final Convolution
        self.conv_last = nn.Conv2d(16, out_classes, kernel_size=1)

        # scalar part
        self.down_conv1_sc = DownBlock(in_channels, 16)
        self.down_conv2_sc = DownBlock(16, 32)
        self.down_conv3_sc = DownBlock(32, 64)
        self.down_conv4_sc = DownBlock(64, 64)  # 43 x 42 out
        self.down_conv5_sc = DownBlock(64, 32)  # 21 x 12 out
        self.down_conv6_sc = DownBlock(32, 16)  # 10 x 10 out
        self.down_conv7_sc = DownBlock(16, 1)  # 5 x 5 out
        # Final FC
        self.FC = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(4, 1),
        )

        # CS-units
        self.cs1 = nn.Linear(2, 2)
        with torch.no_grad():
            self.cs1.weight[0, 0] = .9
            self.cs1.weight[0, 1] = .1
            self.cs1.weight[1, 0] = .1
            self.cs1.weight[1, 1] = .9
        self.cs2 = nn.Linear(2, 2)
        with torch.no_grad():
            self.cs2.weight[0, 0] = .9
            self.cs2.weight[0, 1] = .1
            self.cs2.weight[1, 0] = .1
            self.cs2.weight[1, 1] = .9

    def forward(self, x):
        x1, skip1_out = self.down_conv1(x)
        x2, _ = self.down_conv1_sc(x)

        x1_t, skip2_out = self.down_conv2(x1 * self.cs1.weight[0, 0] + x2 * self.cs1.weight[0, 1])
        x2_t, _ = self.down_conv2_sc(x1 * self.cs1.weight[1, 0] + x2 * self.cs1.weight[1, 1])

        x1, skip3_out = self.down_conv3(x1_t * self.cs2.weight[0, 0] + x2_t * self.cs2.weight[0, 1])
        x2, _ = self.down_conv3_sc(x1_t * self.cs2.weight[1, 0] + x2_t * self.cs2.weight[1, 1])

        x2, _ = self.down_conv4_sc(x2)
        x2, _ = self.down_conv5_sc(x2)
        x2, _ = self.down_conv6_sc(x2)
        x2, _ = self.down_conv7_sc(x2)
        x2 = self.FC(x2)

        x1 = self.double_conv(x1)

        x1 = self.up_conv3(x1, skip3_out)
        x1 = self.up_conv2(x1, skip2_out)
        x1 = self.up_conv1(x1, skip1_out)
        x1 = self.conv_last(x1)

        return x1, x2

class U_net_cat(nn.Module):
    def __init__(self, in_channels=1, out_classes=1, up_sample_mode='bilinear'):
        super(U_net_cat, self).__init__()

        # U-net part
        self.up_sample_mode = up_sample_mode
        # Downsampling Path
        self.down_conv1 = DownBlock(in_channels, 16)
        self.down_conv2 = DownBlock(16, 32)
        self.down_conv3 = DownBlock(32, 64)
        # Bottleneck
        self.double_conv = DoubleConv(64, 128)
        # Upsampling Path
        self.up_conv3 = UpBlock(64 + 128, 64, self.up_sample_mode, size=(87, 85))
        self.up_conv2 = UpBlock(64 + 32, 32, self.up_sample_mode, size=(175, 170))
        self.up_conv1 = UpBlock(32 + 16, 16, self.up_sample_mode, size=(351, 341))
        # Final Convolution
        self.conv_last = nn.Conv2d(16, out_classes, kernel_size=1)

        # scalar part
        self.down_conv1_sc = DownBlock(in_channels+16, 16)
        self.down_conv2_sc = DownBlock(16+32, 32)
        self.down_conv3_sc = DownBlock(32+64, 64)
        self.down_conv4_sc = DownBlock(64, 64)  # 43 x 42 out
        self.down_conv5_sc = DownBlock(64, 32)  # 21 x 12 out
        self.down_conv6_sc = DownBlock(32, 16)  # 10 x 10 out
        self.down_conv7_sc = DownBlock(16, 1)  # 5 x 5 out
        # Final FC
        self.FC = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(4, 1),
        )

    def forward(self, x):
        x1, skip1_out = self.down_conv1(x)
        x2, _ = self.down_conv1_sc(torch.cat([x, skip1_out], dim=1))

        x1, skip2_out = self.down_conv2(x1)
        x2, _ = self.down_conv2_sc(torch.cat([x2, skip2_out], dim=1))

        x1, skip3_out = self.down_conv3(x1)
        x2, _ = self.down_conv3_sc(torch.cat([x2, skip3_out], dim=1))

        x2, _ = self.down_conv4_sc(x2)
        x2, _ = self.down_conv5_sc(x2)
        x2, _ = self.down_conv6_sc(x2)
        x2, _ = self.down_conv7_sc(x2)
        x2 = self.FC(x2)

        x1 = self.double_conv(x1)

        x1 = self.up_conv3(x1, skip3_out)
        x1 = self.up_conv2(x1, skip2_out)
        x1 = self.up_conv1(x1, skip1_out)
        x1 = self.conv_last(x1)

        return x1, x2