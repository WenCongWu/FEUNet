import torch
import torch.nn as nn
from torch.autograd import Variable
import functions
import torch.nn.functional as F


class UpSampleFeatures(nn.Module):
    r"""Implements the last layer of FEUNet
    """
    def __init__(self):
        super(UpSampleFeatures, self).__init__()

    def forward(self, x):
        return functions.upsamplefeatures(x)


class Up(nn.Module):

    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x1, x):
        x2 = self.up(x1)

        diffY = x.size()[2] - x2.size()[2]
        diffX = x.size()[3] - x2.size()[3]
        x3 = F.pad(x2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x4 = torch.cat([x3, x], dim=1)
        return x4


class Down(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.MaxPool2d(2)

    def forward(self, x):
        out = self.down(x)
        return out


class Intermediate(nn.Module):
    r"""Implements the middel part of the FEUNet architecture
    """
    def __init__(self, input_features, middle_features, num_conv_layers):
        super(Intermediate, self).__init__()
        self.kernel_size = 3
        self.padding = 1
        self.input_features = input_features
        self.num_conv_layers = num_conv_layers
        self.middle_features = middle_features
        if self.input_features == 5:
            self.output_features = 4 #Grayscale image
        elif self.input_features == 15:
            self.output_features = 12 #RGB image
        else:
            raise Exception('Invalid number of input features')

        part = [1, 2, 3, 4, 5, 6]
        for i in range(len(part)):
            if part[i] == 1:
                layers_1 = []
                layers_1.append(nn.Conv2d(in_channels=self.input_features, out_channels=self.middle_features, kernel_size=self.kernel_size, padding=self.padding, bias=False))
                layers_1.append(nn.ReLU(inplace=True))

                layers_1.append(nn.Conv2d(in_channels=self.middle_features, out_channels=self.middle_features, kernel_size=self.kernel_size, padding=self.padding, bias=False))
                layers_1.append(nn.BatchNorm2d(self.middle_features))
                layers_1.append(nn.ReLU(inplace=True))

                layers_1.append(nn.Conv2d(in_channels=self.middle_features, out_channels=self.middle_features, kernel_size=self.kernel_size, padding=self.padding, bias=False))
                layers_1.append(nn.BatchNorm2d(self.middle_features))
                layers_1.append(nn.ReLU(inplace=True))

                layers_1.append(nn.Conv2d(in_channels=self.middle_features, out_channels=self.middle_features, kernel_size=self.kernel_size, padding=self.padding, bias=False))
                layers_1.append(nn.BatchNorm2d(self.middle_features))
                layers_1.append(nn.ReLU(inplace=True))

                self.itermediate_1 = nn.Sequential(*layers_1)

            if part[i] == 2:
                layers_2 = []
                layers_2.append(nn.Conv2d(in_channels=self.middle_features, out_channels=self.middle_features, kernel_size=self.kernel_size, padding=self.padding, bias=False))
                layers_2.append(nn.BatchNorm2d(self.middle_features))
                layers_2.append(nn.ReLU(inplace=True))

                layers_2.append(nn.Conv2d(in_channels=self.middle_features, out_channels=self.middle_features, kernel_size=self.kernel_size, padding=self.padding, bias=False))
                layers_2.append(nn.BatchNorm2d(self.middle_features))
                layers_2.append(nn.ReLU(inplace=True))

                self.itermediate_2 = nn.Sequential(*layers_2)

            if part[i] == 3:
                layers_3 = []
                layers_3.append(nn.Conv2d(in_channels=self.middle_features, out_channels=self.middle_features, kernel_size=self.kernel_size, padding=self.padding, bias=False))
                layers_3.append(nn.BatchNorm2d(self.middle_features))
                layers_3.append(nn.ReLU(inplace=True))

                layers_3.append(nn.Conv2d(in_channels=self.middle_features, out_channels=self.middle_features, kernel_size=self.kernel_size, padding=self.padding, bias=False))
                layers_3.append(nn.BatchNorm2d(self.middle_features))
                layers_3.append(nn.ReLU(inplace=True))

                self.itermediate_3 = nn.Sequential(*layers_3)

            if part[i] == 4:
                layers_4 = []
                self.middle_features = 192

                layers_4.append(nn.Conv2d(in_channels=self.middle_features, out_channels=self.middle_features, kernel_size=self.kernel_size, padding=self.padding, bias=False))
                layers_4.append(nn.BatchNorm2d(self.middle_features))
                layers_4.append(nn.ReLU(inplace=True))

                layers_4.append(nn.Conv2d(in_channels=self.middle_features, out_channels=self.middle_features, kernel_size=self.kernel_size, padding=self.padding, bias=False))
                layers_4.append(nn.BatchNorm2d(self.middle_features))
                layers_4.append(nn.ReLU(inplace=True))

                layers_4.append(nn.Conv2d(in_channels=self.middle_features, out_channels=self.middle_features, kernel_size=self.kernel_size, padding=self.padding, bias=False))
                layers_4.append(nn.BatchNorm2d(self.middle_features))
                layers_4.append(nn.ReLU(inplace=True))

                layers_4.append(nn.Conv2d(in_channels=self.middle_features, out_channels=self.output_features, kernel_size=self.kernel_size, padding=self.padding, bias=False))

                self.itermediate_4 = nn.Sequential(*layers_4)

            if part[i] == 5:
                self.up = Up()

            if part[i] == 6:
                self.down = Down()

    def forward(self, x):
        x1 = self.itermediate_1(x)
        x2 = self.down(x1)
        x3 = self.itermediate_2(x2)
        x4 = self.itermediate_3(x3)
        x5 = self.up(x4, x1)
        out = self.itermediate_4(x5)
        return out


class FEUNet(nn.Module):
    r"""Implements the FEUNet architecture
    """
    def __init__(self, num_input_channels):
        super(FEUNet, self).__init__()
        self.num_input_channels = num_input_channels
        if self.num_input_channels == 1:
            # Grayscale image
            self.num_feature_maps = 64
            self.num_conv_layers = 15
            self.downsampled_channels = 5
            self.output_features = 4
        elif self.num_input_channels == 3:
            # RGB image
            self.num_feature_maps = 96
            self.num_conv_layers = 12
            self.downsampled_channels = 15
            self.output_features = 12
        else:
            raise Exception('Invalid number of input features')

        self.intermediate = Intermediate(input_features=self.downsampled_channels, middle_features=self.num_feature_maps, num_conv_layers=self.num_conv_layers)
        self.upsamplefeatures = UpSampleFeatures()

    def forward(self, x, noise_sigma):
        concat_noise_x = functions.concatenate_input_noise_map(x.data, noise_sigma.data)
        concat_noise_x = Variable(concat_noise_x)
        h = self.intermediate(concat_noise_x)
        pred_noise = self.upsamplefeatures(h)
        return pred_noise
