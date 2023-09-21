import math

import torch.nn as nn
import torchvision.models

#from utils.logconf import logging
from torchvision.models import resnet34, resnet18, ResNet18_Weights, ResNet34_Weights


#log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
#log.setLevel(logging.DEBUG)


class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.activation = nn.Tanh()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 50 * 192, 1024)  # Assuming input image size is (128, 431)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool(self.activation(self.conv1(x)))
        #print(x.shape)
        x = x.view(-1, 16 * 50 * 192)
        #print(x.shape)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)

        return x

# input is  (128, 431)
class CNNModel_2(nn.Module):
    def __init__(self, num_classes, n_channels=32):
        super(CNNModel_2, self).__init__()
        self.n_channels = n_channels
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=n_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels // 2, kernel_size=3, stride=1, padding=1)
        self.activation = nn.Tanh()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.fc1 = nn.Linear(16 * 64 * n_channels // 2, 128)
        self.fc1 = nn.Linear(16 * 32 * 107, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        #print("Input shape:", x.shape) [320, 1, 128, 431])
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))
        #print("Shape before reshaping:", x.shape) ([80, 16, 32, 107])
        # num_elements = x.size(0) * x.size(1) * x.size(2)
        x = x.view(-1, 16 * 32 * 107)
        # x = x.view(-1, 16 * 64 * self.n_channels // 2)
        #print("Shape after reshaping:", x.shape)

        x = self.activation(self.fc1(x))
        x = self.fc2(x)

        return x

class CNNModel_BatchNorm(nn.Module):
    def __init__(self, num_classes, n_channels=32):
        super(CNNModel_BatchNorm, self).__init__()
        self.n_channels = n_channels
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=n_channels, kernel_size=3, stride=1, padding=1)
        self.conv1_batchnorm = nn.BatchNorm2d(num_features=n_channels)
        self.conv2 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels // 2, kernel_size=3, stride=1, padding=1)
        self.conv2_batchnorm = nn.BatchNorm2d(num_features=n_channels // 2)

        self.activation = nn.Tanh()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.fc1 = nn.Linear(16 * 64 * n_channels // 2, 128)
        self.fc1 = nn.Linear(16 * 32 * 107, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        #print("Input shape:", x.shape) [320, 1, 128, 431])
        x = self.conv1_batchnorm(self.conv1(x))
        x = self.pool(self.activation(x))
        x = self.conv2_batchnorm(self.conv2(x))
        x = self.pool(self.activation(x))
        #print("Shape before reshaping:", x.shape) ([80, 16, 32, 107])
        # num_elements = x.size(0) * x.size(1) * x.size(2)
        x = x.view(-1, 16 * 32 * 107)
        # x = x.view(-1, 16 * 64 * self.n_channels // 2)
        #print("Shape after reshaping:", x.shape)

        x = self.activation(self.fc1(x))
        x = self.fc2(x)

        return x


class ESC50_Block(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super(ESC50_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.batch_norm = nn.BatchNorm2d(num_features=in_channels)

        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1,
                               bias=False)
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=conv_channels, kernel_size=3, padding=1,
                               bias=False)
        self.conv_res = nn.Conv2d(in_channels=in_channels, out_channels=conv_channels, kernel_size=1, padding=0,
                               bias=False)
        self.batch_norm3 = nn.BatchNorm2d(num_features=conv_channels)

        nn.init.kaiming_normal_(self.conv1.weight,
                                      nonlinearity='relu')
        nn.init.constant_(self.batch_norm.weight, 0.5)
        nn.init.zeros_(self.batch_norm.bias)

    def forward(self, x):
        out1 = self.conv_res(x)
        x = self.relu(self.batch_norm(self.conv1(x)))
        x = self.relu(self.batch_norm(self.conv2(x)))
        x = self.batch_norm3(self.conv3(x))
        return self.relu(x + out1)


class CNNModel_ResBlocks(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel_ResBlocks, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, padding=1, stride=2,
                               bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.activation = nn.ReLU()

        self.block1 = ESC50_Block(64, 128)
        self.block2 = ESC50_Block(128, 256)
        self.block3 = ESC50_Block(256, 512)

        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(512 * 3 * 11, 256)
        self.fc2 = nn.Linear(256, num_classes)


    def forward(self, x):
        #print("original dimensions: ", x.shape)
        x = self.maxpool(self.activation(self.conv1(x)))
        """shape before blocks:  torch.Size([160, 64, 23, 94])
        shape before reshape:  torch.Size([160, 512, 2, 11])"""
        #print("shape before blocks: ", x.shape)
        x = self.maxpool(self.block1(x))
        x = self.maxpool(self.block2(x))
        x = self.avgpool(self.block3(x))
        #print("shape before reshape: ", x.shape)  [512, 3, 11]
        x = x.view(-1, 512 * 3 * 11)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x





class CNNModel_Res(nn.Module):
    def __init__(self, num_classes, n_channels=32):
        super(CNNModel_Res, self).__init__()

        self.n_channels = n_channels
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=n_channels, kernel_size=3, stride=1, padding=1)
        self.conv1_batchnorm = nn.BatchNorm2d(num_features=n_channels)
        self.conv2 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels // 2, kernel_size=3, stride=1, padding=1)
        self.conv2_batchnorm = nn.BatchNorm2d(num_features=n_channels // 2)
        self.conv3 = nn.Conv2d(in_channels=n_channels // 2, out_channels=n_channels // 2, kernel_size=3, stride=1,
                               padding=1)


        self.activation = nn.Tanh()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=0.5)
        # self.fc1 = nn.Linear(16 * 64 * n_channels // 2, 128)
        self.fc1 = nn.Linear(16 * 16 * 53, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1_batchnorm(self.conv1(x))
        x = self.pool(self.activation(x))
        x = self.dropout(x)
        x = self.conv2_batchnorm(self.conv2(x))
        x = self.pool(self.activation(x))
        res1 = x
        x = self.dropout(x)
        x = self.conv2_batchnorm(self.conv3(x))
        x = self.pool(self.activation(x)+res1)
        x = self.dropout(x)
        x = x.view(-1, 16 * 16 * 53)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

class CNNModel_Regularization(nn.Module):
    def __init__(self, num_classes, n_channels=32):
        super(CNNModel_Regularization, self).__init__()

        self.n_channels = n_channels
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=n_channels, kernel_size=3, stride=1, padding=1)
        self.conv1_batchnorm = nn.BatchNorm2d(num_features=n_channels)
        self.conv2 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels // 2, kernel_size=3, stride=1, padding=1)
        self.conv2_batchnorm = nn.BatchNorm2d(num_features=n_channels // 2)
        self.conv3 = nn.Conv2d(in_channels=n_channels // 2, out_channels=n_channels // 2, kernel_size=3, stride=1,
                               padding=1)


        self.activation = nn.Tanh()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=0.7)
        # self.fc1 = nn.Linear(16 * 64 * n_channels // 2, 128)
        self.fc1 = nn.Linear(16 * 16 * 53, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1_batchnorm(self.conv1(x))
        x = self.pool(self.activation(x))
        x = self.dropout(x)
        x = self.conv2_batchnorm(self.conv2(x))
        x = self.pool(self.activation(x))
        x = self.dropout(x)
        x = self.conv2_batchnorm(self.conv3(x))
        x = self.pool(self.activation(x))
        x = self.dropout(x)
        x = x.view(-1, 16 * 16 * 53)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x


class CNNModel_ResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.resnet = resnet18(ResNet18_Weights.DEFAULT)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.resnet(x)
        return x

class CNNModel_AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel_AlexNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 96, kernel_size=(11, 11), stride=(2, 2), padding=0, bias=False)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.linear1 = nn.Linear(256 * 6 * 25, 1536)
        self.linear2 = nn.Linear(1536, 1536)
        self.linear3 = nn.Linear(1536, num_classes)


        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.maxpool(self.activation(self.conv1(x)))
        x = self.maxpool(self.activation(self.conv2(x)))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        x = self.maxpool(self.activation(self.conv5(x)))
        #print("shape before flatten: ", x.shape) [200, 256, 2, 12] using normal alexNet
        #print("shape before flatten: ", x.shape) [200, 256, 6, 25]

        x = x.view(-1, 256 * 6 * 25)
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.activation(self.linear3(x))

        return self.softmax(x)


class CNNModel_BaseModel(nn.Module):
    def __init__(self, num_classes, n_channels=64):
        super(CNNModel_BaseModel, self).__init__()
        self.n_channels = n_channels
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=n_channels, kernel_size=9, stride=2, padding=1)
        self.conv1_batchnorm = nn.BatchNorm2d(num_features=n_channels)
        self.conv2 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels * 2, kernel_size=3, stride=1, padding=1)
        self.conv2_batchnorm = nn.BatchNorm2d(num_features=n_channels * 2)
        self.conv3 = nn.Conv2d(in_channels=n_channels * 2, out_channels=n_channels * 4, kernel_size=3, stride=1, padding=1)
        self.conv3_batchnorm = nn.BatchNorm2d(num_features=n_channels * 4)

        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(256 * 5 * 23, 8192)
        # V2 adds one more linear layer to go from 29440 to 8192 and then to 2028 to finally 50,
        # added init weights and changed activation from tanh to relu
        self.fc2 = nn.Linear(8192, 2048)
        self.fc3 = nn.Linear(2048, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {
                nn.Linear,
                nn.Conv3d,
                nn.Conv2d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d,
            }:
                nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    fan_in, fan_out = \
                        nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, x):
        #print("Input shape:", x.shape) [320, 1, 128, 431])
        x = self.conv1_batchnorm(self.conv1(x))
        x = self.pool(self.activation(x))
        x = self.conv2_batchnorm(self.conv2(x))
        x = self.pool(self.activation(x))
        x = self.conv3_batchnorm(self.conv3(x))
        x = self.pool(self.activation(x))
        #print("before reshape: ", x.shape)
        x = x.view(-1, 256 * 5 * 23)
        # x = x.view(-1, 16 * 64 * self.n_channels // 2)
        #print("Shape after reshaping:", x.shape)

        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)

        return x

class CNNModel_BaseModelV3(nn.Module):
    def __init__(self, num_classes, n_channels=64):
        super(CNNModel_BaseModelV3, self).__init__()
        self.n_channels = n_channels
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=n_channels, kernel_size=9, stride=2, padding=1)
        self.conv1_batchnorm = nn.BatchNorm2d(num_features=n_channels)
        self.conv2 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels * 2, kernel_size=3, stride=1, padding=1)
        self.conv2_batchnorm = nn.BatchNorm2d(num_features=n_channels * 2)
        self.conv3 = nn.Conv2d(in_channels=n_channels * 2, out_channels=n_channels * 4, kernel_size=3, stride=1, padding=1)
        self.conv3_batchnorm = nn.BatchNorm2d(num_features=n_channels * 4)
        self.conv4 = nn.Conv2d(in_channels=n_channels * 4, out_channels=n_channels * 8, kernel_size=3, stride=1, padding=1)
        self.conv4_batchnorm = nn.BatchNorm2d(num_features=n_channels * 8)

        self.activation = nn.Tanh()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(512 * 2 * 11, 2048)
        self.fc2 = nn.Linear(2048, num_classes)

    def forward(self, x):
        #print("Input shape:", x.shape) [320, 1, 128, 431])
        x = self.conv1_batchnorm(self.conv1(x))
        x = self.pool(self.activation(x))
        x = self.conv2_batchnorm(self.conv2(x))
        x = self.pool(self.activation(x))
        x = self.conv3_batchnorm(self.conv3(x))
        x = self.pool(self.activation(x))
        x = self.conv4_batchnorm(self.conv4(x))
        x = self.pool(self.activation(x))

        #print("before reshape: ", x.shape)
        x = x.view(-1, 512 * 2 * 11)
        # x = x.view(-1, 16 * 64 * self.n_channels // 2)
        #print("Shape after reshaping:", x.shape)

        x = self.activation(self.fc1(x))
        x = self.fc2(x)

        return x

class CNNModel_BaseModelV4(nn.Module):
    def __init__(self, num_classes, n_channels=32):
        super(CNNModel_BaseModelV4, self).__init__()
        self.n_channels = n_channels
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=n_channels, kernel_size=9, stride=2, padding=1)
        self.conv1_batchnorm = nn.BatchNorm2d(num_features=n_channels)
        self.conv2 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels * 2, kernel_size=3, stride=1, padding=1)
        self.conv2_batchnorm = nn.BatchNorm2d(num_features=n_channels * 2)
        self.conv3 = nn.Conv2d(in_channels=n_channels * 2, out_channels=n_channels * 4, kernel_size=3, stride=1, padding=1)
        self.conv3_batchnorm = nn.BatchNorm2d(num_features=n_channels * 4)
        self.conv4 = nn.Conv2d(in_channels=n_channels * 4, out_channels=n_channels * 8, kernel_size=3, stride=1, padding=1)
        self.conv4_batchnorm = nn.BatchNorm2d(num_features=n_channels * 8)

        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(256 * 2 * 11, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {
                nn.Linear,
                nn.Conv3d,
                nn.Conv2d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d,
            }:
                nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    fan_in, fan_out = \
                        nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, x):
        #print("Input shape:", x.shape) [320, 1, 128, 431])
        x = self.conv1_batchnorm(self.conv1(x))
        x = self.pool(self.activation(x))
        x = self.conv2_batchnorm(self.conv2(x))
        x = self.pool(self.activation(x))
        x = self.conv3_batchnorm(self.conv3(x))
        x = self.pool(self.activation(x))
        x = self.conv4_batchnorm(self.conv4(x))
        x = self.pool(self.activation(x))
        x = x.view(-1, 256 * 2 * 11)

        x = self.fc(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=conv_channels, kernel_size=3, stride=2, padding=1)
        self.conv1_batchnorm = nn.BatchNorm2d(num_features=conv_channels)
        self.conv2 = nn.Conv2d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=3, stride=1, padding=1)
        self.conv2_batchnorm = nn.BatchNorm2d(num_features=conv_channels)
        self.activation = nn.ReLU()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {
                nn.Linear,
                nn.Conv3d,
                nn.Conv2d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d,
            }:
                nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    fan_in, fan_out = \
                        nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, x):
        x = self.conv1_batchnorm(self.conv1(x))
        x = self.activation(x)
        x = self.conv2_batchnorm(self.conv2(x))
        x = self.activation(x)

        return x



class CNNModel_BaseModelV5(nn.Module):
    def __init__(self, num_classes, n_channels=64):
        super(CNNModel_BaseModelV5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=n_channels//2, kernel_size=3, stride=2, padding=1)
        self.conv1_batchnorm = nn.BatchNorm2d(num_features=n_channels//2)
        self.block1 = ResnetBlock(n_channels//2, n_channels)
        self.block2 = ResnetBlock(n_channels, n_channels*2)
        self.block3 = ResnetBlock(n_channels*2, n_channels*4)
        self.block4 = ResnetBlock(n_channels*4, n_channels*8)

        self.activation = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(256, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {
                nn.Linear,
                nn.Conv3d,
                nn.Conv2d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d,
            }:
                nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    fan_in, fan_out = \
                        nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, x):
        #print("Input shape:", x.shape) [320, 1, 128, 431])
        x = self.conv1_batchnorm(self.conv1(x))
        x = self.block1(self.activation(x))
        x = self.block2(x)
        x = self.block3(x)
        x = self.avgpool(x)
        #print("shape after avgpool")
        x = x.view(-1, 256)
        x = self.fc(x)

        return x

