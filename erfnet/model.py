# ERFNet full model definition for Pytorch
# Sept 2017
# Eduardo Romera
#######################

import torch
import torch.nn as nn
import torch.nn.functional as F

class DownsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)
    

class NonBottleneck1D(nn.Module):
    def __init__(self, chann, dropprob, dilated):        
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1,0), bias=True)

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1*dilated,0), bias=True, dilation = (dilated,1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1*dilated), bias=True, dilation = (1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)
        

    def forward(self, input):
        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        output = self.dropout(output)
        
        return F.relu(output+input)    #+input = identity (residual connection)


class Encoder(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.initial_block = DownsamplerBlock(n_channels,16)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16,64))

        for x in range(0, 5):    #5 times
           self.layers.append(NonBottleneck1D(64, 0.03, 1)) 

        self.layers.append(DownsamplerBlock(64,128))

        for x in range(0, 2):    #2 times
            self.layers.append(NonBottleneck1D(128, 0.3, 2))
            self.layers.append(NonBottleneck1D(128, 0.3, 4))
            self.layers.append(NonBottleneck1D(128, 0.3, 8))
            self.layers.append(NonBottleneck1D(128, 0.3, 16))

    def forward(self, input):
        output = self.initial_block(input)
        for layer in self.layers:
            output = layer(output)
        return output


class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(128,64))
        self.layers.append(NonBottleneck1D(64, 0, 1))
        self.layers.append(NonBottleneck1D(64, 0, 1))

        self.layers.append(UpsamplerBlock(64,16))
        self.layers.append(NonBottleneck1D(16, 0, 1))
        self.layers.append(NonBottleneck1D(16, 0, 1))

        self.layers.append(nn.ConvTranspose2d(16, 1, 2, stride=2, padding=0, output_padding=0, bias=True))

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer(output)
        return output

#ERFNet
class Net(nn.Module):
    def __init__(self, n_channels):  #use encoder to pass pretrained encoder
        super().__init__()

        self.n_channels = n_channels
        self.encoder = Encoder(n_channels)
        self.decoder = Decoder()

    def forward(self, input):
        output = self.encoder(input)
        return self.decoder.forward(output)