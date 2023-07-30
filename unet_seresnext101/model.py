import torch
from torch import nn, optim
import torch.nn.functional as F
import sys
from utils.init_weight import init_weight
import pretrainedmodels

def Conv3x3(in_channel, out_channel): #not change resolution
    return nn.Conv2d(in_channel,out_channel,
                      kernel_size=3,stride=1,padding=1,dilation=1,bias=False)

def Conv1x1(in_channel, out_channel): #not change resolution
    return nn.Conv2d(in_channel,out_channel,
                      kernel_size=1,stride=1,padding=0,dilation=1,bias=False)
    

class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channel, reduction):
        super().__init__()
        self.global_maxpool = nn.AdaptiveMaxPool2d(1)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1) 
        self.fc = nn.Sequential(
            Conv1x1(in_channel, in_channel//reduction).apply(init_weight),
            nn.ReLU(True),
            Conv1x1(in_channel//reduction, in_channel).apply(init_weight)
        )
        
    def forward(self, inputs):
        x1 = self.global_maxpool(inputs)
        x2 = self.global_avgpool(inputs)
        x1 = self.fc(x1)
        x2 = self.fc(x2)
        x  = torch.sigmoid(x1 + x2)
        return x
    
    
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv3x3 = Conv3x3(2,1).apply(init_weight)
        
    def forward(self, inputs):
        x1,_ = torch.max(inputs, dim=1, keepdim=True)
        x2 = torch.mean(inputs, dim=1, keepdim=True)
        x  = torch.cat([x1,x2], dim=1)
        x  = self.Conv3x3(x)
        x  = torch.sigmoid(x)
        return x
    
    
class CBAM(nn.Module):
    def __init__(self, in_channel, reduction):
        super().__init__()
        self.channel_attention = ChannelAttentionModule(in_channel, reduction)
        self.spatial_attention = SpatialAttentionModule()
        
    def forward(self, inputs):
        x = inputs * self.channel_attention(inputs)
        x = x * self.spatial_attention(x)
        return x
    
    
class CenterBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = Conv3x3(in_channel, out_channel).apply(init_weight)
        
    def forward(self, inputs):
        x = self.conv(inputs)
        return x


class DecodeBlock(nn.Module):
    def __init__(self, in_channel, out_channel, upsample):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channel).apply(init_weight)
        self.upsample = nn.Sequential()
        if upsample:
            self.upsample.add_module('upsample',nn.Upsample(scale_factor=2, mode='nearest'))
        self.Conv3x3_1 = Conv3x3(in_channel, in_channel).apply(init_weight)
        self.bn2 = nn.BatchNorm2d(in_channel).apply(init_weight)
        self.Conv3x3_2 = Conv3x3(in_channel, out_channel).apply(init_weight)
        self.cbam = CBAM(out_channel, reduction=16)
        self.Conv1x1   = Conv1x1(in_channel, out_channel).apply(init_weight)
        
    def forward(self, inputs):
        x  = F.relu(self.bn1(inputs))
        x  = self.upsample(x)
        x  = self.Conv3x3_1(x)
        x  = self.Conv3x3_2(F.relu(self.bn2(x)))
        x  = self.cbam(x)
        x += self.Conv1x1(self.upsample(inputs)) #shortcut
        return x
 
            
#U-Net SeResNext101 + CBAM + hypercolumns + deepsupervision
class UNet_SeresNext101(nn.Module):
    def __init__(self, load_weights=True):
        super().__init__()
        self.n_channels = 3
        
        #encoder
        model_name = 'se_resnext101_32x4d'
        if load_weights:
            seresnext101 = pretrainedmodels.__dict__[model_name](pretrained='imagenet')
        else:
            seresnext101 = pretrainedmodels.__dict__[model_name](pretrained=None)
        
        self.encoder0 = nn.Sequential(
            seresnext101.layer0.conv1, #(*,3,h,w)->(*,64,h/2,w/2)
            seresnext101.layer0.bn1,
            seresnext101.layer0.relu1,
        )
        self.encoder1 = nn.Sequential(
            seresnext101.layer0.pool, #->(*,64,h/4,w/4)
            seresnext101.layer1 #->(*,256,h/4,w/4)
        )
        self.encoder2 = seresnext101.layer2 #->(*,512,h/8,w/8)
        self.encoder3 = seresnext101.layer3 #->(*,1024,h/16,w/16)
        self.encoder4 = seresnext101.layer4 #->(*,2048,h/32,w/32)
        
        #center
        self.center  = CenterBlock(2048,512) #->(*,512,h/32,w/32)
        
        #decoder
        self.decoder4 = DecodeBlock(512+2048,64, upsample=True) #->(*,64,h/16,w/16)
        self.decoder3 = DecodeBlock(64+1024,64, upsample=True) #->(*,64,h/8,w/8)
        self.decoder2 = DecodeBlock(64+512,64,  upsample=True) #->(*,64,h/4,w/4) 
        self.decoder1 = DecodeBlock(64+256,64,   upsample=True) #->(*,64,h/2,w/2) 
        self.decoder0 = DecodeBlock(64,64, upsample=True) #->(*,64,h,w) 
        
        #upsample
        self.upsample4 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        #final conv
        self.final_conv = nn.Sequential(
            Conv3x3(320,64).apply(init_weight),
            nn.ELU(True),
            Conv1x1(64,1).apply(init_weight)
        )
        
    def forward(self, inputs):
        #encoder
        x0 = self.encoder0(inputs) #->(*,64,h/2,w/2)
        x1 = self.encoder1(x0) #->(*,256,h/4,w/4)
        x2 = self.encoder2(x1) #->(*,512,h/8,w/8)
        x3 = self.encoder3(x2) #->(*,1024,h/16,w/16)
        x4 = self.encoder4(x3) #->(*,2048,h/32,w/32)
        
        #center
        y5 = self.center(x4) #->(*,320,h/32,w/32)
        
        #decoder
        y4 = self.decoder4(torch.cat([x4,y5], dim=1)) #->(*,64,h/16,w/16)
        y3 = self.decoder3(torch.cat([x3,y4], dim=1)) #->(*,64,h/8,w/8)
        y2 = self.decoder2(torch.cat([x2,y3], dim=1)) #->(*,64,h/4,w/4)
        y1 = self.decoder1(torch.cat([x1,y2], dim=1)) #->(*,64,h/2,w/2) 
        y0 = self.decoder0(y1) #->(*,64,h,w)
        
        #hypercolumns
        y4 = self.upsample4(y4) #->(*,64,h,w)
        y3 = self.upsample3(y3) #->(*,64,h,w)
        y2 = self.upsample2(y2) #->(*,64,h,w)
        y1 = self.upsample1(y1) #->(*,64,h,w)
        hypercol = torch.cat([y0,y1,y2,y3,y4], dim=1)
        
        #final conv
        logits = self.final_conv(hypercol) #->(*,1,h,w)
        
        return logits