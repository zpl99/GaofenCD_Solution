import sys

import torch
from torch import nn
import torch.nn.functional as F
from .PPM import PPM
from .FPN import FPN
from lib.batchnorm import SynchronizedBatchNorm2d


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)

def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            SynchronizedBatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )
class PPM_FPN(nn.Module):
    def __init__(self,configs):
        super(PPM_FPN, self).__init__()
        self.configs=configs
        # self.ppm_bins=[1,2,3,6] # bins for seperate ppm
        # self.ppm_in_dim=self.configs.decoder.ppm_in_dim # 768
        # self.ppm_head=PPM(in_dim=self.ppm_in_dim,reduction_dim=int(self.ppm_in_dim/(len(self.ppm_bins))),bins=self.ppm_bins)
        self.FPN_head=FPN(self.configs)
        self.fusion_module=conv3x3_bn_relu(self.configs["decoder"]["reduction_dim"]*4,self.configs["decoder"]["reduction_dim"])
        self.scene_head = nn.Sequential(
            conv3x3_bn_relu(self.configs["decoder"]["reduction_dim"], self.configs["decoder"]["reduction_dim"], 1),
            nn.Conv2d( self.configs["decoder"]["reduction_dim"], 1, kernel_size=1, bias=True) # 分成两类，前景和背景
        )
    def forward(self,inputs):
        # inputs[-1]=self.ppm_head(inputs[-1])
        p2, p3, p4, p5=self.FPN_head(inputs)
        p2=F.interpolate(p2,224,mode='bilinear')
        p3=F.interpolate(p3,224,mode='bilinear')
        p4=F.interpolate(p4,224,mode='bilinear')
        p5 = F.interpolate(p5, 224, mode='bilinear')
        fusion_features=self.fusion_module(torch.cat([p2,p3,p4,p5],dim=1))
        result=self.scene_head(fusion_features)
        return result
def test():
    print("aaa")








