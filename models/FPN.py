'''FPN in PyTorch.

See the paper "Feature Pyramid Networks for Object Detection" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class FPN(nn.Module):
    def __init__(self, configs):
        super(FPN, self).__init__()
        self.each_out_dim=configs["decoder"]["each_out_dim"] #[96,192,384,768]
        self.top_layer_dim=self.each_out_dim[-1]
        self.reduction_dim=configs["decoder"]["reduction_dim"] # [96]

        # Top layer
        self.toplayer = nn.Conv2d(self.top_layer_dim, self.reduction_dim, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(self.reduction_dim, self.reduction_dim, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(self.reduction_dim, self.reduction_dim, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(self.reduction_dim, self.reduction_dim, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(self.each_out_dim[-2], self.reduction_dim, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(self.each_out_dim[-3], self.reduction_dim, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(self.each_out_dim[-4], self.reduction_dim, kernel_size=1, stride=1, padding=0)



    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.

        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.

        Returns:
          (Variable) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]

        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self, inputs):
        # Bottom-up
        c2=inputs[0] # 96
        c3=inputs[1] # 192
        c4=inputs[2] # 384
        # 768
        p5 = self.toplayer(inputs[-1])
        # Top-down
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        return p2, p3, p4, p5



def test():
    pass


