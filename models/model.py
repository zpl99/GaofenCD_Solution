from .swin_transformer import SwinTransformer
from .FPN_PPM_decoder import PPM_FPN
import torch
from torch import nn
import torch.nn.functional as F

class Encoder_Decoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs=configs
        self.encoder = SwinTransformer()
        self.decoder=PPM_FPN(self.configs)
    def forward(self,x):
        encode_features=self.encoder(x)
        decode_features=self.decoder(encode_features)
        return decode_features

