#/***************************************************************************************
#*    Modified by: M. Haider Azam
#*    Original Source
#*    Title: CAIN
#*    Author: myungsub    
#*    Availability: https://github.com/myungsub/CAIN
#*
#***************************************************************************************/
import torch.nn as nn
from common import *


class Encoder(nn.Module):
    def __init__(self, in_channels=3, depth=3):
        super(Encoder, self).__init__()

        # Shuffle pixels to expand in channel dimension
        # shuffler_list = [PixelShuffle(0.5) for i in range(depth)]
        # self.shuffler = nn.Sequential(*shuffler_list)
        self.shuffler = PixelShuffle(1 / 2**depth)

        relu = nn.LeakyReLU(0.2, True)
        
        # FF_RCAN or FF_Resblocks
        self.interpolate = Interpolation(5, 12, in_channels * (2**depth), act=relu)
        
    def forward(self, x1, x2):
        """
        Encoder: Shuffle-spread --> Feature Fusion --> Return fused features
        """
        feats1 = self.shuffler(x1)
        feats2 = self.shuffler(x2)

        feats = self.interpolate(feats1, feats2)

        return feats


class Decoder(nn.Module):
    def __init__(self, depth=3):
        super(Decoder, self).__init__()

        # shuffler_list = [PixelShuffle(2) for i in range(depth)]
        # self.shuffler = nn.Sequential(*shuffler_list)
        self.shuffler = PixelShuffle(2**depth)

    def forward(self, feats):
        out = self.shuffler(feats)
        return out


class CAIN(nn.Module):
    def __init__(self, depth=3,in_channels=20):
        super(CAIN, self).__init__()
        
        self.encoder = Encoder(in_channels=in_channels, depth=depth)
        self.decoder = Decoder(depth=depth)

    def forward(self, x1, x2):
        x1, m1 = sub_mean(x1)
        x2, m2 = sub_mean(x2)

        #if not self.training:
        #    paddingInput, paddingOutput = InOutPaddings(x1)
        #    x1 = paddingInput(x1)
        #    x2 = paddingInput(x2)

        feats = self.encoder(x1, x2)
        out = self.decoder(feats)

        #if not self.training:
        #    out = paddingOutput(out)

        mi = (m1 + m2) / 2
        out += mi

        return out, feats