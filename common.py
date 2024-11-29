#/***************************************************************************************
#*    Modified by: M. Haider Azam
#*    Original Source
#*    Title: CAIN
#*    Author: myungsub
#*    Availability: https://github.com/myungsub/CAIN
#*
#***************************************************************************************/
import torch
import torch.nn as nn

def sub_mean(x):
    mean = x.mean(2, keepdim=True)
    return x-mean, mean

def InOutPaddings(x):
    length = x.size(2)
    padding_length = 0
    if length != ((length >> 7) << 7):
        padding_length = (((length >> 7) + 1) << 7) - length

    paddingInput = nn.ReflectionPad1d(padding=[padding_length // 2, padding_length - padding_length // 2])
    paddingOutput = nn.ReflectionPad1d(padding=[0 - padding_length // 2, padding_length // 2 - padding_length])
    return paddingInput, paddingOutput


class ConvNorm(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size, stride=1, norm=False):
        super(ConvNorm, self).__init__()

        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad1d(reflection_padding)
        self.conv = nn.Conv1d(in_feat, out_feat, stride=stride, kernel_size=kernel_size, bias=True)

        self.norm = norm
        if norm == 'IN':
            self.norm = nn.InstanceNorm1d(out_feat, track_running_stats=True)
        elif norm == 'BN':
            self.norm = nn.BatchNorm1d(out_feat)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv(out)
        if self.norm:
            out = self.norm(out)
        return out


class UpConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, mode='transpose', norm=False):
        super(UpConvNorm, self).__init__()

        if mode == 'transpose':
            self.upconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        elif mode == 'shuffle':
            self.upconv = nn.Sequential(
                ConvNorm(in_channels, 4*out_channels, kernel_size=3, stride=1, norm=norm),
                PixelShuffle(2))
        else:
            # out_channels is always going to be the same as in_channels
            self.upconv = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False),
                ConvNorm(in_channels, out_channels, kernel_size=1, stride=1, norm=norm))
    
    def forward(self, x):
        out = self.upconv(x)
        return out



class meanShift(nn.Module):
    def __init__(self, rgbRange, rgbMean, sign, nChannel=3):
        super(meanShift, self).__init__()
        if nChannel == 1:
            l = rgbMean[0] * rgbRange * float(sign)

            self.shifter = nn.Conv1d(1, 1, kernel_size=1, stride=1, padding=0)
            self.shifter.weight.data = torch.eye(1).view(1, 1, 1, 1)
            self.shifter.bias.data = torch.Tensor([l])
        elif nChannel == 3:  
            r = rgbMean[0] * rgbRange * float(sign)
            g = rgbMean[1] * rgbRange * float(sign)
            b = rgbMean[2] * rgbRange * float(sign)

            self.shifter = nn.Conv1d(3, 3, kernel_size=1, stride=1, padding=0)
            self.shifter.weight.data = torch.eye(3).view(3, 3, 1, 1)
            self.shifter.bias.data = torch.Tensor([r, g, b])
        else:
            r = rgbMean[0] * rgbRange * float(sign)
            g = rgbMean[1] * rgbRange * float(sign)
            b = rgbMean[2] * rgbRange * float(sign)
            self.shifter = nn.Conv1d(6, 6, kernel_size=1, stride=1, padding=0)
            self.shifter.weight.data = torch.eye(6).view(6, 6, 1, 1)
            self.shifter.bias.data = torch.Tensor([r, g, b, r, g, b])

        # Freeze the meanShift layer
        for params in self.shifter.parameters():
            params.requires_grad = False

    def forward(self, x):
        x = self.shifter(x)

        return x


""" CONV - (BN) - RELU - CONV - (BN) """
class ResBlock(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size=3, reduction=False, bias=True, # 'reduction' is just for placeholder
                 norm=False, act=nn.ReLU(True), downscale=False):
        super(ResBlock, self).__init__()

        self.body = nn.Sequential(
            ConvNorm(in_feat, out_feat, kernel_size=kernel_size, stride=2 if downscale else 1),
            act,
            ConvNorm(out_feat, out_feat, kernel_size=kernel_size, stride=1)
        )
        
        self.downscale = None
        if downscale:
            self.downscale = nn.Conv1d(in_feat, out_feat, kernel_size=1, stride=2)

    def forward(self, x):
        res = x
        out = self.body(x)
        if self.downscale is not None:
            res = self.downscale(res)
        out += res

        return out 


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv1d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv1d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y, y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size, reduction, bias=True,
            norm=False, act=nn.ReLU(True), downscale=False, return_ca=False):
        super(RCAB, self).__init__()

        self.body = nn.Sequential(
            ConvNorm(in_feat, out_feat, kernel_size, stride=2 if downscale else 1, norm=norm),
            act,
            ConvNorm(out_feat, out_feat, kernel_size, stride=1, norm=norm),
            CALayer(out_feat, reduction)
        )
        self.downscale = downscale
        if downscale:
            self.downConv = nn.Conv1d(in_feat, out_feat, kernel_size=3, stride=2, padding=1)
        self.return_ca = return_ca

    def forward(self, x):
        res = x
        out, ca = self.body(x)
        if self.downscale:
            res = self.downConv(res)
        out += res

        if self.return_ca:
            return out, ca
        else:
            return out


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, Block, n_resblocks, n_feat, kernel_size, reduction, act, norm=False):
        super(ResidualGroup, self).__init__()

        modules_body = [Block(n_feat, n_feat, kernel_size, reduction, bias=True, norm=norm, act=act)
            for _ in range(n_resblocks)]
        modules_body.append(ConvNorm(n_feat, n_feat, kernel_size, stride=1, norm=norm))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


def pixel_shuffle(input, scale_factor):
    #batch_size, channels, in_height, in_width = input.size()
    batch_size, channels, in_length = input.size()
    out_channels = int(int(channels / scale_factor))
    #out_height = int(in_height * scale_factor)
    #out_width = int(in_width * scale_factor)
    out_length=int(in_length * scale_factor)

    if scale_factor >= 1:
        #input_view = input.contiguous().view(batch_size, out_channels, scale_factor, scale_factor, in_height, in_width)
        input_view = input.contiguous().view(batch_size, out_channels, scale_factor, in_length)
        shuffle_out = input_view.permute(0, 1, 3, 2).contiguous()
    else:
        block_size = int(1 / scale_factor)
        #input_view = input.contiguous().view(batch_size, channels, out_height, block_size, out_width, block_size)
        input_view = input.contiguous().view(batch_size, channels, out_length, block_size)
        shuffle_out = input_view.permute(0, 1, 3, 2).contiguous()

    #return shuffle_out.view(batch_size, out_channels, out_height, out_width)
    return shuffle_out.view(batch_size, out_channels, out_length)


class PixelShuffle(nn.Module):
    def __init__(self, scale_factor):
        super(PixelShuffle, self).__init__()
        self.scale_factor = scale_factor
    
    def forward(self, x):
        return pixel_shuffle(x, self.scale_factor)
    def extra_repr(self):
        return 'scale_factor={}'.format(self.scale_factor)


def conv(in_channels, out_channels, kernel_size, 
         stride=1, bias=True, groups=1):
    return nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        padding=kernel_size//2,
        stride=1,
        bias=bias,
        groups=groups)


def conv1x1(in_channels, out_channels, stride=1, bias=True, groups=1):
    return nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        bias=bias,
        groups=groups)

def conv3x3(in_channels, out_channels, stride=1, 
            padding=1, bias=True, groups=1):    
    return nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)

def conv5x5(in_channels, out_channels, stride=1, 
            padding=2, bias=True, groups=1):    
    return nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size=5,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)

def conv7x7(in_channels, out_channels, stride=1, 
            padding=3, bias=True, groups=1):    
    return nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size=7,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)

def upconv2x2(in_channels, out_channels, mode='shuffle'):
    if mode == 'transpose':
        return nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1)
    elif mode == 'shuffle':
        return nn.Sequential(
            conv3x3(in_channels, 4*out_channels),
            PixelShuffle(2))
    else:
        # out_channels is always going to be the same as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False),
            conv1x1(in_channels, out_channels))



class Interpolation(nn.Module):
    def __init__(self, n_resgroups, n_resblocks, n_feats, 
                 reduction=16, act=nn.LeakyReLU(0.2, True), norm=False):
        super(Interpolation, self).__init__()
        self.n_feats=n_feats
        # define modules: head, body, tail
        self.headConv = conv3x3(n_feats * 2, n_feats)

        modules_body = [
            ResidualGroup(
                RCAB,
                n_resblocks=n_resblocks,
                n_feat=n_feats,
                kernel_size=3,
                reduction=reduction, 
                act=act, 
                norm=norm)
            for _ in range(n_resgroups)]
        self.body = nn.Sequential(*modules_body)

        self.tailConv = conv3x3(n_feats, n_feats)

    def forward(self, x0, x1):
        # Build input tensor
        x = torch.cat([x0, x1], dim=1)

        x = self.headConv(x)

        res = self.body(x)
        res += x

        out = self.tailConv(res)
        return out


class Interpolation_res(nn.Module):
    def __init__(self, n_resgroups, n_resblocks, n_feats,
                 act=nn.LeakyReLU(0.2, True), norm=False):
        super(Interpolation_res, self).__init__()

        # define modules: head, body, tail (reduces concatenated inputs to n_feat)
        self.headConv = conv3x3(n_feats * 2, n_feats)

        modules_body = [ResidualGroup(ResBlock, n_resblocks=n_resblocks, n_feat=n_feats, kernel_size=3,
                            reduction=0, act=act, norm=norm)
                        for _ in range(n_resgroups)]
        self.body = nn.Sequential(*modules_body)

        self.tailConv = conv3x3(n_feats, n_feats)

    def forward(self, x0, x1):
        # Build input tensor
        x = torch.cat([x0, x1], dim=1)
        x = self.headConv(x)

        res = x
        for m in self.body:
            res = m(res)
        res += x

        x = self.tailConv(res)

        return x