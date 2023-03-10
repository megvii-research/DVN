import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def f(x, y):
    return 1 / 2 * (1 - x) * (1 - y) + x * y


Sobel = np.array([[-1,-2,-1],
                  [ 0, 0, 0],
                  [ 1, 2, 1]])
Robert = np.array([[0, 0],
                  [-1, 1]])
Sobel = torch.Tensor(Sobel)
Robert = torch.Tensor(Robert)


# binarize the deep structures
class Norm(nn.Module):
    def __init__(self):
        super(Norm, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        y = self.avg(x)
        x = torch.sign(x - y)
        out = (x + 1) / 2
        return out


# use Sobel operator to calculate the deep structures
class Edge(nn.Module):
    def __init__(self, channel, kernel='sobel'):
        super(Edge, self).__init__()
        self.channel = channel
        self.kernel = kernel
        if self.kernel == 'sobel':
            self.kernel_x = Sobel.repeat(channel, 1, 1, 1)
        elif self.kernel == 'robert':
            self.kernel_x = Robert.repeat(channel, 1, 1, 1)
        else:
            raise Exception('the kernel must br sobel or robert')

        self.kernel_y = self.kernel_x.permute(0, 1, 3, 2)

        self.kernel_x = nn.Parameter(self.kernel_x, requires_grad=False)
        self.kernel_y = nn.Parameter(self.kernel_y, requires_grad=False)

    def forward(self, current):
        if self.kernel == 'robert':
            current = F.pad(current, (0,1,0,1), mode='reflect')
        elif self.kernel == 'sobel':
            current = F.pad(current, (1,1,1,1), mode='reflect')

        gradient_x = torch.abs(F.conv2d(current, weight=self.kernel_x, groups=self.channel, padding=0))
        gradient_y = torch.abs(F.conv2d(current, weight=self.kernel_y, groups=self.channel, padding=0))
        out = gradient_x + gradient_y
        return out


def Conv(in_channels, out_channels, kernel_size, stride = 1, bias=False):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), stride = stride, bias=bias)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(RCAB, self).__init__()
        feats_cal = []
        feats_cal.append(Conv(n_feat, n_feat, kernel_size, bias=bias))
        feats_cal.append(act)
        feats_cal.append(Conv(n_feat, n_feat, kernel_size, bias=bias))

        self.SE = SELayer(n_feat, reduction, bias=bias)
        self.feats_cal = nn.Sequential(*feats_cal)

    def forward(self, x):
        feats = self.feats_cal(x)
        feats = self.SE(feats)
        feats += x
        return feats


class Output(nn.Module):
    def __init__(self, n_feat, kernel_size, bias, output_channel=3, residual=True):
        super(Output, self).__init__()
        self.conv = Conv(n_feat, output_channel, kernel_size, bias=bias)
        self.residual = residual

    def forward(self, x, x_img):
        x = self.conv(x)
        if self.residual:
            x += x_img
        return x


class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, atten):
        super(Encoder, self).__init__()
        self.atten = atten

        self.encoder_level1 = RCAB(n_feat,                     kernel_size, reduction, bias=bias, act=act)
        self.encoder_level2 = RCAB(n_feat+scale_unetfeats,     kernel_size, reduction, bias=bias, act=act)
        self.encoder_level3 = RCAB(n_feat+(scale_unetfeats*2), kernel_size, reduction, bias=bias, act=act)

        self.down12  = DownSample(n_feat, n_feat+scale_unetfeats)
        self.down23  = DownSample(n_feat+scale_unetfeats, n_feat+(scale_unetfeats*2))

        if self.atten:  # feature attention
            self.atten_conv1 = Conv(n_feat, n_feat, 1, bias=bias)
            self.atten_conv2 = Conv(n_feat+scale_unetfeats, n_feat+scale_unetfeats, 1, bias=bias)
            self.atten_conv3 = Conv(n_feat+(scale_unetfeats*2), n_feat+(scale_unetfeats*2), 1, bias=bias)

    def forward(self, x, encoder_outs=None, mask=None):
        if not self.atten or mask is None:
            enc1 = self.encoder_level1(x)
            x = self.down12(enc1)
            enc2 = self.encoder_level2(x)
            x = self.down23(enc2)
            enc3 = self.encoder_level3(x)

            return [enc1, enc2, enc3]
        else:
            assert encoder_outs is not None
            assert mask is not None


            enc1 = self.encoder_level1(x)
            enc1_fuse_nir = enc1 + self.atten_conv1(mask[0] * encoder_outs[0])
            x = self.down12(enc1_fuse_nir)
            enc2 = self.encoder_level2(x)
            enc2_fuse_nir = enc2 + self.atten_conv2(mask[1] * encoder_outs[1])
            x = self.down23(enc2_fuse_nir)
            enc3 = self.encoder_level3(x)
            enc3_fuse_nir = enc3 + self.atten_conv3(mask[2] * encoder_outs[2])

            return [enc1_fuse_nir, enc2_fuse_nir, enc3_fuse_nir], encoder_outs
        


class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, residual=True):
        super(Decoder, self).__init__()

        self.residual = residual

        self.decoder_level1 = RCAB(n_feat,                     kernel_size, reduction, bias=bias, act=act)
        self.decoder_level2 = RCAB(n_feat+scale_unetfeats,     kernel_size, reduction, bias=bias, act=act)
        self.decoder_level3 = RCAB(n_feat+(scale_unetfeats*2), kernel_size, reduction, bias=bias, act=act)

        self.skip_conv_1 = Conv(n_feat, n_feat, kernel_size, bias=bias)
        self.skip_conv_2 = Conv(n_feat+scale_unetfeats, n_feat+scale_unetfeats, kernel_size, bias=bias)

        self.up21  = UpSample(n_feat+scale_unetfeats, n_feat)
        self.up32  = UpSample(n_feat+(scale_unetfeats*2), n_feat+scale_unetfeats)

    def forward(self, outs):
        enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3(enc3)

        x = self.up32(dec3)
        if self.residual:
            x += self.skip_conv_2(enc2)
        dec2 = self.decoder_level2(x)

        x = self.up21(dec2)
        if self.residual:
            x += self.skip_conv_1(enc1)
        dec1 = self.decoder_level1(x)

        return [dec1, dec2, dec3]

 
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channel):
        super(DownSample, self).__init__()
        self.conv = Conv(in_channels, out_channel, 1, stride=1, bias=False)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x = self.conv(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channel):
        super(UpSample, self).__init__()
        self.conv = Conv(in_channels, out_channel, 1, stride=1, bias=False)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv(x)
        return x

class CrossStageFF(nn.Module):
    def __init__(self, n_feat, scale_unetfeats, kernel_size, reduction, act, bias):
        super(CrossStageFF, self).__init__()
        self.conv_1 = RCAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.conv_2 = RCAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.downsample = DownSample(n_feat-scale_unetfeats, n_feat)

    def forward(self, x, y):
        if y is None:
            return self.conv_2(self.conv_1(x))
        x = self.conv_1(x)
        x += self.downsample(y)
        res = self.conv_2(x)
        return res

class CrossStageF(nn.Module):
    def __init__(self, n_feat, scale_unetfeats, kernel_size, reduction, act, bias):
        super(CrossStageF, self).__init__()
        self.conv_1 = Conv(n_feat, n_feat, 1, stride=1, bias=bias)
        self.conv_2 = Conv(n_feat, n_feat, 1, stride=1, bias=bias)
        self.downsample = DownSample(n_feat-scale_unetfeats, n_feat)

    def forward(self, x, y):
        if y is None:
            return self.conv_2(self.conv_1(x))
        x = self.conv_1(x)
        x += self.downsample(y)
        res = self.conv_2(x)
        return res


class Structure(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats):
        super(Structure, self).__init__()

        act = nn.PReLU()

        self.ff_enc1 = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias),
            act,)
        self.ff_enc2 = nn.Sequential(
            nn.Conv2d(n_feat+scale_unetfeats, n_feat+scale_unetfeats, kernel_size=1, bias=bias),
            act,)
        self.ff_enc3 = nn.Sequential(
            nn.Conv2d(n_feat+(scale_unetfeats*2), n_feat+(scale_unetfeats*2), kernel_size=1, bias=bias),
            act,)

        self.ff_dec1 = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias),
            act,)
        self.ff_dec2 = nn.Sequential(
            nn.Conv2d(n_feat+scale_unetfeats, n_feat+scale_unetfeats, kernel_size=1, bias=bias),
            act,)
        self.ff_dec3 = nn.Sequential(
            nn.Conv2d(n_feat+(scale_unetfeats*2), n_feat+(scale_unetfeats*2), kernel_size=1, bias=bias),
            act,)

        self.csff_1 = CrossStageFF(n_feat, scale_unetfeats, kernel_size, reduction, act, bias)
        self.csff_2 = CrossStageF(n_feat + scale_unetfeats, scale_unetfeats, kernel_size, reduction, act, bias)
        self.csff_3 = CrossStageF(n_feat + scale_unetfeats * 2, scale_unetfeats, kernel_size, reduction, act, bias)

        self.sigmoid_1 = nn.Sequential(
            Conv(n_feat, n_feat, kernel_size, bias=bias), 
            nn.Sigmoid())
        self.sigmoid_2 = nn.Sequential(
            Conv(n_feat + scale_unetfeats, n_feat + scale_unetfeats, kernel_size, bias=bias), 
            nn.Sigmoid())
        self.sigmoid_3 = nn.Sequential(
            Conv(n_feat + scale_unetfeats*2, n_feat + scale_unetfeats*2, kernel_size, bias=bias), 
            nn.Sigmoid())

    def forward(self, encfeats, decfeats):
        enc1, enc2, enc3 = encfeats
        dec1, dec2, dec3 = decfeats

        feat1 = self.ff_enc1(enc1) + self.ff_dec1(dec1)
        stru1 = self.csff_1(feat1, None)
        feat2 = self.ff_enc2(enc2) + self.ff_dec2(dec2)
        stru2 = self.csff_2(feat2, stru1)
        feat3 = self.ff_enc3(enc3) + self.ff_dec3(dec3)
        stru3 = self.csff_3(feat3, stru2)

        outedge1 = self.sigmoid_1(stru1)
        outedge2 = self.sigmoid_2(stru2)
        outedge3 = self.sigmoid_3(stru3)


        return [outedge1, outedge2, outedge3]



