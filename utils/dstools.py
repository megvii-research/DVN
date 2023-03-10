from enum import Enum
from collections import OrderedDict
import numpy as np
import cv2

class OV4686BayerPatternV2_RGBIR(Enum):
    B1 = 0; G21 = 1; R2 = 2; G23 = 3
    G11 = 4; IR1 = 5; G13 = 6; IR3 = 7
    R1 = 8; G22 = 9; B2 = 10; G24 = 11
    G12 = 12; IR2 = 13; G14 = 14; IR4 = 15


class OV4686BayerPatternV2_RGB(Enum):
    B1 = 0; G21 = 1; R2 = 2; G23 = 3
    G11 = 4; R3 = 5; G13 = 6; B3 = 7
    R1 = 8; G22 = 9; B2 = 10; G24 = 11
    G12 = 12; B4 = 13; G14 = 14; R4 = 15


class OV4686BayerPatternV2_IR(Enum):
    B1_IR = 0; G21_IR = 1; R2_IR = 2; G23_IR = 3
    G11_IR = 4; IR1_IR = 5; G13_IR = 6; IR3_IR = 7
    R1_IR = 8; G22_IR = 9; B2_IR = 10; G24_IR = 11
    G12_IR = 12; IR2_IR = 13; G14_IR = 14; IR4_IR = 15


class RGBIR_with_channels():
    def __init__(self):
        self.pattern = OV4686BayerPatternV2_RGBIR

    def do(self, img):
        rgbir_with_channels = OrderedDict()
        for k in self.pattern:
            v = k.value
            k_ = k.name.split('_')[0]
            rgbir_with_channels[k_] = img[..., v]
        return rgbir_with_channels
        
def show_bgr(outputs):
    out_rgb = outputs
    show_rgb = np.zeros((out_rgb.shape[0] * 4, out_rgb.shape[1] * 4))
    # B
    show_rgb[0::4, 0::4] = out_rgb[:, :, OV4686BayerPatternV2_RGB.B1.value] * 2
    show_rgb[2::4, 0::4] = out_rgb[:, :, OV4686BayerPatternV2_RGB.B4.value] * 2
    show_rgb[0::4, 2::4] = out_rgb[:, :, OV4686BayerPatternV2_RGB.B3.value] * 2
    show_rgb[2::4, 2::4] = out_rgb[:, :, OV4686BayerPatternV2_RGB.B2.value] * 2
    # R
    show_rgb[1::4, 1::4] = out_rgb[:, :, OV4686BayerPatternV2_RGB.R3.value] * 1.5
    show_rgb[3::4, 1::4] = out_rgb[:, :, OV4686BayerPatternV2_RGB.R1.value] * 1.5
    show_rgb[1::4, 3::4] = out_rgb[:, :, OV4686BayerPatternV2_RGB.R2.value] * 1.5
    show_rgb[3::4, 3::4] = out_rgb[:, :, OV4686BayerPatternV2_RGB.R4.value] * 1.5
    # G1
    show_rgb[1::4, 0::4] = out_rgb[:, :, OV4686BayerPatternV2_RGB.G11.value]
    show_rgb[3::4, 0::4] = out_rgb[:, :, OV4686BayerPatternV2_RGB.G12.value]
    show_rgb[1::4, 2::4] = out_rgb[:, :, OV4686BayerPatternV2_RGB.G13.value]
    show_rgb[3::4, 2::4] = out_rgb[:, :, OV4686BayerPatternV2_RGB.G14.value]
    # G2
    show_rgb[0::4, 1::4] = out_rgb[:, :, OV4686BayerPatternV2_RGB.G21.value]
    show_rgb[2::4, 1::4] = out_rgb[:, :, OV4686BayerPatternV2_RGB.G22.value]
    show_rgb[0::4, 3::4] = out_rgb[:, :, OV4686BayerPatternV2_RGB.G23.value]
    show_rgb[2::4, 3::4] = out_rgb[:, :, OV4686BayerPatternV2_RGB.G24.value]

    show_rgb = show_rgb.clip(0, 1023)
    show_rgb = show_rgb / 1023 * (2 ** 16 -1)

    show_rgb = show_rgb.astype('uint16')
    show_rgb = cv2.cvtColor(show_rgb, cv2.COLOR_BAYER_RG2BGR)

    show_rgb = show_rgb / (2 ** 16 -1)

    show_rgb = np.power(show_rgb, 0.45)
    show_rgb = np.nan_to_num(show_rgb)
    
    return show_rgb

def show_ir(outputs):
    out_ir = outputs
    show_ir = np.zeros((out_ir.shape[0] * 4, out_ir.shape[1] * 4))
    show_ir[0::4, 0::4] = out_ir[:, :, OV4686BayerPatternV2_IR.B1_IR.value]  # B1
    show_ir[2::4, 2::4] = out_ir[:, :, OV4686BayerPatternV2_IR.B2_IR.value]  # B2
    show_ir[2::4, 0::4] = out_ir[:, :, OV4686BayerPatternV2_IR.R1_IR.value]  # R1
    show_ir[0::4, 2::4] = out_ir[:, :, OV4686BayerPatternV2_IR.R2_IR.value]  # R2
    show_ir[1::4, 0::4] = out_ir[:, :, OV4686BayerPatternV2_IR.G11_IR.value]  # G11
    show_ir[3::4, 0::4] = out_ir[:, :, OV4686BayerPatternV2_IR.G12_IR.value]  # G12
    show_ir[1::4, 2::4] = out_ir[:, :, OV4686BayerPatternV2_IR.G13_IR.value]  # G13
    show_ir[3::4, 2::4] = out_ir[:, :, OV4686BayerPatternV2_IR.G14_IR.value]  # G14
    show_ir[0::4, 1::4] = out_ir[:, :, OV4686BayerPatternV2_IR.G21_IR.value]  # G21
    show_ir[2::4, 1::4] = out_ir[:, :, OV4686BayerPatternV2_IR.G22_IR.value]  # G22
    show_ir[0::4, 3::4] = out_ir[:, :, OV4686BayerPatternV2_IR.G23_IR.value]  # G23
    show_ir[2::4, 3::4] = out_ir[:, :, OV4686BayerPatternV2_IR.G24_IR.value]  # G24
    show_ir[1::4, 1::4] = out_ir[:, :, OV4686BayerPatternV2_IR.IR1_IR.value]  # IR1
    show_ir[3::4, 1::4] = out_ir[:, :, OV4686BayerPatternV2_IR.IR2_IR.value]  # IR2
    show_ir[1::4, 3::4] = out_ir[:, :, OV4686BayerPatternV2_IR.IR3_IR.value]  # IR3
    show_ir[3::4, 3::4] = out_ir[:, :, OV4686BayerPatternV2_IR.IR4_IR.value]  # IR4

    show_ir = show_ir.clip(0, 1023)
    show_ir = show_ir[..., np.newaxis] / 1023
    show_ir = show_ir ** 0.45
    show_ir = np.nan_to_num(show_ir)

    return show_ir


def _interp_B(b1, b2):
    assert b1.shape == b2.shape
    x = np.zeros((2 + b1.shape[0] * 2, 2 + b1.shape[1] * 2))
    h, w = x.shape

    x[1:(h-1):2, 1:(w-1):2] = b1
    x[2:(h-1):2, 2:(w-1):2] = b2

    # padding
    x[0, :] = x[2, :]
    x[-1, :] = x[-3, :]
    x[:, 0] = x[:, 2]   
    x[:, -1] = x[:, -3]

    # interpolate
    x[1:(h-1):2, 2:(w-1):2] = (
        x[1:(h-1):2, 1:(w-2):2]
        + x[1:(h-1):2, 3:w:2]
        + x[0:(h-2):2, 2:(w-1):2]
        + x[2:h:2, 2:(w-1):2]
    ) / 4
    x[2:(h-1):2, 1:(w-1):2] = (
        x[2:(h-1):2, 0:(w-2):2]
        + x[2:(h-1):2, 2:w:2]
        + x[1:(h-2):2, 1:(w-1):2]
        + x[3:h:2, 1:(w-1):2]
    ) / 4
    x = x.astype('float64')

    return x[1:(h-1):2, 2:(w-1):2], x[2:(h-1):2, 1:(w-1):2]


def _interp_R(r1, r2):
    assert r1.shape == r2.shape
    x = np.zeros((2 + r1.shape[0] * 2, 2 + r1.shape[1] * 2))
    h, w = x.shape

    x[2:(h-1):2, 1:(w-1):2] = r1
    x[1:(h-1):2, 2:(w-1):2] = r2

    # padding
    x[0, :] = x[2, :]
    x[-1, :] = x[-3, :]
    x[:, 0] = x[:, 2]
    x[:, -1] = x[:, -3]

    # interpolate
    x[1:(h-1):2, 1:(w-1):2] = (
        x[1:(h-1):2, 0:(w-2):2]
        + x[1:(h-1):2, 2:w:2]
        + x[0:(h-2):2, 1:(w-1):2]
        + x[2:h:2, 1:(w-1):2]
    ) / 4
    x[2:(h-1):2, 2:(w-1):2] = (
        x[2:(h-1):2, 1:(w-2):2]
        + x[2:(h-1):2, 3:w:2]
        + x[1:(h-2):2, 2:(w-1):2]
        + x[3:h:2, 2:(w-1):2]
    ) / 4
    x = x.astype('float64')

    return x[1:(h-1):2, 1:(w-1):2], x[2:(h-1):2, 2:(w-1):2]


def get_gt_rgb_with_channels(raw_rgb_with_channels):
    gt_rgb_with_channels = raw_rgb_with_channels.copy()
    gt_rgb_with_channels['B3'], gt_rgb_with_channels['B4'] = _interp_B(
        raw_rgb_with_channels['B1'], raw_rgb_with_channels['B2']
    )
    gt_rgb_with_channels['R3'], gt_rgb_with_channels['R4'] = _interp_R(
        raw_rgb_with_channels['R1'], raw_rgb_with_channels['R2']
    )
    for _tag in ["IR1","IR2","IR3","IR4"]:
        gt_rgb_with_channels.pop(_tag)
    return gt_rgb_with_channels


def get_gt_ir_with_channels(raw_ir_with_channels):
    gt_ir_with_channels = {}
    for i in OV4686BayerPatternV2_IR:
        gt_key = i.name
        raw_key = i.name.split('_')[0]
        gt_ir_with_channels[gt_key] = raw_ir_with_channels[raw_key]
    return gt_ir_with_channels


class RawImageWithChannels(OrderedDict):
    def __init__(self, inp_img_channel_map, bayer_pattern:Enum):
        cfa_orders = [enum_value.name for enum_value in list(bayer_pattern)]
        assert len(cfa_orders) == len(inp_img_channel_map), "The input data does not match the BayerPattern."
        imgs, channels = [], []
        for enum_name in cfa_orders:
            imgs.append(inp_img_channel_map[enum_name])
            channels.append(enum_name)
        OrderedDict.__init__(self, zip(channels, imgs))


def getshow_bgr(img):
    show_rgb = img / np.average(img) * 100
    show_rgb = (np.clip(show_rgb, 0, 255)).astype('uint8')
    return show_rgb

def getshow_ir(img):
    show_ir = img / np.average(img) * 100
    show_ir = np.clip(show_ir, 0, 255).astype('uint8')
    return show_ir

