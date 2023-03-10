import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic_ops import Conv, RCAB, Encoder, Decoder, Output, Structure, f


class RGB_NIR_Fusion(nn.Module):
    def __init__(self, n_feat=80, scale_unetfeats=48, kernel_size=3, reduction=4, bias=False):
        super(RGB_NIR_Fusion, self).__init__()

        act=nn.PReLU()

        # To extract features from RGB and NIR 
        self.shallow_feat_nir = nn.Sequential(Conv(1, n_feat, kernel_size, bias=bias), RCAB(n_feat,kernel_size, reduction, bias=bias, act=act), RCAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat_rgb = nn.Sequential(Conv(3, n_feat, kernel_size, bias=bias), RCAB(n_feat,kernel_size, reduction, bias=bias, act=act), RCAB(n_feat, kernel_size, reduction, bias=bias, act=act))

        # UNet for RGB and NIR
        self.nir_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, atten=False)
        self.nir_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, residual=True)

        self.rgb_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, atten=True)
        self.rgb_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, residual=True)

        # For Output
        self.nir_output = Output(n_feat, kernel_size=1, bias=bias, output_channel=1)
        self.rgb_output = Output(n_feat, kernel_size=1, bias=bias, output_channel=3)

        # DSEM Module
        self.structure_nir = Structure(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)
        self.structure_rgb = Structure(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)


    def forward(self, rgb, nir):

        # UNet for NIR
        feat_nir = self.shallow_feat_nir(nir)

        feat_nir_encode = self.nir_encoder(feat_nir)           
        feat_nir_decode = self.nir_decoder(feat_nir_encode)

        # Deep Structure of NIR
        nir_structure = self.structure_nir(feat_nir_encode, feat_nir_decode)

        # Reconstruction of NIR
        nir_recons = self.nir_output(feat_nir_decode[0], nir)

        # UNet for RGB
        feat_rgb = self.shallow_feat_rgb(rgb)

        feat_rgb_encode = self.rgb_encoder(feat_rgb)
        feat_rgb_decode  = self.rgb_decoder(feat_rgb_encode)

        # The denoising result of RGB
        rgb_out1 = self.rgb_output(feat_rgb_decode[0], rgb)

        feat_rgb = self.shallow_feat_rgb(rgb_out1)
        # Deep Structure of RGB
        rgb_structure = self.structure_rgb(feat_rgb_encode, feat_rgb_decode)

        # To Calculate DIP
        dips = []
        for i in range(len(rgb_structure)):
            dips.append(f(rgb_structure[i], nir_structure[i]))
        
        # The Fusion and Denoising of RGB
        feat_rgb_encode, nir_feat = self.rgb_encoder(feat_rgb, nir_structure, dips)

        feat_rgb_decode = self.rgb_decoder(feat_rgb_encode)
        rgb_recons = self.rgb_output(feat_rgb_decode[0], rgb_out1)
        
        weighted_ir = [None for _ in range(3)]
        for i in range(3):
            weighted_ir[i] = dips[i] * nir_feat[i]

        return [rgb_recons, rgb_out1, nir_recons], [rgb_structure, nir_structure], [feat_rgb_decode, nir_feat.copy(), dips, weighted_ir]


if __name__ == "__main__":
    from thop import profile
    from thop import clever_format

    model = RGB_NIR_Fusion()
    input_bgr = torch.randn(1, 3, 128, 128)
    input_ir = torch.randn(1, 1, 128, 128)
    flops, params = profile(model, inputs=(input_bgr, input_ir))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)
