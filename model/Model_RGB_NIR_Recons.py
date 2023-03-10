import torch
import torch.nn as nn

from .basic_ops_pretrain import Conv, RCAB, Encoder, Decoder, Output, Edge, Norm

class RGB_NIR_Recons(nn.Module):
    def __init__(self, n_feat=80, scale_unetfeats=48, kernel_size=3, reduction=4, bias=False):
        super(RGB_NIR_Recons, self).__init__()

        act=nn.PReLU()

        self.shallow_feat_nir = nn.Sequential(Conv(1, n_feat, kernel_size, bias=bias), RCAB(n_feat,kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat_rgb = nn.Sequential(Conv(3, n_feat, kernel_size, bias=bias), RCAB(n_feat,kernel_size, reduction, bias=bias, act=act))

        self.nir_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, atten=False)
        self.nir_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, residual=False)

        self.rgb_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, atten=False)
        self.rgb_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, residual=False)

        self.output_nir = Output(n_feat, kernel_size=1, bias=bias, output_channel=1, residual=False)
        self.output_rgb = Output(n_feat, kernel_size=1, bias=bias, output_channel=3, residual=False)

        self.nir_edge_1 = nn.Sequential(Edge(n_feat), Norm())
        self.nir_edge_2 = nn.Sequential(Edge(n_feat+scale_unetfeats), Norm())
        self.nir_edge_3 = nn.Sequential(Edge(n_feat+scale_unetfeats*2), Norm())

        self.rgb_edge_1 = nn.Sequential(Edge(n_feat), Norm())
        self.rgb_edge_2 = nn.Sequential(Edge(n_feat+scale_unetfeats), Norm())
        self.rgb_edge_3 = nn.Sequential(Edge(n_feat+scale_unetfeats*2), Norm())

    def forward(self, rgb, nir):

        nir_feat  = self.shallow_feat_nir(nir)
        nir_feat = self.nir_encoder(nir_feat)
        nir_feat = self.nir_decoder(nir_feat)
        nir_recons = self.output_nir(nir_feat[0], None)

        rgb_feat  = self.shallow_feat_rgb(rgb)
        rgb_feat = self.rgb_encoder(rgb_feat)
        rgb_feat = self.rgb_decoder(rgb_feat)
        rgb_recons = self.output_rgb(rgb_feat[0], None)

        edge_nir = [self.nir_edge_1(nir_feat[0]), self.nir_edge_2(nir_feat[1]), self.nir_edge_3(nir_feat[2])]
        edge_rgb = [self.rgb_edge_1(rgb_feat[0]), self.rgb_edge_2(rgb_feat[1]), self.rgb_edge_3(rgb_feat[2])]
        
        return [rgb_recons, nir_recons], [edge_rgb, edge_nir]

