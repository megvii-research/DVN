from config import Config 
opt = Config('training.yml')

import os
import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader
import random
import numpy as np
from collections import OrderedDict

from dataset import DataLoaderReal
from tqdm import tqdm
from pdb import set_trace as stx
from model.get_model import get_model, get_pretrain, load_model
from utils.tools import gather_patches_into_whole, validation_on_PSNR_and_SSIM
from utils.tools import make_view
from utils.dstools import getshow_bgr, getshow_ir
from utils.util import saveImgForVis
from tqdm import tqdm


def main():
    net = load_model(opt.TEST_REAL.MODEL, mode='Fusion')
    net.eval()
    test_real_dataset = DataLoaderReal(opt.TEST_REAL.TEST_DIR)
    test_real_dataset = DataLoader(dataset=test_real_dataset, batch_size=1, shuffle=False, drop_last=False, pin_memory=True)

    glob_dct = {}

    for i, data in tqdm(enumerate(test_real_dataset)):
        rgb, nir = data
        rgb, nir = rgb.float().cuda(), nir.float().cuda()
        label_rgb = rgb.clone()  # fake, just used as a placeholder
        with torch.no_grad():
            print(rgb.shape, nir.shape)
            fusion_img, _ = gather_patches_into_whole(net, rgb, nir, label_rgb)
        
        rgb = rgb[0, ...].permute(1,2,0).cpu().numpy()
        nir = nir[0, ...].permute(1,2,0).cpu().numpy()
        fusion_img = fusion_img[0, ...].transpose(1,2,0)

        show_dct = {
                        "input": getshow_bgr(rgb),
                        'nir': getshow_ir(nir),        
                        'fusion': getshow_bgr(fusion_img),     
        }

        glob_dct.update({i: show_dct})

    saveImgForVis(opt.TEST_REAL.VIS_DIR, glob_dct)


if __name__ == "__main__":
    main()
