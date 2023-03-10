import os
import sys
sys.path.append('..')

from config import Config 
opt = Config('training.yml')

import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader
import random
import numpy as np
from collections import OrderedDict

from dataset import DataLoaderTest
from tqdm import tqdm
from pdb import set_trace as stx
from model.get_model import get_model, get_pretrain, load_model
from utils.tools import gather_patches_into_whole, validation_on_PSNR_and_SSIM
from utils.tools import make_view
from utils.dstools import getshow_bgr, getshow_ir
from utils.util import saveImgForVis
from tqdm import tqdm
######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)


def main():
    net = load_model(opt.TEST.MODEL, mode='Fusion')
    net.eval()
    test_dataset = DataLoaderTest(opt.TEST.TEST_DIR, opt.TEST.SIGMA)
    test_dataset = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1, drop_last=False, pin_memory=True)

    metrics = {'psnr_i': [], 'psnr_o': [], 'ssim_i': [], 'ssim_o': []}
    glob_dct = {}

    for i, data in tqdm(enumerate(test_dataset)):
        inp_rgb, nir, gt_rgb = data
        inp_rgb, nir, gt_rgb = inp_rgb.cuda(), nir.cuda(), gt_rgb.cuda()
        with torch.no_grad():
            output, output1 = gather_patches_into_whole(net, inp_rgb, nir, gt_rgb)
            i_psnr, i_ssim, o_psnr, o_ssim = validation_on_PSNR_and_SSIM(net, inp_rgb[:, :, 259:1263, 451: 2239], nir[:, :, 259:1263, 451: 2239], gt_rgb[:, :, 259:1263, 451: 2239])
            metrics['psnr_i'].append(i_psnr)
            metrics['ssim_i'].append(i_ssim)
            metrics['psnr_o'].append(o_psnr)
            metrics['ssim_o'].append(o_ssim)


        inp_rgb = inp_rgb[0, ...].permute(1,2,0).cpu().numpy()
        nir = nir[0, ...].permute(1,2,0).cpu().numpy()
        gt_rgb = gt_rgb[0, ...].permute(1,2,0).cpu().numpy()
        output = output[0, ...].transpose(1,2,0)
        output1 = output1[0, ...].transpose(1,2,0)

        show_dct = {
                        "input": getshow_bgr(inp_rgb),
                        "label" : getshow_bgr(gt_rgb),
                        'ir': getshow_ir(nir),        
                        'output': getshow_bgr(output),
                        'output1': getshow_bgr(output1)        
        }

        glob_dct.update({i: show_dct})
    
    saveImgForVis(opt.TEST.VIS_DIR, glob_dct)

    print('The calculate metrices is:')
    for k, v in metrics.items():
        print("{}: {}".format(k, np.array(v).mean()))
    

if __name__ == "__main__":
    main()
    


