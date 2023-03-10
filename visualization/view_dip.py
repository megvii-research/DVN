import os
import sys
sys.path.append('..')

from config import Config 
opt = Config('training.yml')

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
from utils.tools_feature import view_features
from utils.tools import make_view
from utils.dstools import getshow_bgr, getshow_ir
from utils.util import saveImgForVis

######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)


def main():
    net = load_model(opt.VIEW_DIP.MODEL, mode='Fusion')
    net.eval()
    test_dataset = DataLoaderTest(opt.VIEW_DIP.TEST_DIR, opt.VIEW_DIP.SIGMA)
    test_dataset = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1, drop_last=True, pin_memory=True)

    for data in test_dataset:
        inp_rgb, nir, gt_rgb = data
        inp_rgb, nir, gt_rgb = inp_rgb.cuda(), nir.cuda(), gt_rgb.cuda()
        with torch.no_grad():
            output, output_feats = view_features(net, inp_rgb, nir, gt_rgb)


        inp_rgb = inp_rgb[0, ...].permute(1,2,0).cpu().numpy()
        nir = nir[0, ...].permute(1,2,0).cpu().numpy()
        gt_rgb = gt_rgb[0, ...].permute(1,2,0).cpu().numpy()

        show_dct = {
                        "input": getshow_bgr(inp_rgb),
                        "label" : getshow_bgr(gt_rgb),
                        'ir': getshow_ir(nir),                
        }

        for i in range(3):
            for k, v in output_feats[i].items():
                if k == 'rgb_feature':
                    x = make_view(v, equ=True, norm=False)
                else:
                    x = make_view(v, equ=False, norm=False)
                show_dct.update({'{}_{}:'.format(k, i): x})

        break
    
    saveImgForVis(opt.VIEW_DIP.VIS_DIR, show_dct, sub=False)
    

if __name__ == "__main__":
    main()
    


