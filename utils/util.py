import torch
import os
from collections import OrderedDict
import cv2

def load_start_epoch(weights):
    checkpoint = torch.load(weights)
    epoch = checkpoint["epoch"]
    return epoch

def load_optim(optimizer, weights):
    checkpoint = torch.load(weights)
    optimizer.load_state_dict(checkpoint['optimizer'])
    # for p in optimizer.param_groups: lr = p['lr']
    # return lr

def saveImgForVis(DIR_path, dic: dict, sub=True):
    if not os.path.exists(DIR_path):
        os.makedirs(DIR_path)

    if sub:
        for k, v in dic.items():
            imgPath = os.path.join(DIR_path, str(k))
            if not os.path.exists(imgPath):
                os.makedirs(imgPath)
            for k_sub, v_sub in v.items():
                cv2.imwrite(os.path.join(imgPath, str(k_sub)+'.png'), v_sub)
    else:
        for k, v in dic.items():
            cv2.imwrite(os.path.join(DIR_path, str(k)+'.png'), v)

