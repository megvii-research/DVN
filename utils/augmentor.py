import math
import pickle as pkl
from refile import smart_open
import cv2
import numpy as np
import torch
from collections import OrderedDict


class BaseAug:
    def __init__(self, input_shape, rng=None):
        if rng is None:
            self.rng = np.random
        else:
            self.rng = rng
        self.shape = input_shape


class GetTrainArea(BaseAug):
    def do(self, img, input_shape=(256, 256)):
        if not hasattr(self, 'box'):
            h, w = self.shape
            h_i, w_i = input_shape
            x1 = int(self.rng.rand() * (w-w_i))
            y1 = int(self.rng.rand() * (h-h_i))
            x2 = x1 + w_i
            y2 = y1 + h_i
            setattr(self, 'box', (x1, y1, x2, y2))
        else:
            x1, y1, x2, y2 = self.box
        if len(img.shape) == 2:
            return img[y1:y2, x1:x2]
        elif len(img.shape) == 3:
            return img[y1:y2, x1:x2, :]


class CalibratedNoiseParam():
    def __init__(self, kfile_path):
        self.k_map = pkl.load(smart_open(kfile_path,"rb"))
    def get_param(self, again_value:int):
        assert again_value in self.k_map, "The entered Again value is not within the calibration range."
        return self.k_map[again_value]


def flip_and_route(img, rng):
    if rng==1:
        img = np.flip(img , axis=1)
    elif rng==2:
        img = np.flip(img , axis=1)
    elif rng==3:
        img = np.rot90(img,axes=(1,2))
    elif rng==4:
        img = np.rot90(img,axes=(1,2), k=2)
    elif rng==5:
        img = np.rot90(img,axes=(1,2), k=3)
    elif rng==6:
        img = np.rot90(np.flip(img , axis=1),axes=(1,2))
    elif rng==7:
        img = np.rot90(np.flip(img , axis=2),axes=(1,2))
    return img


class AddNoise():
    def __init__(self, noise_param: CalibratedNoiseParam):
        self.guassian_sigma = noise_param
        self.poisson_k = noise_param
        
    def do(self, _img, again):
        _noise_img = OrderedDict()
        for k in _img:
            _std = np.sqrt(self.guassian_sigma.get_param(again)[k])
            _k = np.array(self.poisson_k.get_param(again)[k])

            _std *= 4
            _k *= 4

            # add gaussian noise on raw data
            _gaussian_img = np.random.normal(loc=0, scale=_std, size=_img[k].shape)
            # add poisson nosie on raw data
            _poisson_img = np.random.poisson(lam=_img[k]/_k)*_k
            _noise_img[k] = _gaussian_img + _poisson_img
        return _noise_img

class AddFixNoise():
    def __init__(self, noise_param: CalibratedNoiseParam):
        self.guassian_sigma = noise_param
        self.poisson_k = noise_param

    def do(self, _img, noiselevel):
        _noise_img = OrderedDict()
        _std = noiselevel
        _k = noiselevel
        for k in _img:
            # add gaussian noise on raw data
            _gaussian_img = np.random.normal(loc=0, scale=_std, size=_img[k].shape)
            # add poisson nosie on raw data
            _poisson_img = np.random.poisson(lam=_img[k]/_k)*_k
            _noise_img[k] = _gaussian_img + _poisson_img
        return _noise_img


def brightness_aug(img, img_mean, base_brightness, low, high, rng):
    
    _low, _high = np.log(base_brightness + low), np.log(base_brightness + high)
    target_brightness = np.exp(rng.uniform(_low, _high)) - base_brightness  
    darken_ratio = target_brightness / img_mean

    if darken_ratio > 0.99:
        return img
    else:
        for i in img:
            img[i] *= darken_ratio
    return img 