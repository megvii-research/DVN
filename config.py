#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, List
from yacs.config import CfgNode as CN


class Config(object):
    def __init__(self, config_yaml: str, config_override: List[Any] = []):

        self._C = CN()
        self._C.GPU = [0]
        self._C.VERBOSE = False

        self._C.MODEL = CN()
        self._C.MODEL.MODE = 'global'
        self._C.MODEL.SESSION = 'ps128_bs1'

        self._C.OPTIM = CN()
        self._C.OPTIM.BATCH_SIZE = 16
        self._C.OPTIM.LR_INITIAL = 2e-4
        self._C.OPTIM.LR_MIN = 1e-6

        self._C.Recons = CN()
        self._C.Recons.RESUME = False
        self._C.Recons.RESUME_PATH = './checkpoints/Recons/models/model_epoch_1.pth'
        self._C.Recons.NUM_EPOCHS = 4
        self._C.Recons.TRAIN_DIR = './Dataset/DVD_train'
        self._C.Recons.SAVE_DIR = './checkpoints'
        self._C.Recons.TRAIN_PS = 128

        self._C.Fusion = CN()
        self._C.Fusion.RESUME = False
        self._C.Fusion.RESUME_PATH = './checkpoints/Fusion/models/model_epoch_1.pth'
        self._C.Fusion.NUM_EPOCHS = 80
        self._C.Fusion.TRAIN_DIR = './Dataset/DVD_train'
        self._C.Fusion.SAVE_DIR = './checkpoints'
        self._C.Fusion.TRAIN_PS = 128

        self._C.Edge_Detect = CN()
        self._C.Edge_Detect.TEST_DIR = './Dataset/DVD_test'
        self._C.Edge_Detect.MODEL = './checkpoints/Recons/models/model_epoch_4.pth'
        self._C.Edge_Detect.VIS_DIR = './results/visEdges/'

        self._C.TEST = CN()
        self._C.TEST.TEST_DIR = './Dataset/DVD_test'
        self._C.TEST.MODEL = './checkpoints/Fusion/models/model_epoch_80.pth'
        self._C.TEST.SIGMA = 4
        self._C.TEST.VIS_DIR = './visResults/'

        self._C.TEST_REAL = CN()
        self._C.TEST_REAL.TEST_DIR = './Dataset/DVD_real'
        self._C.TEST_REAL.MODEL = './checkpoints/Fusion/models/model_epoch_80.pth'
        self._C.TEST_REAL.VIS_DIR = './visRealResults/'

        self._C.VIEW_DIP = CN()
        self._C.VIEW_DIP.TEST_DIR = './Dataset/DVD_test'
        self._C.VIEW_DIP.MODEL = './checkpoints/Fusion/models/model_epoch_80.pth'
        self._C.VIEW_DIP.SIGMA = 4
        self._C.VIEW_DIP.VIS_DIR = './visResults/'

        # Override parameter values from YAML file first, then from override list.
        self._C.merge_from_file(config_yaml)
        self._C.merge_from_list(config_override)

        # Make an instantiated object of this class immutable.
        self._C.freeze()

    def dump(self, file_path: str):
        self._C.dump(stream=open(file_path, "w"))

    def __getattr__(self, attr: str):
        return self._C.__getattr__(attr)

    def __repr__(self):
        return self._C.__repr__()