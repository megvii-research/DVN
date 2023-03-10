import os
from config import Config 
opt = Config('training.yml')

gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

import torch
torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import random
import time
import numpy as np

from utils.dir_utils import mkdir, mkdirs
from utils import util
from dataset import DataLoaderTrain
import losses
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from pdb import set_trace as stx
from model.get_model import get_model, get_pretrain, load_model
import time
import logging
from logging import handlers
logging.basicConfig(format='',
                    level=logging.DEBUG)

######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

start_epoch = 1
mode = opt.MODEL.MODE
try:
    assert (mode == 'Recons' or mode == 'Fusion')
except:
    raise Exception("The mode must be Recons or Fusion")

opt_train = opt.Recons if mode == 'Recons' else opt.Fusion

model_dir  = os.path.join(opt_train.SAVE_DIR, mode, 'models')

mkdir(model_dir)

train_dir = opt_train.TRAIN_DIR


######### Model ###########
if not opt_train.RESUME:
    net = get_pretrain() if mode == 'Recons' else get_model()
    net.cuda()
else:
    net = load_model(opt_train.RESUME_PATH, mode=mode)

if mode == 'Fusion':
    net_edge = load_model(opt.Edge_Detect.MODEL, mode='Recons')
    net_edge.eval()


device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
  print("\n\nLet's use ", torch.cuda.device_count(), " GPUs!\n\n")


new_lr = opt.OPTIM.LR_INITIAL

optimizer = optim.Adam(net.parameters(), lr=new_lr, betas=(0.9, 0.999),eps=1e-8, weight_decay=1e-8)


######### Scheduler ###########
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt_train.NUM_EPOCHS - warmup_epochs+40, eta_min=opt.OPTIM.LR_MIN)
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
scheduler.step()


######### Resume ###########
if opt_train.RESUME:
    start_epoch = util.load_start_epoch(opt_train.RESUME_PATH) + 1
    util.load_optim(optimizer, opt_train.RESUME_PATH)

    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')

if len(device_ids)>1:
    net = nn.DataParallel(net, device_ids = device_ids)
    if mode == 'Fusion':
        net_edge = nn.DataParallel(net_edge, device_ids = device_ids)


######### Loss ###########
criterion_recon = losses.CharbonnierLoss()
if mode == 'Fusion':
    criterion_grad = losses.DiceLoss()


######### DataLoaders ###########
train_dataset = DataLoaderTrain(opt_train.TRAIN_DIR, opt_train.TRAIN_PS)
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=32, drop_last=True, pin_memory=True)

print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt_train.NUM_EPOCHS + 1))
print('===> Loading datasets')


######### Training ###########
for epoch in range(start_epoch, opt_train.NUM_EPOCHS + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1

    net.train()
    time_epoch_start = tstart = time.time()

    for i, data in enumerate(train_loader):
        # zero_grad
        for param in net.parameters():
            param.grad = None

        cur_time = time.time()
        inp_rgb, nir, gt_rgb = data
        inp_rgb, nir, gt_rgb = inp_rgb.cuda(), nir.cuda(), gt_rgb.cuda()
        
        tdata = time.time() - tstart

        # Compute loss at each stage
        if mode == 'Recons':
            pred, *_ = net(gt_rgb, nir)
            loss_rgb = criterion_recon(torch.clamp(pred[0],0,1), gt_rgb)
            loss_nir = criterion_recon(torch.clamp(pred[1],0,1), nir)
            loss = loss_rgb + loss_nir
        else:
            with torch.no_grad():
                *_, [rgb_mask, ir_mask] = net_edge(gt_rgb, nir)

            pred, pred_edges, *_ = net(inp_rgb, nir)
            loss_rgb = criterion_recon(torch.clamp(pred[0], 0, 1), gt_rgb)
            loss_rgb_mid = criterion_recon(torch.clamp(pred[1], 0, 1), gt_rgb)
            loss_nir = criterion_recon(torch.clamp(pred[2], 0, 1), nir)
            loss_rgb_stru = np.sum([criterion_grad(torch.clamp(pred_edges[0][i], 0, 1), rgb_mask[i]) for i in range(3)]) / 1000
            loss_nir_stru = np.sum([criterion_grad(torch.clamp(pred_edges[1][i], 0, 1), ir_mask[i]) for i in range(3)]) / 3000
            loss = loss_rgb + loss_rgb_mid + loss_nir + loss_rgb_stru + loss_nir_stru

        loss.backward()
        optimizer.step()

        ttrain = time.time() - tstart
        time_passed = time.time() - time_epoch_start

        epoch_loss +=loss.item()
        net.train()

        tstart = time.time()

        outputs = [
                "e: {}, {}/{}".format(epoch, i, 1000),
                "tdata:{:.2f} s".format(ttrain),
                "loss_rgb {:.4f} ".format(loss_rgb.item() * 1000),
                "loss_rgb_mid {:.4f} ".format(loss_rgb_mid.item() * 1000),
        ]

        if mode == 'Fusion':
            outputs += [
                "loss_nir {:.4f} ".format(loss_nir.item() * 1000),
                "loss_rgb_stru {:.4f} ".format(loss_rgb_stru.item() * 1000),
                "loss_nir_stru {:.4f} ".format(loss_nir_stru.item() * 1000),
            ]
        outputs += [
                'passed:{:.2f}'.format(time_passed),
                "lr:{:.4g}".format(optimizer.param_groups[0]['lr'] * 100000),
                "dp/tot: {:.2g}".format(tdata / ttrain),
            ]
        logging.info("  ".join(outputs))


    scheduler.step()
    
    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time, epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")

    torch.save({'epoch': epoch, 
                'state_dict': net.state_dict(),
                'optimizer' : optimizer.state_dict()
                }, os.path.join(model_dir,"model_epoch_{}.pth".format(epoch))) 

print('Training done.')
