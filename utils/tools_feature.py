import cv2
import numpy as np
from scipy import signal
import math
from skimage.measure import compare_ssim
import losses
import torch
from torchvision.utils import make_grid
from torch.nn import functional as F
from .tools import gen_jobs

def view_features(net, input_bgr, input_ir, label_bgr):
    n, c, h, w = input_bgr.shape
    output_img = np.zeros(input_bgr.shape, dtype=np.float32)
    mask = np.zeros(input_bgr.shape[-2:], dtype=np.uint8)
    structures = ['rgb_structure', 'ir_structure']
    features = ['rgb_feature', 'nir_structure', 'DIP', 'weighted_ir']
    struct_imgs = [{k: [] for k in structures} for _ in range(3)] # three different scales
    feat_imgs = [{k: [] for k in features} for _ in range(3)]
    bboxs  = [[] for _ in range(3)]

    for i, [bgr_patch, ir_patch, label_patch, bbox] in enumerate(gen_jobs(input_bgr, input_ir, label_bgr, patch_shape=(800, 480), stride=(200,120))):
        pred_out = net(bgr_patch, ir_patch)
        out = pred_out[0][0].cpu().numpy()

        x_tf,y_tf,x_br,y_br = bbox
        output_img[:, :, y_tf:y_br, x_tf:x_br] = output_img[:, :, y_tf:y_br, x_tf:x_br] + out
        mask[y_tf:y_br, x_tf:x_br] += 1
        output_img[:, :, y_tf:y_br, x_tf:x_br] = output_img[:, :, y_tf:y_br, x_tf:x_br] / mask[y_tf:y_br, x_tf:x_br]

        for u in range(3):
            for j, k in enumerate(structures):
                struct_imgs[u][k].append(pred_out[1][j][u].cpu().numpy())
            for j, k in enumerate(features):
                feat_imgs[u][k].append(pred_out[2][j][u].cpu().numpy())

            bboxs[u].append(list(map(lambda x: x // (2 ** u), bbox)))
        

    out_feats = [None for _ in range(3)]
    channels = [80, 80 + 48, 80 + 48 * 2]
    for i in range(3):
        connact = Connact_and_View([1, channels[i], h // (2 ** i), w // (2 ** i)])
        out_feats[i] = connact.connact_view({**struct_imgs[i], **feat_imgs[i]}, bboxs[i])
    
    return output_img, out_feats

def view_edgedetect(net, input_bgr, input_ir, label_bgr):
    n, c, h, w = input_bgr.shape
    edges = ['rgb_edges', 'ir_edges']
    edges_imgs = [{k: [] for k in edges} for _ in range(3)] # three different scales
    bboxs  = [[] for _ in range(3)]

    for i, [bgr_patch, ir_patch, label_patch, bbox] in enumerate(gen_jobs(input_bgr, input_ir, label_bgr, patch_shape=(800, 480), stride=(200,120))):
        *_, pred_edges = net(label_patch, ir_patch)
        
        x_tf,y_tf,x_br,y_br = bbox

        for u in range(3):
            for j, k in enumerate(edges):
                edges_imgs[u][k].append(pred_edges[j][u].cpu().numpy())

            bboxs[u].append(list(map(lambda x: x // (2 ** u), bbox)))
        

    out_edges = [None for _ in range(3)]
    channels = [80, 80 + 48, 80 + 48 * 2]
    for i in range(3):
        connact = Connact_and_View([1, channels[i], h // (2 ** i), w // (2 ** i)])
        out_edges[i] = connact.connact_view(edges_imgs[i], bboxs[i])
    
    return out_edges


class Connact_and_View():
    def __init__(self, img_shape):
        n, c, h, w = img_shape
        self.idx = np.arange(3) if c < 9 else np.random.choice(np.arange(c), 9, replace=False)
        self.img_shape = img_shape

    def connact_view(self, imgs, bboxs):
        outputs = {}
        mask = np.zeros(self.img_shape[-2:], dtype=np.uint8)
        for k in imgs:
            outputs[k] = np.zeros(self.img_shape, dtype=np.float32)

        for i, [x_tf, y_tf, x_br, y_br] in enumerate(bboxs):
            for k, v in imgs.items():
                outputs[k][:,:,y_tf:y_br,x_tf:x_br] = outputs[k][:,:,y_tf:y_br,x_tf:x_br] * mask[y_tf:y_br,x_tf:x_br] + v[i]

            mask[y_tf:y_br,x_tf:x_br] += 1

            for k, v in imgs.items():
                outputs[k][:,:,y_tf:y_br,x_tf:x_br] = outputs[k][:,:,y_tf:y_br,x_tf:x_br] / mask[y_tf:y_br,x_tf:x_br]

        for k in outputs:
            outputs[k] = outputs[k][:, self.idx, ...]

        return outputs
