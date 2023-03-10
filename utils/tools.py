import cv2
import numpy as np
from scipy import signal
import math
from skimage.measure import compare_ssim
import torch
import itertools
import lpips
from torchvision.utils import make_grid

def make_view(img, equ=False, norm=False):
    n, c, h, w = img.shape
    for i in range(c):
        x = img[0,i, ...]
        if norm:
            x = (x - x.min()) / (x.max() - x.min()) * 255
        else:
            x = x / x.max() * 255
        x = x.astype(np.uint8)
        if equ:
            img[0,i, ...] = cv2.equalizeHist(x)
        else:
            img[0,i, ...] = x

    img = torch.from_numpy(img)
    img = img.permute(1, 0, 2, 3)
    new_img = make_grid(img, nrow=3, padding=5, normalize=False)
    new_img = new_img.permute(1, 2, 0).numpy()
    return new_img

def compute_psnr(img1, img2):
    img1 = img1.reshape(-1)
    img2 = img2.reshape(-1)
    err = np.mean((img1 - img2) ** 2)
    if err < 1e-5:
        return 100
    return 10 * math.log10(1 / err)

def compute_ssim(img1, img2):
    res = compare_ssim(img2, img1, win_size=11, multichannel=True)
    return res


def gather_patches_into_whole(net, input_bgr, input_ir, label_bgr):
    output_img = np.zeros(input_bgr.shape, dtype=np.float32)   # record the final fusion image
    out1_img = np.zeros(input_bgr.shape, dtype=np.float32)     # record the middle denoised image
    mask = np.zeros(input_bgr.shape[-2:], dtype=np.uint8)
    for i, [bgr_patch, ir_patch, label_patch, bbox] in enumerate(gen_jobs(input_bgr, input_ir, label_bgr, patch_shape=(800, 480), stride=(200,120))):
        output_patch, out1_patch = net(bgr_patch, ir_patch)[0][:2]
        output_patch = output_patch.cpu().numpy()
        out1_patch = out1_patch.cpu().numpy()

        x_tf,y_tf,x_br,y_br = bbox
        output_img[:,:,y_tf:y_br,x_tf:x_br] = \
            output_img[:,:,y_tf:y_br,x_tf:x_br] * mask[y_tf:y_br,x_tf:x_br] + output_patch
        out1_img[:,:,y_tf:y_br,x_tf:x_br] = \
            out1_img[:,:,y_tf:y_br,x_tf:x_br] * mask[y_tf:y_br,x_tf:x_br] + out1_patch

        mask[y_tf:y_br,x_tf:x_br] += 1
        output_img[:,:,y_tf:y_br,x_tf:x_br] = output_img[:,:,y_tf:y_br,x_tf:x_br] / mask[y_tf:y_br,x_tf:x_br]
        out1_img[:,:,y_tf:y_br,x_tf:x_br] = out1_img[:,:,y_tf:y_br,x_tf:x_br] / mask[y_tf:y_br,x_tf:x_br]
    return output_img, out1_img



def get_patches(input_bgr, input_ir, label, patch_shape=(256, 256), stride=(0, 0)):
    assert input_bgr.ndim == input_ir.ndim == label.ndim, "You must enter the data in the shape of NCHW"
    assert input_bgr.shape[-2:] == input_ir.shape[-2:] == label.shape[-2:]

    n, c, h, w = input_bgr.shape
    w_patch, h_patch = patch_shape
    w_stride, h_stride = stride
    assert h >= h_patch and w >= w_patch, "The size of the input image is too small or the size of the patch is too large."

    topleft_x_array = np.arange(w)[::w_patch-w_stride]
    topleft_y_array = np.arange(h)[::h_patch-h_stride]
    bgr_patch_list, ir_patch_list, label_patch_list = [], [], []
    bbox_list = []
    for x_tf, y_tf in itertools.product(topleft_x_array, topleft_y_array):
        x_br, y_br = min(x_tf+w_patch, w), min(y_tf+h_patch, h)
        x_tf, y_tf = max(x_br-w_patch, 0), max(y_br-h_patch, 0)
        bgr_patch_list.append(input_bgr[:,:,y_tf:y_br,x_tf:x_br])
        ir_patch_list.append(input_ir[:,:,y_tf:y_br,x_tf:x_br])
        label_patch_list.append(label[:,:,y_tf:y_br,x_tf:x_br])
        bbox_list.append([x_tf,y_tf,x_br,y_br])
    return bgr_patch_list, ir_patch_list, label_patch_list, bbox_list


def gen_jobs(data_bgr, label_ir, label_bgr, patch_shape=(256, 256), stride=(0,0)):
    bgr_patch_list, ir_patch_list, label_patch_list, bbox_list = get_patches(
        data_bgr, label_ir, label_bgr,
        patch_shape=patch_shape, stride=stride
    )
    for bgr_patch, ir_patch, label_patch, bbox in zip(bgr_patch_list, ir_patch_list, label_patch_list, bbox_list):
        yield bgr_patch, ir_patch, label_patch, bbox



def validation_on_PSNR_and_SSIM(net, input_rgb, input_ir, label):
    assert type(input_rgb) is torch.Tensor
    assert type(input_ir) is torch.Tensor
    ssims_o = []
    psnrs_o = []
    ssims_i = []
    psnrs_i = []

    for bgr_patch, ir_patch, label_patch, _ in gen_jobs(input_rgb, input_ir, label):
        out = torch.clamp(net(bgr_patch, ir_patch)[0][0], 0, 1)
        out = out.cpu().numpy()
        inp = bgr_patch[0, ...].permute(1,2,0).cpu().numpy()
        oup = out[0, ...].transpose(1,2,0)
        lab = label_patch[0, ...].permute(1,2,0).cpu().numpy()
        L = lab.max()
        inp = inp / L
        lab = lab / L
        oup = oup / L
    
        i_psnr = compute_psnr(lab, inp)
        i_ssim = compute_ssim(lab, inp)
        psnrs_i.append(i_psnr)
        ssims_i.append(i_ssim)

        o_psnr = compute_psnr(lab, oup)
        o_ssim = compute_ssim(lab, oup)
        psnrs_o.append(o_psnr)
        ssims_o.append(o_ssim)

    return np.array(psnrs_i).mean(), np.array(ssims_i).mean(), np.array(psnrs_o).mean(), np.array(ssims_o).mean()

