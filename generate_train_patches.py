from tqdm import tqdm
import numpy as np
import os
from joblib import Parallel, delayed
import argparse

parser = argparse.ArgumentParser(description='Generate patches from Full Resolution images')
parser.add_argument('--src_dir', default='./Dataset/DVD_train_raw', type=str, help='Directory for full resolution images')
parser.add_argument('--tar_dir', default='./Dataset/DVD_train',type=str, help='Directory for image patches')
parser.add_argument('--ps', default=64, type=int, help='Raw Image Patch Size')
parser.add_argument('--num_patches', default=60, type=int, help='Number of patches per image')
parser.add_argument('--num_cores', default=10, type=int, help='Number of CPU Cores')

args = parser.parse_args()

src = args.src_dir
tar = args.tar_dir
PS = args.ps  # this is the raw image patch size, which is 1/4 of the output image patch size
NUM_PATCHES = args.num_patches
NUM_CORES = args.num_cores

rgb_patchDir = os.path.join(tar, 'RGB')
nir_patchDir = os.path.join(tar, 'NIR')

if os.path.exists(tar):
    os.system("rm -r {}".format(tar))

os.makedirs(rgb_patchDir)
os.makedirs(nir_patchDir)

# get sorted folders
files = os.listdir(os.path.join(src, 'RGB'))
files = list(filter(lambda x: x.endswith('npy'), files))

rgb_filenames = [os.path.join(src, 'RGB', x)  for x in files]
nir_filenames = [os.path.join(src, 'NIR', x) for x in files]


def save_files(i):
    rgb_file, nir_file = rgb_filenames[i], nir_filenames[i]

    rgb = np.load(rgb_file)
    nir = np.load(nir_file)
    rgb = rgb[64:316, 112:560, ...]
    nir = nir[64:316, 112:560, ...]

    H = rgb.shape[0]
    W = rgb.shape[1]
    for j in range(NUM_PATCHES):
        rr = np.random.randint(0, H - PS)
        cc = np.random.randint(0, W - PS)
        rgb_patch = rgb[rr:rr + PS, cc:cc + PS, :]
        nir_patch = nir[rr:rr + PS, cc:cc + PS, :]

        np.save(os.path.join(rgb_patchDir, '{}_{}.png'.format(i+1,j+1)), rgb_patch)
        np.save(os.path.join(nir_patchDir, '{}_{}.png'.format(i+1,j+1)), nir_patch)

Parallel(n_jobs=NUM_CORES)(delayed(save_files)(i) for i in tqdm(range(len(files))))
