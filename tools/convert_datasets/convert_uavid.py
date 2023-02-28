"""
@Project : semantic-segmentation 
@File    : convert_uavid.py
@IDE     : PyCharm 
@Date    : 2022/6/9 15:01
@Author  : Wang Liu
@e-mail  : liuwa@hnu.edu.cn
"""

import os
import os.path as osp
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image


def gen_test_lbl(test_ann_dir='../../data/UAVid2020/ann_dir/test'):
    imgs = glob(test_ann_dir + r'/*.png')
    for img_path in tqdm(imgs):
        img_bgr = cv2.imread(img_path)
        lbl_np = np.zeros((img_bgr.shape[0], img_bgr.shape[1], 1), dtype=np.uint8)
        cv2.imwrite(img_path, lbl_np)


def split_image_and_label(img_path, lbl_path, img_save_dir, lbl_save_dir,
                          height=1080, width=1920, h_step=540, w_step=960):
    os.makedirs(img_save_dir, exist_ok=True)
    os.makedirs(lbl_save_dir, exist_ok=True)
    img_base = osp.basename(img_path)
    lbl_base = osp.basename(lbl_path)
    img_bgr = cv2.imread(img_path)
    lbl_bgr = cv2.imread(lbl_path)
    h, w = img_bgr.shape[0], img_bgr.shape[1]   # 2160, 3840
    for i in range(0, h - height + 1, h_step):
        for j in range(0, w - width + 1, w_step):
            img_bgr_split = img_bgr[i:i+height, j:j+width, :]
            lbl_bgr_split = lbl_bgr[i:i+height, j:j+width, :]
            img_base_new = img_base.replace('.png', f'_{i}_{j}.png')
            lbl_base_new = lbl_base.replace('.png', f'_{i}_{j}.png')
            cv2.imwrite(osp.join(img_save_dir, img_base_new), img_bgr_split)
            cv2.imwrite(osp.join(lbl_save_dir, lbl_base_new), lbl_bgr_split)


def do_split(load_dir='../../data/UAVid2020', split='train', save_dir='../../data/UAVid2020_split'):
    os.makedirs(save_dir, exist_ok=True)
    img_dir = osp.join(load_dir, 'img_dir', split)
    imgs = os.listdir(img_dir)
    print(len(imgs))
    for img_base in tqdm(imgs):
        img_path = osp.join(img_dir, img_base)
        lbl_path = img_path.replace('img_dir', 'ann_dir')
        img_save_dir = osp.join(save_dir, 'img_dir', split)
        lbl_save_dir = osp.join(save_dir, 'ann_dir', split)
        split_image_and_label(img_path, lbl_path, img_save_dir, lbl_save_dir)
        # break


def lblshow(lblidpath='../../data/UAVid2020_mm/ann_dir/train/seq35_000900.png'):
    img = Image.open(lblidpath).convert('P')
    colormap = np.array([
        [0, 0, 0], [128, 0, 0], [128, 64, 128], [192, 0, 192],
        [0, 128, 0], [128, 128, 0], [64, 64, 0], [64, 0, 128]
    ], dtype=np.uint8)
    img.putpalette(colormap.flatten())
    img.save('test_color.png')


def lblshow_so(lblidpath='../../data/UAVid2020_mm/ann_dir/train/seq1_000000.png'):
    img = Image.open(lblidpath).convert('P')
    colormap = np.array([
        [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 255, 255],
        [0, 0, 0], [0, 0, 0], [255, 255, 255], [255, 255, 255]
    ], dtype=np.uint8)
    '''
            'label name'        'R,G,B'         'label id'
        Background clutter  (0,0,0)         0
        Building            (128,0,0)       1
        Road                (128,64,128)    2
        Static car          (192,0,192)     3
        Tree                (0,128,0)       4
        Low vegetation      (128,128,0)     5
        Human               (64,64,0)       6
        Moving car          (64,0,128)      7 '''
    img.putpalette(colormap.flatten())
    img.save('test_so.png')


if __name__ == '__main__':
    # lblshow_so()
    # gen_test_lbl(test_ann_dir='../../data/UAVid2020_mm/ann_dir/test')
    # do_split(load_dir='../../data/UAVid2020_mm', split='train', save_dir='../../data/UAVid2020')
    # do_split(load_dir='../../data/UAVid2020_mm', split='val', save_dir='../../data/UAVid2020')
    # do_split(load_dir='../../data/UAVid2020_mm', split='test', save_dir='../../data/UAVid2020')
    lblshow('../../data/UAVid2020_mm/ann_dir/train/seq1_000000.png')
    lblshow_so('../../data/UAVid2020_mm/ann_dir/train/seq1_000000.png')
