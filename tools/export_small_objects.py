"""
@Project : semantic-segmentation
@File    : export_small_objects.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2023/6/20 下午4:51
@e-mail  : 1183862787@qq.com
"""

import cv2
import os
import numpy as np
from semseg.datasets import *
from torch.utils.data import DataLoader
from glob import glob
import argparse
from tqdm import tqdm
import yaml


def filtering_image(lbl, out_path, min_area=0, max_area=1024, num_classes=11, ignore_label=255):
    lbl_out = np.ones_like(lbl) * ignore_label
    for _id in range(num_classes):
        label = np.zeros_like(lbl)
        label[lbl == _id] = 255
        # 找到所有的轮廓
        contours, _ = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # 遍历每个轮廓
        for contour in contours:
            # 忽略太大的轮廓
            if min_area <= cv2.contourArea(contour) <= max_area :#and cv2.contourArea(contour) >= min_area:
                # 将轮廓内部的像素设为0
                cv2.drawContours(lbl_out, [contour], 0, _id, -1)
        # cv2.imshow('window', lbl_out)
        # cv2.waitKey(0)
    # 保存抠出小目标后的语义分割标签
    cv2.imwrite(out_path, lbl_out)
    # cv2.imshow('window', lbl_out)
    # cv2.waitKey(0)


def filtering_by_area(in_dir,  min_area=0, max_area=1024, num_classes=11, ignore_label=255):
    out_dir = f'{in_dir}_so_{min_area}_{max_area}'
    os.makedirs(out_dir, exist_ok=True)
    img_paths = glob(r''+in_dir+'/*.png')
    img_paths.sort()
    for img_path in tqdm(img_paths):
        lbl = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        out_path = os.path.join(out_dir, os.path.basename(img_path))
        filtering_image(lbl, out_path=out_path, min_area=min_area, max_area=max_area, num_classes=num_classes,
                        ignore_label=ignore_label)
        # break

def filter_objects(in_dir, num_classes=11, ignore_label=255):
    areas = [0, 1024, 4096, 16384, 65536, 1048576]
    for i in range(len(areas) - 1):
        filtering_by_area(in_dir, 0, areas[i + 1], num_classes, ignore_label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/Bisenetv2/camvid.yaml', help='config file path')
    parser.add_argument('--input-dir', type=str, default='data/CamVid/testannot', help='directory of label files')
    # parser.add_argument('--max-area', type=int, default=16384, help='maximum area for objects. 1024, 4096, 16384')
    args = parser.parse_args()
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    datasets = eval(cfg["DATASET"]["NAME"])(cfg["DATASET"]["ROOT"], 'test')
    filter_objects(in_dir=args.input_dir, #out_dir=args.input_dir + '_so' + str(args.max_area),
                   num_classes=datasets.n_classes, ignore_label=datasets.ignore_label)
