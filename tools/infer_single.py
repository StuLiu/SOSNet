"""
@Project : semantic-segmentation 
@File    : infer_single.py
@IDE     : PyCharm 
@Author  : Wang Liu
@Date    : 2022/10/14 下午9:26
@e-mail  : 1183862787@qq.com
"""


import os
import argparse
import yaml
from torchvision import io
from tools.infer import SemSeg
from PIL import Image
import numpy as np
import cv2
from semseg.datasets import *

def overlay_gt(dataset, img_path, lbl_path, save_dir, overlay=False, img_ratio=0.3):
    img = Image.open(img_path)
    lbl = Image.open(lbl_path).convert('P')
    colormap = dataset.PALETTE.numpy().astype(np.uint8)
    lbl.putpalette(colormap.flatten())
    lbl = lbl.convert('RGB')
    if overlay:
        img = (np.array(img) * img_ratio) + (np.array(lbl) * (1 - img_ratio))
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_dir, f"{str(os.path.basename(img_path))}"), img)
    else:
        lbl.save(os.path.join(save_dir, f"{str(os.path.basename(img_path))}"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='data/CamVid/test/0001TP_008550.png')
    parser.add_argument('--cfg', type=str,
        default='output_ablation/deeplabv3plus/camvid/DeeplabV3Plus_CamVid_hier_soem_59.41_38.53_71.33/config.yaml')
    parser.add_argument('--overlay', type=str, default=True)
    parser.add_argument('--gt', type=str, default=False)    # if generate gt from label
    parser.add_argument('--ratio', type=float, default=0.3)
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    test_file = os.path.join(args.img_path)
    save_dir = f'./vis_results/{cfg["MODEL"]["NAME"]}'
    os.makedirs(save_dir, exist_ok=True)

    semseg = SemSeg(cfg)

    print(f'Inferencing {test_file} by {cfg["MODEL"]["NAME"]}...')
    segmap = semseg.predict(str(test_file), args.overlay, args.ratio)
    io.write_png(segmap, os.path.join(save_dir, f"{str(os.path.basename(test_file))}"))

    trainset = eval(cfg['DATASET']['NAME'])(cfg['DATASET']['ROOT'], 'train', None)

    import shutil
    if args.gt:
        gt_dir = os.path.join(save_dir, '..', 'gt')
        os.makedirs(gt_dir, exist_ok=True)
        shutil.copy(args.img_path, os.path.join(gt_dir, f"{str(os.path.basename(args.img_path))}.img.png"))
        if cfg['DATASET']['NAME'] == 'ISAID':
            overlay_gt(trainset, args.img_path,
                       args.img_path.replace('img_dir', 'ann_dir').replace('.png', '_instance_color_RGB.png'),
                       gt_dir, True)
        elif cfg['DATASET']['NAME'] == 'CamVid':
            overlay_gt(trainset, args.img_path,
                       args.img_path.replace('test/', 'testannot/'),
                       gt_dir, True, img_ratio=args.ratio)
        elif cfg['DATASET']['NAME'] == 'UAVid':
            overlay_gt(trainset, args.img_path,
                       args.img_path.replace('img_dir/', 'ann_dir/'),
                       gt_dir, True, img_ratio=args.ratio)