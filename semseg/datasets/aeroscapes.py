"""
@Project : semantic-segmentation 
@File    : uavid2020.py
@IDE     : PyCharm 
@Author  : Wang Liu
@Date    : 2022/4/30 下午8:02
@e-mail  : 1183862787@qq.com
"""
import os
import os.path as osp
import torch
import logging
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io
from pathlib import Path
from typing import Tuple
from glob import glob


class Aeroscapes(Dataset):
    """UDD6 dataset.

        'label name'        'R,G,B'         'label id'
        unlabeled,          0, 0, 0
        paved-area,         128, 64, 128
        dirt,               130, 76, 0
        grass,              0, 102, 0
        gravel,             112, 103, 87
        water,              28, 42, 168
        rocks,              48, 41, 30
        pool,               0, 50, 89
        vegetation,         107, 142, 35
        roof,               70, 70, 70
        wall,               102, 102, 156
        window,             254, 228, 12
        door,               254, 148, 12
        fence,              190, 153, 153
        fence-pole,         153, 153, 153
        person,             255, 22, 96
        dog,                102, 51, 0
        car,                9, 143, 150
        bicycle,            119, 11, 32
        tree,               51, 51, 0
        bald-tree,          190, 250, 190
        ar-marker,          112, 150, 146
        obstacle,           2, 135, 115
        conflicting,        255, 0, 0

    """

    CLASSES = ('unlabeled', 'paved-area', 'dirt', 'grass', 'gravel',
               'water', 'rocks', 'pool', 'vegetation', 'roof',
               'wall', 'window', 'door', 'fence', 'fence-pole',
               'person', 'dog', 'car', 'bicycle', 'tree',
               'bald-tree', 'ar-marker', 'obstacle', 'conflicting')

    PALETTE = torch.tensor([
        [0, 0, 0],
        [128, 64, 128],
        [130, 76, 0],
        [0, 102, 0],
        [112, 103, 87],
        [28, 42, 168],
        [48, 41, 30],
        [0, 50, 89],
        [107, 142, 35],
        [70, 70, 70],
        [102, 102, 156],
        [254, 228, 12],
        [254, 148, 12],
        [190, 153, 153],
        [153, 153, 153],
        [255, 22, 96],
        [102, 51, 0],
        [9, 143, 150],
        [119, 11, 32],
        [51, 51, 0],
        [190, 250, 190],
        [112, 150, 146],
        [2, 135, 115],
        [255, 0, 0],
    ])

    SMALL_OBJECT = []

    def __init__(self, root: str, split: str = 'train', transform=None, preload=False) -> None:
        super().__init__()
        assert split in ['train', 'val']
        # assert split in ['train', 'val', 'test']
        self.split = split
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 0
        self.preload = preload
        self.pairs = []
        # r=osp.join(root, 'img_dir', self.split) + '/*.png'
        imgs = glob(osp.join(root, 'img_dir', self.split) + '/*.jpg')
        for img_path in imgs:
            lbl_path = img_path.replace('img_dir', 'ann_dir').replace('.jpg', '.png')
            data_pair = [
                io.read_image(img_path) if self.preload else img_path,
                io.read_image(lbl_path)[-1:] if self.preload else lbl_path,
            ]
            self.pairs.append(data_pair)

        assert len(self.pairs) > 0, f"No images found in {root}"
        logging.info(f"Found {len(self.pairs)} {split} images.")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        image, label = self.pairs[index]
        if not self.preload:
            image = io.read_image(image)
            label = io.read_image(label)[-1:]

        if self.transform:
            image, label = self.transform(image, label)
        return image, torch.squeeze(label.long())


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import cv2
    import numpy as np


    train_dataset = Aeroscapes('../../data/Aeroscapes', split='train')
    val_dataset = Aeroscapes('../../data/Aeroscapes', split='val')
    print(f'train size={len(train_dataset)}, val size={len(val_dataset)}')

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    for _img, _lbl in train_dataloader:
        print(_img.cpu().numpy().shape, _lbl.cpu().numpy().shape, np.unique(_lbl.cpu().numpy()))
        cc = _lbl.cpu().numpy().squeeze().astype(np.uint8)
        cv2.imshow('img', _img.cpu().numpy().squeeze().transpose((1,2,0)))
        # cv2.imshow('lbl', np.array([cc, cc, cc]).transpose((1, 2, 0)).astype(np.uint8))
        cv2.imshow('lbl', cc)
        cv2.waitKey(0)
