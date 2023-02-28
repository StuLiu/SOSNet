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


class UDD6(Dataset):
    """UDD6 dataset.

        'label name'        'R,G,B'         'label id'
        Other               (0,0,0)          0
        Facade              (102,102,156)    1
        Road                (128,64,128)     2
        Vegetation	        (107,142,35)     3
        Vehicle             (0,0,142)        4
        Roof                (70,70,70)       5

    """

    CLASSES = ('Other', 'Facade', 'Road', 'Vegetation', 'Vehicle', 'Roof')

    PALETTE = torch.tensor([[0, 0, 0], [102, 102, 156], [128, 64, 128], [107, 142, 35],
                            [0, 0, 142], [70, 70, 70]])

    SMALL_OBJECT = [4]

    def __init__(self, root: str, split: str = 'train', transform=None, preload=False) -> None:
        super().__init__()
        assert split in ['train', 'val']
        # assert split in ['train', 'val', 'test']
        self.split = split
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255
        self.preload = preload
        self.pairs = []
        # r=osp.join(root, 'img_dir', self.split) + '/*.png'
        imgs = glob(osp.join(root, 'img_dir', self.split) + '/*.png')
        for img_path in imgs:
            lbl_path = img_path.replace('img_dir', 'ann_dir')
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
    _dataset = UDD6('../../data/UDD6', 'train', preload=False)
    for _i, _l in _dataset:
        break
