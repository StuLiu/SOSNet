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


class UAVid2020(Dataset):
    """UAVid2020 dataset.

    In segmentation map annotation for UAVid2020, 0 stands for background, which is
    included in 8 categories. ``reduce_zero_label`` is fixed to False. The
    ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to '.png', too.
    In UAVid2020, 200 images for training, 70 images for validating, and 150 images for testing.
    The 8 classes and corresponding label color (R,G,B) are as follows:
        'label name'        'R,G,B'         'label id'
        Background clutter  (0,0,0)         0
        Building            (128,0,0)       1
        Road                (128,64,128)    2
        Static car          (192,0,192)     3
        Tree                (0,128,0)       4
        Low vegetation      (128,128,0)     5
        Human               (64,64,0)       6
        Moving car          (64,0,128)      7

    """

    CLASSES = ('Background clutter', 'Building', 'Road', 'Static car',
               'Tree', 'Low vegetation', 'Human', 'Moving car')

    PALETTE = torch.tensor([[0, 0, 0], [128, 0, 0], [128, 64, 128], [192, 0, 192],
                            [0, 128, 0], [128, 128, 0], [64, 64, 0], [64, 0, 128]])

    SMALL_OBJECT = [3, 6, 7]

    def __init__(self, root: str, split: str = 'train', transform=None, preload=False) -> None:
        super().__init__()
        assert split in ['train', 'val', 'test']
        # assert split in ['train', 'val', 'test']
        self.split = split
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255
        self.preload = preload
        self.pairs = []

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
    _dataset = UAVid2020('../../data/UAVid2020_mm', 'train', preload=False)
    for _i, _l in _dataset:
        print(_i.size(), _l.size())
