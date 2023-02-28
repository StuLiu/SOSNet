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
import cv2
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io
from pathlib import Path
from typing import Tuple
from glob import glob
import skimage


class HTHT2022Base(Dataset):

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

        imgs = glob(osp.join(root, self.split, 'images') + '/*.tif')
        for img_path in imgs:
            lbl_path = img_path.replace('images', 'labels').replace('.tif', '.png')
            data_pair = [img_path, lbl_path]
            self.pairs.append(data_pair)

        assert len(self.pairs) > 0, f"No images found in {root}"
        logging.info(f"Found {len(self.pairs)} {split} images.")

    def __len__(self) -> int:
        return len(self.pairs)


class HTHT2022Coarse(HTHT2022Base):
    """
    无效标注 	0
    水体	    100
    交通运输	200
    建筑	    300
    耕地	    400
    草地	    500
    林地	    600
    裸土	    700
    其它	    800
    """
    CLASSES = ('clutter', 'water', 'transport', 'building', 'arable_land',
               'grassland', 'woodland', 'bare_soil', 'others')

    PALETTE = torch.tensor([[0, 0, 0], [0, 0, 128], [128, 64, 128], [192, 0, 64], [0, 128, 0],
                            [0, 64, 0], [0, 256, 0], [128, 128, 128], [128, 0, 0]])

    SMALL_OBJECT = []

    def __init__(self, root: str, split: str = 'train', transform=None, preload=False):
        super().__init__(root, split, transform, preload)

        self.ignore_label = 0

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        image, label = self.pairs[index]

        image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
        label = skimage.io.imread(label) // 100
        # print(np.unique(label))
        image = torch.tensor(image, dtype=torch.uint8).permute(2, 0, 1)
        label = torch.unsqueeze(torch.tensor(label.astype(np.int32), dtype=torch.long), 0)

        if self.transform:
            image, label = self.transform(image, label)
        return image, torch.squeeze(label.long())


# class HTHT2022Fine(HTHT2022Base):
#     """
#     水体	01	普通耕地	09
#     道路	02	农业大棚	10
#     建筑物	03	自然草地	11
#     机场	04	绿地绿化	12
#     火车站	05	自然林	    13
#     光伏	06	人工林      14
#     停车场	07	自然裸土    15
#     操场	08	人为裸土    16
#     其它无法确定归属地物	17
#     """
#     CLASSES = ('Background clutter', 'Building', 'Road', 'Static car',
#                'Tree', 'Low vegetation', 'Human', 'Moving car')
#
#     PALETTE = torch.tensor([[0, 0, 0], [128, 0, 0], [128, 64, 128], [192, 0, 192],
#                             [0, 128, 0], [128, 128, 0], [64, 64, 0], [64, 0, 128]])
#
#     SMALL_OBJECT = []
#
#     def __init__(self, root: str, split: str = 'train', transform=None, preload=False):
#         super().__init__(root, split, transform, preload)
#
#         self.ignore_label = 0
#
#     def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
#         image, label = self.pairs[index]
#
#         image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
#         label = skimage.io.imread(label) % 100
#         print(np.unique(label))
#         image = torch.tensor(image, dtype=torch.uint8).permute(2, 0, 1)
#         label = torch.tensor(label.astype(np.long), dtype=torch.long)
#
#         if self.transform:
#             image, label = self.transform(image, label)
#         return image, torch.squeeze(label.long())


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import multiprocessing as mp
    _dataset = HTHT2022Coarse('../../data/HTHT2022', 'train', preload=False)
    _dataloader = DataLoader(_dataset, batch_size=16, num_workers=mp.cpu_count(), drop_last=True,
                             pin_memory=True)
    for _i, _l in _dataloader:
        print(_i.size(), _l.size())
        # break
