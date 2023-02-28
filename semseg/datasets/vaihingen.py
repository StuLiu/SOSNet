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


class Vaihingen(Dataset):
    """
    num_classes: 6, ignore index is 5.

    CLASSES = ('impervious_surface', 'building', 'low_vegetation', 'tree',
               'car', 'clutter')

    PALETTE = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0],
               [255, 255, 0], [255, 0, 0]]

    """
    CLASSES = ['impervious_surface', 'building', 'low_vegetation', 'tree', 'car', 'clutter']

    PALETTE = torch.tensor([[255, 255, 255], [0, 0, 255], [0, 255, 255],
                            [0, 255, 0], [255, 255, 0], [255, 0, 0]])
    SMALL_OBJECT = [4]

    def __init__(self, root: str, split: str = 'train', transform=None) -> None:
        super().__init__()
        assert split in ['train', 'val', 'test']
        self.split = 'val' if split == 'test' else split
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 5       # ignore_label when calculating metrics
        img_dir = osp.join(root, 'img_dir', self.split)
        self.files = glob(img_dir + '/*.png')
        assert len(self.files) > 0, f"No images found in {root}"
        logging.info(f"Found {len(self.files)} {split} images.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        img_path = str(self.files[index])
        lbl_path = str(self.files[index].replace('img_dir', 'ann_dir'))

        image = io.read_image(img_path)
        label = io.read_image(lbl_path)

        if self.transform:
            image, label = self.transform(image, label)

        return image, torch.squeeze(label.long())


class Vaihingen2(Dataset):
    """
    num_classes: 6

    CLASSES = ('impervious_surface', 'building', 'low_vegetation', 'tree',
               'car', 'clutter')

    PALETTE = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0],
               [255, 255, 0], [255, 0, 0]]

    """
    CLASSES = ['impervious_surface', 'building', 'low_vegetation', 'tree', 'car', 'clutter']

    PALETTE = torch.tensor([[255, 255, 255], [0, 0, 255], [0, 255, 255],
                            [0, 255, 0], [255, 255, 0], [255, 0, 0]])

    META = {
        'train': [1, 3, 5, 7, 11, 13, 15, 17, 21, 23, 26, 28, 30, 32, 34, 37],
        'val': [2, 4, 6, 8, 10, 12, 14, 16, 20, 22, 24, 27, 29, 31, 33, 35, 38],
        'test': [2, 4, 6, 8, 10, 12, 14, 16, 20, 22, 24, 27, 29, 31, 33, 35, 38],
        'prefix': 'top_mosaic_09cm_area',
    }

    def __init__(self, root: str, split: str = 'train', transform=None) -> None:
        super().__init__()
        assert split in ['train', 'val', 'test']
        self.split = split
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.files = []

        img_dir = osp.join(root, 'top')
        lbl_dir = osp.join(root, 'labels')
        assert osp.isdir(img_dir) and osp.isdir(lbl_dir), f'no such dir:{img_dir} or {lbl_pdir}'
        for idx in self.META[split]:
            self.files.append({
                'image': osp.join(img_dir, f'top_mosaic_09cm_area{idx}.tif'),
                'label': osp.join(lbl_dir, f'top_mosaic_09cm_area{idx}.tif')
            })
        assert len(self.files) > 0, f"No images found in {root}"
        logging.info(f"Found {len(self.files)} {split} images.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        img_path = str(self.files[index]['image'])
        lbl_path = str(self.files[index]['label'])

        image = Image.open(img_path)
        label = Image.open(lbl_path)

        if self.transform:
            image, label = self.transform(image, label)

        return image, torch.squeeze(label.long())


if __name__ == '__main__':
    from semseg.utils.visualize import visualize_dataset_sample

    visualize_dataset_sample(Vaihingen, '../../data/ISPRS_DATA/Vaihingen2')
