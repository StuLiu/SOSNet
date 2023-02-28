import os
import os.path as osp

import cv2
import torch
import logging
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io
from pathlib import Path
from typing import Tuple
from glob import glob


class ISAID(Dataset):
    """
    num_classes: 16, ignore index is 255 (impervious_surface).
    """
    CLASSES = ['background', 'ship', 'store_tank', 'baseball_diamond', 'tennis_court',
               'basketball_court', 'Ground_Track_Field', 'Bridge', 'Large_Vehicle', 'Small_Vehicle',
               'Helicopter', 'Swimming_pool', 'Roundabout', 'Soccer_ball_field', 'plane',
               'Harbor']

    PALETTE = torch.tensor([[0, 0, 0], [0, 0, 63], [0, 63, 63], [0, 63, 0], [0, 63, 127],
                            [0, 63, 191], [0, 63, 255], [0, 127, 63], [0, 127, 127], [0, 0, 127],
                            [0, 0, 191], [0, 0, 255], [0, 191, 127], [0, 127, 191], [0, 127, 255],
                            [0, 100, 155]])

    SMALL_OBJECT = [1, 2, 3, 7, 8, 9, 10, 11, 12, 14, 15]

    def __init__(self, root: str, split: str = 'train', transform=None, preload=False) -> None:
        super().__init__()
        assert split in ['train', 'val']
        self.split = split
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255
        self.preload = preload
        self.pairs = []

        imgs = glob(osp.join(root, 'img_dir', self.split) + '/*.png')
        imgs.sort()
        for img_path in imgs:
            lbl_path = img_path.replace('img_dir', 'ann_dir').replace('.png', '_instance_color_RGB.png')
            data_pair = [
                io.read_image(img_path) if self.preload else img_path,
                io.read_image(lbl_path) if self.preload else lbl_path,
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
            label = io.read_image(label)

        if self.transform:
            image, label = self.transform(image, label)
        return image, torch.squeeze(label.long())


if __name__ == '__main__':
    # from semseg.utils.visualize import visualize_dataset_sample
    # visualize_dataset_sample(ISAID, '../../data/iSAID2')


    from torch.utils.data import DataLoader
    import numpy as np


    train_dataset = ISAID('../../data/iSAID2', split='train')
    val_dataset = ISAID('../../data/iSAID2', split='val')
    print(f'train size={len(train_dataset)}, val size={len(val_dataset)}')

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    for _img, _lbl in train_dataloader:
        print(_img.cpu().numpy().shape, _lbl.cpu().numpy().shape, np.unique(_lbl.cpu().numpy()))
        cc = _lbl.cpu().numpy().squeeze().astype(np.uint8)
        cv2.imshow('img', _img.cpu().numpy().squeeze().transpose((1,2,0)))
        # cv2.imshow('lbl', np.array([cc, cc, cc]).transpose((1, 2, 0)).astype(np.uint8))
        cv2.imshow('lbl', cc)
        cv2.waitKey(0)
