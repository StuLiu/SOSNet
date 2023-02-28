import torch
import logging
import os.path as osp

from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io
from pathlib import Path
from typing import Tuple
from glob import glob


class CamVid(Dataset):
    """
    num_classes: 11
    all_num_classes: 31

    id	class	    r	g	b
    0   Sky	        128	128	128
    1	Building	128	0	0
    2   Column_Pole	192	192	128
    3   Road	    128	64	128
    4   Sidewalk	0	0	192
    5   Tree	    128	128	0
    6   SignSymbol	192	128	128
    7   Fence	    64	64	128
    8   Car	        64	0	128
    9   Pedestrian	64	64	0
    10  Bicyclist	0	128	192
    11  unknown     0   0   0
    """

    CLASSES = ['Sky', 'Building', 'Column_Pole', 'Road', 'Sidewalk',
               'Tree', 'SignSymbol', 'Fence', 'Car', 'Pedestrian',
               'Bicyclist', 'unknown']
    PALETTE = torch.tensor(
        [[128, 128, 128], [128, 0, 0], [192, 192, 128], [128, 64, 128], [0, 0, 192],
         [128, 128, 0], [192, 128, 128], [64, 64, 128], [64, 0, 128], [64, 64, 0],
         [0, 128, 192], [0, 0, 0]])

    SMALL_OBJECT = [2, 6, 9, 10]

    def __init__(self, root: str, split: str = 'train', transform=None, preload=False) -> None:
        super().__init__()
        assert split in ['train', 'val', 'test']
        self.split = split
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        # #######################################################
        # self.ignore_label = -1
        self.ignore_label = 11
        self.preload = preload
        self.pairs = []

        imgs = glob(osp.join(root, self.split) + '/*.png')
        for img_path in imgs:
            lbl_path = img_path.replace(self.split, self.split + 'annot')
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


class CamVid729x969(Dataset):
    """
    num_classes: 11
    all_num_classes: 31

    id	class	    r	g	b
    0	Bicyclist	0	128	192
    1	Building	128	0	0
    2	Car	        64	0	128
    3	Column_Pole	192	192	128
    4	Fence	    64	64	128
    5	Pedestrian	64	64	0
    6	Road	    128	64	128
    7	Sidewalk	0	0	192
    8	SignSymbol	192	128	128
    9	Sky	        128	128	128
    10	Tree	    128	128	0
    11	background	0   0   0
    12  unknown     0   0   0

    """
    CLASSES = ['Bicyclist', 'Building', 'Car', 'Column_Pole', 'Fence', 'Pedestrian', 'Road', 'Sidewalk', 'SignSymbol',
               'Sky', 'Tree', 'unkonw']
    CLASSES_ALL = ['Wall', 'Animal', 'Archway', 'Bicyclist', 'Bridge', 'Building', 'Car', 'CarLuggage', 'Child', 'Pole',
                   'Fence', 'LaneDrive', 'LaneNonDrive', 'MiscText', 'Motorcycle/Scooter', 'OtherMoving',
                   'ParkingBlock', 'Pedestrian', 'Road', 'RoadShoulder', 'Sidewalk', 'SignSymbol', 'Sky',
                   'SUV/PickupTruck', 'TrafficCone', 'TrafficLight', 'Train', 'Tree', 'Truck/Bus', 'Tunnel',
                   'VegetationMisc']
    PALETTE = torch.tensor(
        [[0, 128, 192], [128, 0, 0], [64, 0, 128], [192, 192, 128], [64, 64, 128], [64, 64, 0],
         [128, 64, 128], [0, 0, 192], [192, 128, 128], [128, 128, 128], [128, 128, 0], [0, 0, 0]])
    PALETTE_ALL = torch.tensor(
        [[64, 192, 0], [64, 128, 64], [192, 0, 128], [0, 128, 192], [0, 128, 64], [128, 0, 0], [64, 0, 128],
         [64, 0, 192], [192, 128, 64], [192, 192, 128], [64, 64, 128], [128, 0, 192], [192, 0, 64], [128, 128, 64],
         [192, 0, 192], [128, 64, 64], [64, 192, 128], [64, 64, 0], [128, 64, 128], [128, 128, 192], [0, 0, 192],
         [192, 128, 128], [128, 128, 128], [64, 128, 192], [0, 0, 64], [0, 64, 64], [192, 64, 128], [128, 128, 0],
         [192, 128, 192], [64, 0, 64], [192, 192, 0]])

    SMALL_OBJECT = [0, 3, 5, 8]

    def __init__(self, root: str, split: str = 'train', transform=None, preload=False) -> None:
        super().__init__()
        assert split in ['train', 'val', 'test']
        self.split = split
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        # #######################################################
        # self.ignore_label = -1
        self.ignore_label = 11
        self.preload = preload
        self.pairs = []

        imgs = glob(osp.join(root, self.split) + '/*.png')
        for img_path in imgs:
            lbl_path = img_path.replace(self.split, self.split + '_labels').replace('.png', '_L_ids.png')
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
    #
    # visualize_dataset_sample(CamVid2, '../../data/CamVid')
    _dataset = CamVid('../../data/CamVid', 'train', preload=False)
    print(len(_dataset))
    # for _i, _l in _dataset:
    #     print(_i.size(), _l.size())
