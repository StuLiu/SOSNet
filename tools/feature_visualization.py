"""
@Project : semantic-segmentation
@File    : visualization.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2022/7/19 上午10:57
@e-mail  : 1183862787@qq.com
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torch.nn import init
from torchvision import io
from semseg.models.backbones.mobilenetv3 import MobileNetV3
from semseg.augmentations import get_val_augmentation
import numpy as np
import mmcv


def show_features(f_tensor):
    f_np = np.squeeze(f_tensor.detach().cpu().numpy())
    for img_gray in f_np:
        cv2.imshow('feature', img_gray)
        q = cv2.waitKey(0)
        if q == (ord('q') & 0xff):
            break
    return


if __name__ == '__main__':
    model = MobileNetV3('large')
    model.load_state_dict(torch.load('../checkpoints/backbones/mobilenet/mobilenetv3_large.pth',
                                     map_location='cpu'), strict=False)
    model.train()
    model.cuda()
    aug = get_val_augmentation([2160, 3840])
    # img_rgb_tensor = io.read_image('../data/UAVid2020_mm/img_dir/train/seq1_000700.png')
    img_rgb_tensor = io.read_image('../assests/vaihingen_area3.png')[:-1,:,:]
    # img_bgr = mmcv.imread('../assests/vaihingen_area3.png')
    # img_rgb_tensor = io.read_image('../data/ISPRS_DATA/Vaihingen2/img_dir/train/area1_0_0_512_512.png')
    img_rgb_tensor = aug(img_rgb_tensor, img_rgb_tensor[0:1,:,:])[0].unsqueeze(dim=0).cuda()

    # _x = torch.randn(1, 3, 512, 512)
    _outs = model(img_rgb_tensor)

    for y in _outs:
        show_features(y)

    from semseg.utils.utils import model_summary, init_logger

    init_logger()
    model_summary(model, (1, 3, 224, 224))

    # from fvcore.nn import flop_count_table, FlopCountAnalysis
    # print(flop_count_table(FlopCountAnalysis(model, _x.cuda())))
