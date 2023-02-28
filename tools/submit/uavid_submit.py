"""
@Project : semantic-segmentation 
@File    : uavid_submit.py
@IDE     : PyCharm 
@Author  : Wang Liu
@Date    : 2022/6/8 下午8:39
@e-mail  : 1183862787@qq.com
"""

import os
import os.path as osp
import shutil


def transfer(dir_path='../../output/test_results', out_dir='./uavid_submit'):
    os.makedirs(out_dir, exist_ok=True)
    imgs = os.listdir(dir_path)
    for img_name in imgs:
        if img_name.endswith('.png'):
            seq_dir, basename = img_name.split('_')
            seq_dir = osp.join(out_dir, seq_dir, 'Labels')
            os.makedirs(seq_dir, exist_ok=True)
            shutil.copy(osp.join(dir_path, img_name), osp.join(seq_dir, basename))


if __name__ == '__main__':
    # transfer(dir_path='../../output/test_results', out_dir='./submit_SOSNet_mbv3l_soa_epoch100')
    transfer(dir_path='../../output_ablation/UperNet/uavid2020/test_results', out_dir='./UperNet')
