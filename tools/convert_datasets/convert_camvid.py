"""
@Project : DDRNet_pytorch 
@File    : convert_camvid.py
@IDE     : PyCharm 
@Author  : Wang Liu
@Date    : 2022/4/25 下午3:00
@e-mail  : 1183862787@qq.com
"""
import csv
import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image


def read_labels(labelcsv='data/CamVid/class_dict.csv'):
    with open(labelcsv, 'r') as f:
        f_csv = csv.reader(f)
        head = next(f_csv)
        print(head)
        id = 0
        id2name = {}
        id2color = {}
        for line in f_csv:
            # print(line)
            if line[-1] == '1':
                id2name[id] = line[0]
                id2color[id] = [int(e) for e in line[1:]]
                id += 1
    with open(labelcsv.replace('.csv', '.txt'), 'w') as f:
        f.write(f'id\tclass\tr\tg\tb\n')
        for id in range(len(id2name.keys())):
            color = id2color[id]
            line = f'{id}\t{id2name[id]}\t{color[0]}\t{color[1]}\t{color[2]}\n'
            f.write(line)
        f.write(f'{len(id2name.keys())}\tunkonw\t0\t0\t0\n')

    return id2name, id2color


def convert_color2id(label_root='data/CamVid/train_labels'):
    color_imgs = glob(label_root + '/*L.png')
    color_imgs.sort()
    id2name, id2color = read_labels('data/CamVid/class_dict.csv')
    print(id2name)
    print(id2color)
    cls_num = len(id2name.keys())
    for color_img in tqdm(color_imgs):
        img_rgb = cv2.cvtColor(cv2.imread(color_img), cv2.COLOR_BGR2RGB).astype(np.int32)
        img_ids = np.ones((img_rgb.shape[0], img_rgb.shape[1])) * cls_num
        img_rgbvalues = img_rgb[:, :, 0] * 65536 + img_rgb[:, :, 1] * 256 + img_rgb[:, :, 2]
        for id, color in id2color.items():
            img_ids = np.where(img_rgbvalues[:, :] == color[0] * 65536 + color[1] * 256 + color[2], id, img_ids)
        img_ids = img_ids.astype(np.uint8)
        # img_ids = np.expand_dims(img_ids, axis=2)
        # img_ids = np.concatenate([img_ids, img_ids, img_ids], axis=2)
        # print(np.unique(img_ids))
        # cv2.imshow('as', img_ids)
        # cv2.waitKey(0)
        cv2.imwrite(color_img.replace('.png', '_ids.png'), img_ids)
        # break
    pass


def make_list(data_root='data/CamVid', save_dir='data/CamVid/list/', mode='train'):
    imgs_dir = os.path.join(data_root, mode)
    # labels_dir = os.path.join(data_root, f'{mode}_labels')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{mode}.lst')
    img_names = os.listdir(imgs_dir)
    img_names.sort()
    with open(save_path, 'w') as f:
        for img_name in img_names:
            f.write(f'{mode}/{img_name}\t{mode}_labels/{img_name.replace(".png", "_L_ids.png")}\n')


def lblshow(lblidpath='../../data/CamVid/valannot/0016E5_08051.png'):
    img = Image.open(lblidpath).convert('P')
    colormap = np.array([[128, 128, 128], [128, 0, 0], [192, 192, 128], [128, 64, 128], [0, 0, 192],
         [128, 128, 0], [192, 128, 128], [64, 64, 128], [64, 0, 128], [64, 64, 0],
         [0, 128, 192], [0, 0, 0]], dtype=np.uint8)
    img.putpalette(colormap.flatten())
    img.save('test_color.png')



if __name__ == '__main__':
    # # read_labels()
    # convert_color2id('data/CamVid/train_labels')
    # convert_color2id('data/CamVid/val_labels')
    # convert_color2id('data/CamVid/test_labels')
    # make_list(mode='train')
    # make_list(mode='test')
    # make_list(mode='val')

    # a = cv2.imread('data/CamVid/testannot/0001TP_008550.png')
    # a = cv2.imread('data/CamVid/testannot/Seq05VD_f00300.png')
    # print(a.shape)
    # a = np.where(a==7, 255, a)
    # cv2.imshow('1', a)
    # cv2.waitKey(0)
    lblshow()
