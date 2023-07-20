import os.path
import logging
import torch
import argparse
import yaml
import math
from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import DataLoader
from torch.nn import functional as F
from semseg.models import *
from semseg.datasets import *
from semseg.augmentations import get_val_augmentation
from semseg.metrics import Metrics
from semseg.utils.utils import setup_cudnn, count_parameters
from semseg.utils.utils import compute_miou_s_l, init_logger


@torch.no_grad()
def evaluate(model, dataloader, device):
    print('Evaluating...')
    model.eval()
    metrics = Metrics(dataloader.dataset.n_classes, dataloader.dataset.ignore_label, device)

    iter = 0
    for images, labels in tqdm(dataloader):
        images = images.cuda()
        labels = labels.cuda()
        preds = model(images).softmax(dim=1)
        metrics.update(preds, labels)
        iter += 1
        # if iter >= 300:
        #     break

    ious, miou = metrics.compute_iou()
    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()
    oa = metrics.compute_oa()
    return acc, macc, f1, mf1, ious, miou, oa


@torch.no_grad()
def evaluate_msf(model, dataloader, device, scales, flip):
    model.eval()

    n_classes = dataloader.dataset.n_classes
    metrics = Metrics(n_classes, dataloader.dataset.ignore_label, device)

    for images, labels in tqdm(dataloader):
        labels = labels.cuda()
        B, H, W = labels.shape
        scaled_logits = torch.zeros(B, n_classes, H, W).cuda()
        for scale in scales:
            new_H, new_W = int(scale * H), int(scale * W)
            new_H, new_W = int(math.ceil(new_H / 32)) * 32, int(math.ceil(new_W / 32)) * 32
            scaled_images = F.interpolate(images, size=(new_H, new_W), mode='bilinear', align_corners=True)
            scaled_images = scaled_images.cuda()
            logits = model(scaled_images)
            logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=True)
            scaled_logits += logits.softmax(dim=1)

            if flip:
                scaled_images = torch.flip(scaled_images, dims=(3,))
                logits = model(scaled_images)
                logits = torch.flip(logits, dims=(3,))
                logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=True)
                scaled_logits += logits.softmax(dim=1)

        metrics.update(scaled_logits, labels)

    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()
    ious, miou = metrics.compute_iou()
    oa = metrics.compute_oa()
    return acc, macc, f1, mf1, ious, miou, oa


def main(cfg, args):
    device = torch.device(cfg['DEVICE'])

    eval_cfg = cfg['EVAL']
    transform = get_val_augmentation(eval_cfg['IMAGE_SIZE'])

    postfix_dir = f'_so_{args.min_area}_{args.max_area}'
    if args.max_area == -1:
        postfix_dir = ''
    dataset = eval(cfg['DATASET']['NAME'])(cfg['DATASET']['ROOT'], args.val, transform, postfix_dir=postfix_dir)
    dataloader = DataLoader(dataset, 1, num_workers=1, pin_memory=True)

    model_path = Path(eval_cfg['MODEL_PATH'])
    if not model_path.exists():
        model_path = Path(cfg['SAVE_DIR']) / \
                     f"{cfg['MODEL']['NAME']}_{cfg['MODEL']['BACKBONE']}_{cfg['DATASET']['NAME']}.pth"
    print(f"Evaluating {model_path}...")

    model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], dataset.n_classes)
    model.load_state_dict(torch.load(str(model_path), map_location='cpu'))
    model = model.cuda()

    print(f"{cfg['MODEL']['NAME']} parameters: {count_parameters(model)}MB")

    if eval_cfg['MSF']['ENABLE']:
        acc, macc, f1, mf1, ious, miou, oa = evaluate_msf(model, dataloader, device,
                                                      eval_cfg['MSF']['SCALES'], eval_cfg['MSF']['FLIP'])
    else:
        acc, macc, f1, mf1, ious, miou, oa = evaluate(model, dataloader, device)
    miou_s, miou_l = compute_miou_s_l(ious, dataset.SMALL_OBJECT, cfg['DATASET']['IGNORE_LABEL'])
    table = {
        'Class': list(dataset.CLASSES) + ['Mean'],
        'IoU': ious + [miou],
        'F1': f1 + [mf1],
        'Acc': acc + [macc]
    }

    logging.info(tabulate(table, headers='keys'))
    logging.info(f'overall accuracy = {oa}')
    logging.info(f'miou_s = {miou_s}, miou_l = {miou_l}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/helen.yaml')
    parser.add_argument('--val', type=str, default='val')
    parser.add_argument('--min-area', type=int, default=0)
    parser.add_argument('--max-area', type=int, default=-1)
    _args = parser.parse_args()

    with open(_args.cfg) as f:
        _cfg = yaml.load(f, Loader=yaml.SafeLoader)

    _save_dir = os.path.join(os.path.dirname(_args.cfg))
    os.makedirs(_save_dir, exist_ok=True)

    init_logger(logger_name=None, log_dir=os.path.join(_save_dir, 'logs'), log_level=logging.INFO)
    setup_cudnn()
    main(_cfg, _args)
