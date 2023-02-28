import os
import torch
import argparse
import yaml
import time
import math
import logging
import multiprocessing as mp
from tabulate import tabulate
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, RandomSampler
from torch import distributed as dist
from semseg.models import *
from semseg.datasets import *
from semseg.augmentations import get_train_augmentation, get_val_augmentation
from semseg.losses import get_loss, DiceBCELoss
from semseg.schedulers import get_scheduler
from semseg.optimizers import get_optimizer
from semseg.utils.utils import compute_miou_s_l, fix_seeds, setup_cudnn, cleanup_ddp, \
    setup_ddp, init_logger, model_summary
from semseg.models.modules.auxiliary import SmallObjectMask, EdgeMask
from semseg.models.modules.sos import HierarchicalLoss, HierarchicalSegLoss
from val import evaluate


def main(cfg, gpu, save_dir):
    start = time.time()
    best_mIoU = 0.0
    # num_workers = mp.cpu_count()
    num_workers = 8
    device = torch.device(cfg['DEVICE'])
    # torch.cuda.set_device(0)
    train_cfg, eval_cfg = cfg['TRAIN'], cfg['EVAL']
    dataset_cfg, model_cfg = cfg['DATASET'], cfg['MODEL']
    loss_cfg, optim_cfg, sched_cfg = cfg['LOSS'], cfg['OPTIMIZER'], cfg['SCHEDULER']
    epochs, lr = train_cfg['EPOCHS'], optim_cfg['LR']

    traintransform = get_train_augmentation(size=train_cfg['IMAGE_SIZE'],
                                            seg_fill=dataset_cfg['IGNORE_LABEL'],
                                            h_flip=dataset_cfg['H_FLIP'],
                                            v_flip=dataset_cfg['V_FLIP'])
    valtransform = get_val_augmentation(eval_cfg['IMAGE_SIZE'])

    trainset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'train', traintransform)
    valset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], args.val, valtransform)

    model = eval(model_cfg['NAME'])(model_cfg['BACKBONE'], trainset.n_classes)
    model.init_pretrained(model_cfg['PRETRAINED'])
    model_summary(model)
    model = model.to(device)
    # model.freeze_backbone()

    trans_so = SmallObjectMask(trainset.SMALL_OBJECT).to(device)
    trans_edge = EdgeMask().to(device)

    if train_cfg['DDP']:
        sampler = DistributedSampler(trainset, dist.get_world_size(), dist.get_rank(), shuffle=True)
        model = DDP(model, device_ids=[gpu])
    else:
        sampler = RandomSampler(trainset)

    trainloader = DataLoader(trainset, batch_size=train_cfg['BATCH_SIZE'], num_workers=num_workers, drop_last=True,
                             pin_memory=True, sampler=sampler)
    valloader = DataLoader(valset, batch_size=1, num_workers=1, pin_memory=True)

    iters_per_epoch = len(trainset) // train_cfg['BATCH_SIZE']
    # class_weights = trainset.class_weights.to(device)
    loss_edge_fn = DiceBCELoss()
    loss_seg_fn = get_loss(loss_cfg['NAME'], dataset_cfg['IGNORE_LABEL'], None)
    loss_hier_fn = HierarchicalLoss(trainset.n_classes, trainset.SMALL_OBJECT)
    # loss_hier_fn.load_state_dict(torch.load(os.path.join('checkpoints/hier_loss', f'{dataset_cfg["NAME"]}.pth'),
    #                                         map_location='cpu'), strict=True)
    loss_hier_fn = loss_hier_fn.to(device)
    loss_hierSeg_fn = HierarchicalSegLoss(
        loss_seg_fn=loss_seg_fn,
        loss_hier_fn=loss_hier_fn,
        ignore_label=dataset_cfg['IGNORE_LABEL'],
        is_hier=_cfg['args']['hier'],
        is_soem=_cfg['args']['soem'],
        ratio=_cfg['args']['ratio'],
    )
    # soem = SOEM(ignore_label=dataset_cfg['IGNORE_LABEL'], ratio=0.2, threshold=0.5)
    optimizer = get_optimizer(model, optim_cfg['NAME'], lr, optim_cfg['WEIGHT_DECAY'])
    scheduler = get_scheduler(sched_cfg['NAME'],
                              optimizer,
                              epochs * iters_per_epoch,
                              sched_cfg['POWER'],
                              iters_per_epoch * sched_cfg['WARMUP'],
                              sched_cfg['WARMUP_RATIO'])
    scaler = GradScaler(enabled=train_cfg['AMP'])
    writer = SummaryWriter(os.path.join(save_dir, 'logs'))
    for epoch in range(epochs):
        # continue
        model.train()
        iter_local, loss_local_total, loss_local_seg, loss_local_edge = 0, 0.0, 0.0, 0.0
        if train_cfg['DDP']:
            sampler.set_epoch(epoch)

        pbar = tqdm(trainloader, total=iters_per_epoch, desc='', ncols=180)
        for img, lbl in pbar:
            optimizer.zero_grad(set_to_none=True)

            img = img.to(device)
            lbl_seg = lbl.to(device)
            lbl_so = trans_so(lbl_seg)
            with autocast(enabled=train_cfg['AMP']):
                logits_seg, logits_so, logits_edge = model(img)
                loss_hierSeg = loss_hierSeg_fn(logits_seg, logits_so, lbl_seg, lbl_so)
                if _cfg['args']['edge']:
                    lbl_edge = trans_edge(lbl_seg)
                    loss_edge = loss_edge_fn(torch.squeeze(logits_edge), lbl_edge.float())
                    loss = loss_hierSeg + 0.1 * loss_edge
                    loss_local_edge += loss_edge.item()
                else:
                    loss = loss_hierSeg
                if torch.isnan(loss).item():
                    print(f'epoch-{epoch + 1}-iter-{iter_local} loss is nan!')
                    continue
            loss.requires_grad_(True)
            scaler.scale(loss).backward()
            # for name, parms in model.named_parameters():
            #     print('-->name:', name)
            #     print('-->para:', parms)
            #     print('-->grad_requirs:', parms.requires_grad)
            #     print('-->grad_value:', parms.grad)
            #     print("===")
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            torch.cuda.synchronize()

            lr = scheduler.get_lr()
            lr = sum(lr) / len(lr)
            loss_local_total += loss.item()
            loss_local_seg += loss_hierSeg.item()
            iter_local += 1
            log_str = f"Epoch: [{epoch + 1}/{epochs}] " \
                      f"Iter: [{iter_local}/{iters_per_epoch}] " \
                      f"LR: {lr:.8f} " \
                      f"Loss total: {loss_local_total / (iter_local + 1e-7):.8f}\t" \
                      f"Loss seg  : {loss_local_seg   / (iter_local + 1e-7):.8f}\t" \
                      f"Loss edge : {loss_local_edge  / (iter_local + 1e-7):.8f}\t"
            pbar.set_description(log_str)
            if iter_local >= cfg['TRAIN']['MAX_INERITER']:
                # if iter_local >= 20:
                break

        logging.info(f"Epoch: [{epoch + 1}/{epochs}]\tLR: {lr:.8f}\t"
                     f"Loss total: {loss_local_total / (iter_local + 1e-7):.8f}\t"
                     f"Loss seg  : {loss_local_seg   / (iter_local + 1e-7):.8f}\t"
                     f"Loss edge : {loss_local_edge  / (iter_local + 1e-7):.8f}\t")
        loss_local_total /= iter_local
        writer.add_scalar('train/lr', lr, epoch)
        writer.add_scalar('train/loss_total', loss_local_total, epoch)
        writer.add_scalar('train/loss_seg',   loss_local_seg, epoch)
        writer.add_scalar('train/loss_edge',  loss_local_edge, epoch)
        torch.cuda.empty_cache()
        pbar.close()
        if train_cfg['EVAL_INTERVAL'] > epochs: continue
        # evaluate the model
        if (epoch + 1) % train_cfg['EVAL_INTERVAL'] == 0 or (epoch + 1) == epochs:
            # val evaluation
            acc, macc, f1, mf1, ious, miou, oa = evaluate(model, valloader, device)
            table = {
                'Class': list(trainset.CLASSES) + ['Mean'],
                'IoU': ious + [miou],
                'F1': f1 + [mf1],
                'Acc': acc + [macc]
            }
            miou_s, miou_l = compute_miou_s_l(ious, trainset.SMALL_OBJECT, dataset_cfg['IGNORE_LABEL'])
            logging.info(f'\n{tabulate(table, headers="keys")}\n'
                         f'overall accuracy = {oa}\n'
                         f'miou_s = {miou_s}, miou_l = {miou_l}')
            writer.add_scalar('val/mIoU', miou, epoch)
            if cfg['args']['save_epoch']:
                torch.save(model.module.state_dict() if train_cfg['DDP'] else model.state_dict(),
                           os.path.join(save_dir,
                                        f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch-"
                                        f"{epoch + 1}.pth"))
            if miou > best_mIoU:
                best_mIoU = miou
                torch.save(model.module.state_dict() if train_cfg['DDP'] else model.state_dict(),
                           os.path.join(save_dir,
                                        f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_best.pth"))
            # train evaluation
            train_miou = '-'
            if cfg['EVAL']['TRAIN_SET']:
                train_miou = evaluate(model, trainloader, device)[-2]
                writer.add_scalar('train/mIoU', train_miou, epoch)
            # do logging
            logging.info(f"Current train mIoU: {train_miou}, val mIoU: {miou}, Best mIoU: {best_mIoU}")
        pass
    torch.save(model.module.state_dict() if train_cfg['DDP'] else model.state_dict(),
               os.path.join(save_dir, f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_last.pth"))
    writer.close()

    end = time.gmtime(time.time() - start)

    table = [
        ['Best mIoU', f"{best_mIoU:.2f}"],
        ['Total Training Time', time.strftime("%H:%M:%S", end)]
    ]
    logging.info(tabulate(table, numalign='right'))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


if __name__ == '__main__':

    # import matplotlib.pyplot as plt
    # t=[]
    # for i in range(1, 501):
    #     print(i, exp_decay(1, i, 500))
    #
    # plt.plot(range(1,501), [exp_decay(1, i, 500) for i in range(1, 501)])
    # plt.show()
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/camvid.yaml', help='Configuration file to use')
    parser.add_argument('--save_epoch', type=str2bool, default=False, help='Configuration file to use')
    parser.add_argument('--val', type=str, default='val', help='Configuration file to use')
    parser.add_argument('--edge', type=str2bool, default=False, help='if use edge supervision')
    parser.add_argument('--hier', type=str2bool, default=True, help='if use the hierarchical loss')
    parser.add_argument('--soem', type=str2bool, default=True, help='if use the small object example mining alg')
    parser.add_argument('--ratio', type=float, default=0.1, help='ratio for soem algorithm')
    # parser.add_argument('--thre', type=float, default=0.5, help='threshold for soem algorithm')
    args = parser.parse_args()

    with open(args.cfg) as f:
        _cfg = yaml.load(f, Loader=yaml.SafeLoader)
        _cfg['args'] = {
            'cfg': args.cfg,
            'save_epoch': args.save_epoch,
            'val': args.val,
            'edge': args.edge,
            'hier': args.hier,
            'soem': args.soem,
            'ratio': args.ratio,
            # 'thre': args.thre,
        }
        # fix_seeds(3407)
    fix_seeds(int(time.time()))
    setup_cudnn()
    _gpu = setup_ddp()
    _save_dir = os.path.join(Path(_cfg['SAVE_DIR']),
                             f'{_cfg["MODEL"]["NAME"]}_'
                             f'{_cfg["DATASET"]["NAME"]}_'
                             f'{time.strftime("%Y%m%d%H%M%S", time.localtime())}')
    os.makedirs(_save_dir, exist_ok=True)

    init_logger(logger_name=None, log_dir=os.path.join(_save_dir, 'logs'), log_level=logging.INFO)
    logging.info(yaml.dump(_cfg))
    with open(os.path.join(_save_dir, 'config.yaml'), 'w') as sf:
        sf.write(yaml.dump(_cfg))
    main(_cfg, _gpu, _save_dir)
    cleanup_ddp()
