import torch
import numpy as np
import random
import time
import os
import functools
import warnings
import logging
import os.path as osp

from pathlib import Path
from torch.backends import cudnn
from torch import nn, Tensor
from torch.autograd import profiler
from typing import Union
from torch import distributed as dist
from tabulate import tabulate
from semseg import models
from torchinfo import summary


def compute_miou_s_l(ious: list, small_obj_list: list, ignore_label: int):
    iou_s, iou_l = list(), list()
    for _id in range(len(ious)):
        if _id == ignore_label or ious[_id] is None:
            continue
        if _id in small_obj_list:
            iou_s += [ious[_id]]
        else:
            iou_l += [ious[_id]]
    return np.round(np.mean(np.array(iou_s)), 2), np.round(np.mean(np.array(iou_l)), 2)


def fix_seeds(seed: int = 3407) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup_cudnn() -> None:
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    cudnn.benchmark = True
    cudnn.deterministic = False


def time_sync() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def get_model_size(model: Union[nn.Module, torch.jit.ScriptModule]):
    tmp_model_path = Path('temp.p')
    if isinstance(model, torch.jit.ScriptModule):
        torch.jit.save(model, tmp_model_path)
    else:
        torch.save(model.state_dict(), tmp_model_path)
    size = tmp_model_path.stat().st_size
    os.remove(tmp_model_path)
    return size / 1e6  # in MB


@torch.no_grad()
def test_model_latency(model: nn.Module, inputs: torch.Tensor, use_cuda: bool = False) -> float:
    with profiler.profile(use_cuda=use_cuda) as prof:
        _ = model(inputs)
    return prof.self_cpu_time_total / 1000  # ms


def count_parameters(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6  # in M


def setup_ddp() -> int:
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(gpu)
        dist.init_process_group('nccl', init_method="env://", world_size=world_size, rank=rank)
        dist.barrier()
    else:
        gpu = 0
    return gpu


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def reduce_tensor(tensor: Tensor) -> Tensor:
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


@torch.no_grad()
def throughput(dataloader, model: nn.Module, times: int = 30):
    model.eval()
    images, _ = next(iter(dataloader))
    images = images.cuda(non_blocking=True)
    B = images.shape[0]
    print(f"Throughput averaged with {times} times")
    start = time_sync()
    for _ in range(times):
        model(images)
    end = time_sync()

    print(f"Batch Size {B} throughput {times * B / (end - start)} images/s")


def show_models():
    model_names = models.__all__
    model_variants = [list(eval(f'models.{name.lower()}_settings').keys()) for name in model_names]

    print(tabulate({'Model Names': model_names, 'Model Variants': model_variants}, headers='keys'))


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"Elapsed time: {elapsed_time * 1000:.2f}ms")
        return value

    return wrapper_timer


def init_logger(logger_name=None, log_dir=None, log_level=logging.INFO):
    warnings.filterwarnings(action='ignore')
    log_dir = osp.join('Logs', str(time.time())) if log_dir is None else log_dir
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name=logger_name)
    logger.setLevel(log_level)  # Log等级总开关

    # 第二步，创建一个handler，用于写入日志文件
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_file_path = osp.join(log_dir, f'{rq}.log')
    file_handler = logging.FileHandler(log_file_path, mode='w')
    stream_handler = logging.StreamHandler()

    file_handler.setLevel(log_level)  # 输出到file的log等级的开关
    stream_handler.setLevel(log_level)  # 输出到console的log等级的开关

    # 第三步，定义handler的输出格式
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # 第四步，将logger添加到handler里面
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


def model_summary(net: torch.nn.Module, input_shape: tuple = (1, 3, 1024, 512)):
    """ Show net detail including parameter cnt and ops cnt.
        Args:
        net: torch model
        input_shape: shape of input tensor, len=4

    Returns:
        None
    """
    logging.info(summary(net, input_shape, verbose=0))
    # logging.info(f'model size={get_model_size(net)}MB')
    # logging.info(f'model parameters={count_parameters(net)}M')
