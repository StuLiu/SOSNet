import torch
import argparse
import time
from fvcore.nn import flop_count_table, FlopCountAnalysis
from semseg.models import *


def main(
    model_name: str,
    backbone_name: str,
    image_size: list,
    num_classes: int,
    device: str,
):
    device = torch.device('cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    inputs = torch.randn(1, 3, *image_size).to(device)
    model = eval(model_name)(backbone_name, num_classes)
    model = model.to(device)
    model.eval()
    print(model.training)

    print(flop_count_table(FlopCountAnalysis(model, inputs)))

    total_time = 0.0
    for _ in range(10):
        tic = time.perf_counter()
        model(inputs)
        toc = time.perf_counter()
        total_time += toc - tic
    total_time /= 10

    # from semseg.utils.utils import model_summary, init_logger
    #
    # init_logger()
    # model_summary(model, (1, 3, *image_size))
    print(f"Inference time: {total_time*1000:.2f}ms")
    print(f"FPS: {1/total_time}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default='FastSCNN')
    parser.add_argument('--backbone-name', type=str, default='MobileNetV3-large')
    # parser.add_argument('--backbone-name', type=str, default=None)
    parser.add_argument('--image-size', type=list, default=[512, 512])
    parser.add_argument('--num-classes', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    main(args.model_name, args.backbone_name, args.image_size, args.num_classes, args.device)