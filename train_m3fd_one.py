#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train the train_llvip_one.py dual-modal detector on the M3FD split.

这个脚本只适配 M3FD 数据集的 CSV 和类别定义，网络结构、loss、训练流程均复用
rgb_ir/train_llvip_one.py 中的实现：

    - DualYoloV8L 双流 RGB/IR backbone
    - YOLOv8-L 风格 neck + 跨模态 XAttn 融合 neck
    - anchor-free DetectHead + TaskAlignedAssigner + DFL loss

注意：这里不会启用 train_llvip_one.py 中的 VLM output/sample reweight 逻辑；
所有样本权重保持 1.0。

默认使用刚刚生成的：
    /root/WHB/M3FD/m3fd_train_paths.csv
    /root/WHB/M3FD/m3fd_test_paths.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import train_llvip_one as llvip


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 4 卡 24GB 显存上的 M3FD 默认训练配置：
# 已知 imgsz=768、global batch=20（每卡 5）时单卡约 13916MiB / 24564MiB。
# 为尽量吃满显存且保留安全余量，默认改为 global batch=32（每卡 8）；
# 线性估计单卡约 22.3GiB，占用约 90%。
M3FD_DEFAULT_IMGSZ = 768
M3FD_DEFAULT_BATCH = 32
M3FD_BASE_BATCH = 20
M3FD_BASE_LR0 = 0.003
M3FD_DEFAULT_LR0 = M3FD_BASE_LR0 * (M3FD_DEFAULT_BATCH / M3FD_BASE_BATCH)

# M3FD XML 中的类别包括：
#   People, Car, Bus, Lamp, Motorcycle, Truck
# 类别顺序与 M3FD/labels/*.txt 中的 YOLO id 对齐：
#   0 People, 1 Car, 2 Bus, 3 Lamp, 4 Motorcycle, 5 Truck
M3FD_CLASSES = ["people", "car", "bus", "lamp", "motorcycle", "truck"]

M3FD_ALIAS_MAP: Dict[str, Optional[str]] = {
    "people": "people",
    "person": "people",
    "pedestrian": "people",
    "car": "car",
    "bus": "bus",
    "lamp": "lamp",
    "light": "lamp",
    "traffic_light": "lamp",
    "street_light": "lamp",
    "streetlight": "lamp",
    "motorcycle": "motorcycle",
    "motorbike": "motorcycle",
    "bike": "motorcycle",
    "truck": "truck",
    "*": None,
    "": None,
    "unknown": None,
}


def configure_m3fd_classes() -> None:
    """Patch the reused LLVIP module so its XML parser and evaluator use M3FD classes."""
    llvip.ALIAS_MAP = dict(M3FD_ALIAS_MAP)
    llvip.CANONICAL_CLASSES = list(M3FD_CLASSES)
    llvip.CLS2ID = {name: idx for idx, name in enumerate(M3FD_CLASSES)}


def get_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Train train_llvip_one.py's network on the paired M3FD VI/IR split."
    )
    ap.add_argument(
        "--train_csv",
        type=str,
        default=str(PROJECT_ROOT / "M3FD" / "m3fd_train_paths.csv"),
        help="M3FD 训练集 CSV，列为 visible_path,infrared_path,annotation_path",
    )
    ap.add_argument(
        "--val_csv",
        type=str,
        default=str(PROJECT_ROOT / "M3FD" / "m3fd_test_paths.csv"),
        help="M3FD 测试/验证集 CSV，列为 visible_path,infrared_path,annotation_path",
    )
    ap.add_argument("--imgsz", type=int, default=M3FD_DEFAULT_IMGSZ)
    ap.add_argument("--epochs", type=int, default=220)
    ap.add_argument(
        "--batch",
        type=int,
        default=M3FD_DEFAULT_BATCH,
        help="global batch size；DDP 下会自动均分到各 rank。默认 32=4卡每卡8，适配 24GB 显存",
    )
    ap.add_argument("--num_workers", type=int, default=4)

    # ---- YOLOv8-like recipe，针对 M3FD 从 batch20/lr0=0.003 线性缩放到 batch32/lr0=0.0048 ----
    ap.add_argument("--lr0", type=float, default=M3FD_DEFAULT_LR0, help="初始学习率")
    ap.add_argument("--lrf", type=float, default=0.01, help="最终学习率比例 lr_final = lr0 * lrf")
    ap.add_argument("--momentum", type=float, default=0.937)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--warmup_epochs", type=float, default=5.0)
    ap.add_argument("--warmup_momentum", type=float, default=0.8)
    ap.add_argument("--warmup_bias_lr", type=float, default=0.1)
    ap.add_argument("--optimizer", type=str, default="SGD", help="SGD 或 AdamW")
    ap.add_argument("--ema", action="store_true", default=True, help="启用 EMA（默认开启）")
    ap.add_argument("--no-ema", dest="ema", action="store_false", help="关闭 EMA")
    ap.add_argument("--ema_decay", type=float, default=0.9999)
    ap.add_argument("--ema_tau", type=float, default=2000.0)
    ap.add_argument("--lr", type=float, default=None, help="兼容旧参数：等价于 --lr0")

    # ---- evaluation / logging ----
    ap.add_argument("--conf_thres", type=float, default=0.001)
    ap.add_argument("--nms_iou", type=float, default=0.5)
    ap.add_argument(
        "--logdir",
        type=str,
        default=str(PROJECT_ROOT / "rgb_ir" / "runs" / "m3fd" / "0426"),
        help="训练日志和权重输出目录",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_det", type=int, default=300)

    # ---- device / distributed ----
    ap.add_argument("--gpu", type=int, default=0, help="单卡 GPU id；设为 -1 强制 CPU")
    ap.add_argument("--gpus", type=str, default="", help="单进程多 GPU，例如 --gpus 0,1,2；推荐优先用 torchrun/DDP")
    ap.add_argument("--local_rank", type=int, default=-1, help=argparse.SUPPRESS)
    ap.add_argument("--sync_bn", action="store_true", default=True, help="DDP 下启用 SyncBatchNorm（默认开启）")
    ap.add_argument("--no-sync-bn", dest="sync_bn", action="store_false", help="关闭 DDP SyncBatchNorm")

    # ---- augment schedule ----
    ap.add_argument("--close_mosaic", type=int, default=20, help="最后 N 个 epoch 关闭 mosaic/mixup")
    ap.add_argument("--mosaic_prob", type=float, default=0.6)
    ap.add_argument("--mixup_prob", type=float, default=0.0)
    ap.add_argument("--mixup_target_prob", type=float, default=0.04)
    ap.add_argument("--mixup_ramp_start", type=int, default=40)
    ap.add_argument("--mixup_ramp_end", type=int, default=100)
    ap.add_argument("--second_stage_ft", action="store_true", help="最后若干 epoch 关闭强增强并降低学习率")
    ap.add_argument("--ft_epochs", type=int, default=20)
    ap.add_argument("--ft_lr_scale", type=float, default=0.1)
    ap.add_argument("--ft_scale", type=float, default=0.1)
    ap.add_argument("--ft_translate", type=float, default=0.02)

    # ---- amp / auto batch ----
    ap.add_argument("--amp", action="store_true", default=True, help="启用 AMP（默认开启）")
    ap.add_argument("--no-amp", dest="amp", action="store_false", help="关闭 AMP")
    ap.add_argument("--auto_batch", action="store_true", help="自动估算批大小")
    ap.add_argument("--max_vram_frac", type=float, default=0.9)

    args = ap.parse_args()
    if args.lr is not None:
        args.lr0 = float(args.lr)

    # llvip.train() 会读取这些字段；这里显式关闭 VLM output/sample reweight。
    args.vlm_output_dir = None
    args.vlm_weight_min = 1.0
    args.vlm_weight_max = 1.0
    args.vlm_conf_power = 1.0
    return args


def main() -> None:
    configure_m3fd_classes()
    args = get_args()
    llvip.train(args)


if __name__ == "__main__":
    main()
