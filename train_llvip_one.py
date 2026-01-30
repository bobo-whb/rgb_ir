import os
import sys
import math
import time
import random
import logging
import copy
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False

# -------- Logging --------
def setup_logger(log_file: str, *, is_main: bool = True):
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8"); fh.setFormatter(fmt)
    logger.addHandler(fh)
    if is_main:
        sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
        logger.addHandler(sh)
    return logger

def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0

def get_world_size() -> int:
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1

def is_main_process() -> bool:
    return get_rank() == 0

def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model

def parse_gpus_arg(gpus: str) -> List[int]:
    """
    解析 `--gpus 0,1,2` 形式的参数。
    返回空列表表示未启用（走 --gpu 单卡或 CPU）。
    """
    if not gpus:
        return []
    out: List[int] = []
    for x in gpus.split(","):
        x = x.strip()
        if x == "":
            continue
        out.append(int(x))
    return out

# -------- Class alias mapping --------
ALIAS_MAP: Dict[str, Optional[str]] = {
    "person": "person",
    "people": "person",
    "pedestrian": "person",
    "*": None,
    "": None,
    "unknown": None,
}

CANONICAL_CLASSES: List[str] = [
    "person"
]
CLS2ID = {c:i for i,c in enumerate(CANONICAL_CLASSES)}

def clean_class_name(name: str) -> Optional[str]:
    if name is None:
        return None
    n = name.strip().lower()
    if n in ALIAS_MAP:
        return ALIAS_MAP[n]
    return n

# -------- Geometry helpers --------
def polygon_to_rbox(pts: List[float]) -> Tuple[float,float,float,float,float]:
    assert len(pts) == 8
    P = np.array(pts, dtype=np.float32).reshape(4,2)
    cx, cy = P.mean(axis=0)
    Pc = P - np.array([cx, cy], dtype=np.float32)
    C = Pc.T @ Pc
    eigvals, eigvecs = np.linalg.eigh(C)
    v = eigvecs[:, 1]
    angle = math.atan2(v[1], v[0])
    vp = np.array([-v[1], v[0]], dtype=np.float32)
    proj_major = Pc @ v
    proj_minor = Pc @ vp
    w = float(proj_major.max() - proj_major.min())
    h = float(proj_minor.max() - proj_minor.min())
    w = max(w, 1.0); h = max(h, 1.0)
    return float(cx), float(cy), w, h, float(angle)

def rbox_to_aabb(box: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h, ang = box.unbind(-1)
    cosa = torch.cos(ang); sina = torch.sin(ang)
    hw, hh = w/2, h/2
    xs = torch.stack([ hw,  hw, -hw, -hw], dim=-1)
    ys = torch.stack([ hh, -hh, -hh,  hh], dim=-1)
    xr = xs * cosa.unsqueeze(-1) - ys * sina.unsqueeze(-1)
    yr = xs * sina.unsqueeze(-1) + ys * cosa.unsqueeze(-1)
    x = xr + cx.unsqueeze(-1)
    y = yr + cy.unsqueeze(-1)
    x1 = x.min(-1).values; y1 = y.min(-1).values
    x2 = x.max(-1).values; y2 = y.max(-1).values
    return torch.stack([x1,y1,x2,y2], dim=-1)

def rbox_to_corners(boxes: torch.Tensor) -> torch.Tensor:
    """
    将旋转框 (cx, cy, w, h, angle) 转换为四个角点坐标
    Args:
        boxes: (..., 5) tensor, [cx, cy, w, h, angle(弧度)]
    Returns:
        corners: (..., 4, 2) tensor, 四个角点 (x, y)
    """
    cx, cy, w, h, angle = boxes.unbind(-1)
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)

    # 半宽半高
    hw, hh = w * 0.5, h * 0.5

    # 四个角点相对于中心的坐标（逆时针顺序）
    dx = torch.stack([hw, -hw, -hw, hw], dim=-1)
    dy = torch.stack([hh, hh, -hh, -hh], dim=-1)

    # 旋转变换
    cos_a = cos_a.unsqueeze(-1)
    sin_a = sin_a.unsqueeze(-1)
    x = cx.unsqueeze(-1) + dx * cos_a - dy * sin_a
    y = cy.unsqueeze(-1) + dx * sin_a + dy * cos_a

    return torch.stack([x, y], dim=-1)  # (..., 4, 2)


def polygon_area(corners: torch.Tensor) -> torch.Tensor:
    """
    使用 Shoelace 公式计算多边形面积
    Args:
        corners: (..., N, 2) tensor, N个顶点
    Returns:
        area: (...,) tensor
    """
    x = corners[..., 0]
    y = corners[..., 1]
    # Shoelace formula
    area = 0.5 * torch.abs(
        torch.sum(x[..., :-1] * y[..., 1:] - x[..., 1:] * y[..., :-1], dim=-1) +
        x[..., -1] * y[..., 0] - x[..., 0] * y[..., -1]
    )
    return area


def sutherland_hodgman_clip(subject: torch.Tensor, clip: torch.Tensor) -> torch.Tensor:
    """
    Sutherland-Hodgman 多边形裁剪算法（单对样本）
    Args:
        subject: (4, 2) 被裁剪多边形的顶点
        clip: (4, 2) 裁剪多边形的顶点
    Returns:
        result: (M, 2) 裁剪后的多边形顶点，M <= 8
    """
    def inside_edge(point, edge_p1, edge_p2):
        # 判断点是否在边的左侧（使用叉积）
        return (edge_p2[0] - edge_p1[0]) * (point[1] - edge_p1[1]) - \
               (edge_p2[1] - edge_p1[1]) * (point[0] - edge_p1[0]) >= 0

    def intersection(p1, p2, edge_p1, edge_p2):
        # 计算线段与边的交点
        dc = edge_p2 - edge_p1
        dp = p1 - p2
        n1 = (edge_p1[0] - p2[0]) * dc[1] - (edge_p1[1] - p2[1]) * dc[0]
        n2 = dp[0] * dc[1] - dp[1] * dc[0]
        n2 = torch.where(torch.abs(n2) < 1e-8, torch.ones_like(n2) * 1e-8, n2)
        t = n1 / n2
        return p2 + t.unsqueeze(-1) * dp

    output_list = subject

    for i in range(4):
        if output_list.shape[0] == 0:
            break

        edge_p1 = clip[i]
        edge_p2 = clip[(i + 1) % 4]
        input_list = output_list
        output_list = torch.zeros((0, 2), dtype=subject.dtype, device=subject.device)

        if input_list.shape[0] == 0:
            break

        for j in range(input_list.shape[0]):
            current = input_list[j]
            previous = input_list[j - 1]

            current_inside = inside_edge(current, edge_p1, edge_p2)
            previous_inside = inside_edge(previous, edge_p1, edge_p2)

            if current_inside:
                if not previous_inside:
                    inter = intersection(previous, current, edge_p1, edge_p2)
                    output_list = torch.cat([output_list, inter.unsqueeze(0)], dim=0)
                output_list = torch.cat([output_list, current.unsqueeze(0)], dim=0)
            elif previous_inside:
                inter = intersection(previous, current, edge_p1, edge_p2)
                output_list = torch.cat([output_list, inter.unsqueeze(0)], dim=0)

    return output_list


def iou_rotated(boxes1: torch.Tensor, boxes2: torch.Tensor,
                mode: str = "fast") -> torch.Tensor:
    """
    计算旋转框的 IoU (优化版本：两阶段过滤 + 小角度快速路径)
    Args:
        boxes1: (N, 5) tensor, [cx, cy, w, h, angle(弧度)]
        boxes2: (M, 5) tensor, [cx, cy, w, h, angle(弧度)]
        mode: 'fast' - 两阶段过滤(推荐), 'accurate' - 全精确计算, 'aabb' - 纯AABB近似
    Returns:
        iou: (N, M) tensor
    """
    N = boxes1.shape[0]
    M = boxes2.shape[0]

    if N == 0 or M == 0:
        return torch.zeros((N, M), dtype=boxes1.dtype, device=boxes1.device)

    # 模式1: 纯 AABB 近似（最快，精度最低）
    if mode == "aabb":
        aabb1 = rbox_to_aabb(boxes1)
        aabb2 = rbox_to_aabb(boxes2)
        return box_iou_axis_aligned(aabb1, aabb2)

    # 模式2: 全精确计算（最慢，精度最高）
    if mode == "accurate":
        return _iou_rotated_accurate(boxes1, boxes2)

    # 模式3: 快速混合模式（推荐：速度与精度平衡）
    # 策略1：小角度框直接用 AABB（误差 < 1%）
    angle_threshold = 0.087  # ~5度，旋转很小时 AABB 误差可忽略
    max_angle1 = torch.abs(boxes1[:, 4]).max()
    max_angle2 = torch.abs(boxes2[:, 4]).max()

    if max_angle1 < angle_threshold and max_angle2 < angle_threshold:
        # 所有框角度都很小，直接用 AABB
        aabb1 = rbox_to_aabb(boxes1)
        aabb2 = rbox_to_aabb(boxes2)
        return box_iou_axis_aligned(aabb1, aabb2)

    # 策略2：两阶段过滤 - 先用 AABB 粗筛，再精确计算
    # 第一阶段：快速 AABB IoU（完全向量化）
    aabb1 = rbox_to_aabb(boxes1)
    aabb2 = rbox_to_aabb(boxes2)
    aabb_iou = box_iou_axis_aligned(aabb1, aabb2)

    # 找出可能相交的框对（AABB IoU > 0 说明可能有交集）
    candidates = (aabb_iou > 0).nonzero(as_tuple=False)

    if candidates.numel() == 0:
        # 没有候选对，全部 IoU = 0
        return torch.zeros((N, M), dtype=boxes1.dtype, device=boxes1.device)

    # 第二阶段：只对候选框对计算精确旋转 IoU
    ious = torch.zeros((N, M), dtype=boxes1.dtype, device=boxes1.device)

    # 预计算角点（避免重复计算）
    corners1 = rbox_to_corners(boxes1)
    corners2 = rbox_to_corners(boxes2)
    area1 = (boxes1[:, 2] * boxes1[:, 3]).clamp(min=1e-8)
    area2 = (boxes2[:, 2] * boxes2[:, 3]).clamp(min=1e-8)

    # 只计算候选对的精确 IoU
    for idx in range(candidates.shape[0]):
        i, j = candidates[idx, 0].item(), candidates[idx, 1].item()

        # 使用 Sutherland-Hodgman 算法计算交集多边形
        inter_poly = sutherland_hodgman_clip(
            corners1[i].contiguous(),
            corners2[j].contiguous()
        )

        if inter_poly.shape[0] >= 3:
            inter_area = polygon_area(inter_poly.unsqueeze(0)).squeeze(0)
            union_area = area1[i] + area2[j] - inter_area
            ious[i, j] = (inter_area / union_area.clamp(min=1e-8)).clamp(min=0.0, max=1.0)

    return ious


def _iou_rotated_accurate(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """全精确计算版本（内部使用）"""
    N = boxes1.shape[0]
    M = boxes2.shape[0]

    corners1 = rbox_to_corners(boxes1)
    corners2 = rbox_to_corners(boxes2)
    area1 = (boxes1[:, 2] * boxes1[:, 3]).clamp(min=1e-8)
    area2 = (boxes2[:, 2] * boxes2[:, 3]).clamp(min=1e-8)

    ious = torch.zeros((N, M), dtype=boxes1.dtype, device=boxes1.device)

    for i in range(N):
        for j in range(M):
            inter_poly = sutherland_hodgman_clip(
                corners1[i].contiguous(),
                corners2[j].contiguous()
            )

            if inter_poly.shape[0] >= 3:
                inter_area = polygon_area(inter_poly.unsqueeze(0)).squeeze(0)
                union_area = area1[i] + area2[j] - inter_area
                ious[i, j] = (inter_area / union_area.clamp(min=1e-8)).clamp(min=0.0, max=1.0)

    return ious

def box_iou_axis_aligned(b1: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    lt = torch.max(b1[:,None,:2], b2[None,:, :2])
    rb = torch.min(b1[:,None,2:], b2[None,:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:,:,0] * wh[:,:,1]
    area1 = ((b1[:,2]-b1[:,0]) * (b1[:,3]-b1[:,1]))[:,None]
    area2 = ((b2[:,2]-b2[:,0]) * (b2[:,3]-b2[:,1]))[None,:]
    union = area1 + area2 - inter + 1e-6
    return inter / union

# -------- Data --------
def _normalize_header(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def read_csv(path: str) -> List[Dict[str,str]]:
    required = ["visible_path", "infrared_path", "annotation_path"]
    rows: List[Dict[str, str]] = []

    if HAS_PANDAS:
        df = pd.read_csv(path)
        rename = {}
        for c in df.columns:
            norm = _normalize_header(c)
            if norm in required:
                rename[c] = norm
        df = df.rename(columns=rename)
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"CSV 缺少列：需要{required}，实际列：{list(df.columns)}")
        for _, row in df.iterrows():
            rows.append({col: str(row[col]) for col in required})
    else:
        import csv
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
            map_cols = {_normalize_header(h): h for h in header}
            missing = [col for col in required if col not in map_cols]
            if missing:
                raise ValueError(f"CSV 缺少列：需要{required}，实际列：{header}")
            for row in reader:
                rows.append({col: str(row[map_cols[col]]) for col in required})
    return rows

def parse_shared_xml(xml_path: str) -> Tuple[List[List[float]], List[int]]:
    """
    Parse LLVIP shared annotations (Pascal VOC style axis-aligned boxes).
    Returns boxes in absolute xyxy format.
    """
    boxes: List[List[float]] = []
    labels: List[int] = []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for obj in root.findall("object"):
        name = clean_class_name(obj.findtext("name", default=""))
        if name is None or name not in CLS2ID:
            continue
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        try:
            xmin = float(bnd.findtext("xmin", "0"))
            ymin = float(bnd.findtext("ymin", "0"))
            xmax = float(bnd.findtext("xmax", "0"))
            ymax = float(bnd.findtext("ymax", "0"))
        except ValueError:
            continue
        if xmax <= xmin or ymax <= ymin:
            continue
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(CLS2ID[name])
    return boxes, labels

class RGBIRDetDataset(Dataset):
    def __init__(self, csv_path: str, imgsz: int = 640, augment: bool = True,
                 # ---- Augment defaults aligned to YOLOv8 ----
                 mosaic: bool = True, mosaic_prob: float = 1.0, mixup_prob: float = 0.0,
                 mosaic_scale: Tuple[float, float] = (0.5, 1.5),
                 hsv_gain: Tuple[float, float, float] = (0.015, 0.7, 0.4),
                 degrees: float = 0.0, translate: float = 0.1, scale: float = 0.5,
                 shear: float = 0.0, perspective: float = 0.0,
                 # project-specific spectral mixing (disabled by default to match YOLOv8)
                 spectral_prob: float = 0.0, spectral_strength: float = 0.0,
                 hflip_prob: float = 0.5, vflip_prob: float = 0.0):
        super().__init__()
        self.items = read_csv(csv_path)
        self.imgsz = imgsz
        self.augment = augment
        self.mosaic = mosaic
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        lo, hi = mosaic_scale
        if hi < lo:
            lo, hi = hi, lo
        self.mosaic_scale = (max(lo, 0.1), max(hi, 0.1))
        self.hsv_gain = hsv_gain
        # random perspective (Ultralytics/YOLOv8-style)
        self.degrees = float(degrees)
        self.translate = float(translate)
        self.scale = float(scale)
        self.shear = float(shear)
        self.perspective = float(perspective)
        self.spectral_prob = spectral_prob
        self.spectral_strength = max(spectral_strength, 0.0)
        self.hflip_prob = max(min(hflip_prob, 1.0), 0.0)
        self.vflip_prob = max(min(vflip_prob, 1.0), 0.0)

    def __len__(self):
        return len(self.items)

    def _load_image(self, path: str) -> Image.Image:
        img = Image.open(path).convert("RGB")
        return img

    def _load_sample_numpy(self, idx: int):
        it = self.items[idx]
        rgb = np.array(self._load_image(it["visible_path"]), dtype=np.uint8)
        ir = np.array(self._load_image(it["infrared_path"]), dtype=np.uint8)
        boxes, labels = parse_shared_xml(it["annotation_path"])
        boxes_np = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        labels_np = np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)
        return rgb, ir, boxes_np, labels_np

    def __getitem__(self, idx: int):
        it = self.items[idx]
        rgb_path = it["visible_path"]
        ir_path  = it["infrared_path"]
        ann_path = it["annotation_path"]

        if self.augment:
            rgb_np, ir_np, boxes_np, labels_np = self._build_train_sample(idx)
            rgb = Image.fromarray(rgb_np)
            ir = Image.fromarray(ir_np)
        else:
            rgb = self._load_image(rgb_path)
            ir  = self._load_image(ir_path)
            boxes, labels = parse_shared_xml(ann_path)
            boxes_np = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0,4), np.float32)
            labels_np = np.array(labels, dtype=np.int64) if labels else np.zeros((0,), np.int64)

        rgb, scale, pad = letterbox(rgb, self.imgsz)
        ir,  _,    _    = letterbox(ir, self.imgsz)
        if boxes_np.shape[0] > 0:
            boxes_scaled = boxes_np.copy()
            boxes_scaled[:, [0, 2]] = boxes_np[:, [0, 2]] * scale + pad[0]
            boxes_scaled[:, [1, 3]] = boxes_np[:, [1, 3]] * scale + pad[1]
        else:
            boxes_scaled = boxes_np

        # ---- Apply YOLOv8-like augmentations after letterbox (same transform for RGB/IR) ----
        if self.augment:
            rgb_np2 = np.array(rgb, dtype=np.uint8)
            ir_np2  = np.array(ir,  dtype=np.uint8)
            boxes_scaled = boxes_scaled.astype(np.float32) if boxes_scaled is not None else np.zeros((0, 4), np.float32)

            rgb_np2, ir_np2, boxes_scaled, labels_np = self._random_perspective_np(
                rgb_np2, ir_np2, boxes_scaled, labels_np,
                degrees=self.degrees, translate=self.translate, scale=self.scale,
                shear=self.shear, perspective=self.perspective,
            )
            rgb_np2, ir_np2, boxes_scaled = self._random_flip_np(rgb_np2, ir_np2, boxes_scaled)
            rgb_np2, ir_np2 = self._apply_hsv(rgb_np2, ir_np2)
            rgb_np2, ir_np2 = self._spectral_augment(rgb_np2, ir_np2)

            rgb_t = torch.from_numpy(rgb_np2).permute(2, 0, 1).float() / 255.0
            ir_t  = torch.from_numpy(ir_np2 ).permute(2, 0, 1).float() / 255.0
            boxes_final = boxes_scaled
            labels_final = labels_np
        else:
            rgb_t = torch.from_numpy(np.array(rgb)).permute(2,0,1).float()/255.0
            ir_t  = torch.from_numpy(np.array(ir )).permute(2,0,1).float()/255.0
            boxes_final = boxes_scaled
            labels_final = labels_np

        target = {
            "boxes": torch.from_numpy(boxes_final),
            "labels": torch.from_numpy(labels_final),
            "orig_size": torch.tensor([rgb.size[1], rgb.size[0]], dtype=torch.float32)
        }
        return rgb_t, ir_t, target, os.path.basename(rgb_path)

    def _build_train_sample(self, idx: int):
        if self.mosaic and random.random() < self.mosaic_prob:
            rgb, ir, boxes, labels = self._load_mosaic_sample(idx)
            if self.mixup_prob > 0 and random.random() < self.mixup_prob:
                mix_idx = random.randrange(len(self.items))
                rgb2, ir2, boxes2, labels2 = self._load_mosaic_sample(mix_idx)
                rgb, ir, boxes, labels = self._mixup_samples(rgb, ir, boxes, labels, rgb2, ir2, boxes2, labels2)
        else:
            rgb, ir, boxes, labels = self._load_sample_numpy(idx)
        return rgb, ir, boxes, labels

    def _mixup_samples(self, rgb1, ir1, boxes1, labels1, rgb2, ir2, boxes2, labels2):
        lam = np.random.beta(8.0, 8.0)
        lam = float(np.clip(lam, 0.1, 0.9))
        rgb = (rgb1.astype(np.float32) * lam + rgb2.astype(np.float32) * (1.0 - lam)).astype(np.uint8)
        ir = (ir1.astype(np.float32) * lam + ir2.astype(np.float32) * (1.0 - lam)).astype(np.uint8)
        if boxes1.shape[0] and boxes2.shape[0]:
            boxes = np.concatenate([boxes1, boxes2], axis=0)
            labels = np.concatenate([labels1, labels2], axis=0)
        elif boxes2.shape[0]:
            boxes, labels = boxes2, labels2
        else:
            boxes, labels = boxes1, labels1
        return rgb, ir, boxes, labels

    def _random_flip_np(self, rgb, ir, boxes):
        h, w = rgb.shape[:2]
        if self.hflip_prob > 0 and random.random() < self.hflip_prob:
            rgb = np.fliplr(rgb).copy()
            ir = np.fliplr(ir).copy()
            if boxes.shape[0]:
                x1 = boxes[:, 0].copy()
                x2 = boxes[:, 2].copy()
                boxes[:, 0] = w - x2
                boxes[:, 2] = w - x1
        if self.vflip_prob > 0 and random.random() < self.vflip_prob:
            rgb = np.flipud(rgb).copy()
            ir = np.flipud(ir).copy()
            if boxes.shape[0]:
                y1 = boxes[:, 1].copy()
                y2 = boxes[:, 3].copy()
                boxes[:, 1] = h - y2
                boxes[:, 3] = h - y1
        return rgb, ir, boxes

    def _apply_hsv(self, rgb, ir):
        """Apply HSV jittering only on the RGB modality."""
        if not self.hsv_gain:
            return rgb, ir
        hgain, sgain, vgain = self.hsv_gain
        if hgain <= 0 and sgain <= 0 and vgain <= 0:
            return rgb, ir
        gains = np.random.uniform(-1, 1, 3) * np.array([hgain, sgain, vgain]) + 1.0

        def _adjust(img):
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 0] = (hsv[..., 0] * gains[0]) % 180.0
            hsv[..., 1] = np.clip(hsv[..., 1] * gains[1], 0, 255)
            hsv[..., 2] = np.clip(hsv[..., 2] * gains[2], 0, 255)
            return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # Only perturb RGB while keeping IR untouched.
        return _adjust(rgb), ir

    def _spectral_augment(self, rgb, ir):
        """Perform spectral mixing on RGB only, leaving IR as-is."""
        if self.spectral_prob <= 0 or self.spectral_strength <= 0:
            return rgb, ir
        if random.random() >= self.spectral_prob:
            return rgb, ir
        alpha = random.uniform(0.0, self.spectral_strength)
        rgb_f = rgb.astype(np.float32)
        ir_f = ir.astype(np.float32)
        rgb_mix = np.clip(rgb_f * (1.0 - alpha) + ir_f * alpha, 0, 255).astype(np.uint8)
        return rgb_mix, ir

    def _random_perspective_np(self, rgb, ir, boxes, labels=None,
                               degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
                               border=(0, 0)):
        """
        Ultralytics/YOLOv8-style random perspective/affine.
        Applies identical transform to RGB/IR and updates axis-aligned xyxy boxes.
        """
        if boxes is None:
            boxes = np.zeros((0, 4), dtype=np.float32)
        if labels is None:
            labels = np.zeros((boxes.shape[0],), dtype=np.int64)

        h0, w0 = rgb.shape[:2]
        assert ir.shape[:2] == (h0, w0), "RGB/IR size mismatch"

        # Output size (with optional border, kept at imgsz by default)
        height = h0 + border[1] * 2
        width = w0 + border[0] * 2

        # Identity if no aug
        if degrees == 0 and translate == 0 and scale == 0 and shear == 0 and perspective == 0:
            return rgb, ir, boxes, labels

        # Center
        C = np.eye(3, dtype=np.float32)
        C[0, 2] = -w0 / 2
        C[1, 2] = -h0 / 2

        # Perspective
        P = np.eye(3, dtype=np.float32)
        P[2, 0] = random.uniform(-perspective, perspective)
        P[2, 1] = random.uniform(-perspective, perspective)

        # Rotation and Scale
        R = np.eye(3, dtype=np.float32)
        a = random.uniform(-degrees, degrees)
        s = random.uniform(1 - scale, 1 + scale)
        if abs(a) > 1e-3 or abs(s - 1.0) > 1e-3:
            M2 = cv2.getRotationMatrix2D(center=(0, 0), angle=a, scale=s)  # 2x3
            R[:2, :] = M2
        # Shear
        S = np.eye(3, dtype=np.float32)
        if shear != 0:
            sx = math.tan(random.uniform(-shear, shear) * math.pi / 180)
            sy = math.tan(random.uniform(-shear, shear) * math.pi / 180)
            S[0, 1] = sx
            S[1, 0] = sy

        # Translation
        T = np.eye(3, dtype=np.float32)
        T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
        T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height

        # Combined matrix
        M = T @ S @ R @ P @ C  # 3x3

        border_val = (114, 114, 114)
        if perspective:
            rgb_t = cv2.warpPerspective(rgb, M, dsize=(width, height), borderValue=border_val)
            ir_t  = cv2.warpPerspective(ir,  M, dsize=(width, height), borderValue=border_val)
        else:
            rgb_t = cv2.warpAffine(rgb, M[:2], dsize=(width, height), borderValue=border_val)
            ir_t  = cv2.warpAffine(ir,  M[:2], dsize=(width, height), borderValue=border_val)

        if boxes.shape[0] == 0:
            return rgb_t, ir_t, boxes, labels

        # Transform boxes (x1y1, x2y1, x2y2, x1y2)
        n = boxes.shape[0]
        corners = boxes[:, [0, 1, 2, 1, 2, 3, 0, 3]].reshape(n * 4, 2)
        ones = np.ones((n * 4, 1), dtype=np.float32)
        xy1 = np.concatenate([corners, ones], axis=1)  # (n*4,3)
        xy = xy1 @ M.T
        if perspective:
            xy = xy[:, :2] / (xy[:, 2:3] + 1e-9)
        else:
            xy = xy[:, :2]
        xy = xy.reshape(n, 4, 2)

        new_boxes = np.concatenate([xy.min(axis=1), xy.max(axis=1)], axis=1)  # (n,4)
        new_boxes[:, [0, 2]] = new_boxes[:, [0, 2]].clip(0, width)
        new_boxes[:, [1, 3]] = new_boxes[:, [1, 3]].clip(0, height)

        # Filter candidates (Ultralytics-style)
        w = new_boxes[:, 2] - new_boxes[:, 0]
        h = new_boxes[:, 3] - new_boxes[:, 1]
        w0b = boxes[:, 2] - boxes[:, 0]
        h0b = boxes[:, 3] - boxes[:, 1]
        area = w * h
        area0 = w0b * h0b + 1e-9
        ar = np.maximum(w / (h + 1e-9), h / (w + 1e-9))
        keep = (w > 2) & (h > 2) & (area / area0 > 0.1) & (ar < 20)
        new_boxes = new_boxes[keep]
        new_labels = labels[keep] if labels is not None else None

        return rgb_t, ir_t, new_boxes.astype(np.float32), new_labels

    def _load_mosaic_sample(self, idx: int):
        indices = [idx] + [random.randrange(len(self.items)) for _ in range(3)]
        s = self.imgsz
        mosaic_rgb = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        mosaic_ir = np.full_like(mosaic_rgb, 114)
        mosaic_boxes = []
        mosaic_labels = []
        yc = int(random.uniform(0.5 * s, 1.5 * s))
        xc = int(random.uniform(0.5 * s, 1.5 * s))

        for i, index in enumerate(indices):
            rgb, ir, boxes, labels = self._load_sample_numpy(index)
            h0, w0 = rgb.shape[:2]
            scale = random.uniform(self.mosaic_scale[0], self.mosaic_scale[1])
            scale = max(scale, 0.1)
            if abs(scale - 1.0) > 1e-3:
                new_w = max(1, int(w0 * scale))
                new_h = max(1, int(h0 * scale))
                rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                ir = cv2.resize(ir, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                if boxes.shape[0]:
                    gain_w = new_w / w0
                    gain_h = new_h / h0
                    boxes = boxes.copy()
                    boxes[:, [0, 2]] *= gain_w
                    boxes[:, [1, 3]] *= gain_h
            h, w = rgb.shape[:2]
            if i == 0:
                x_offset = xc - w
                y_offset = yc - h
            elif i == 1:
                x_offset = xc
                y_offset = yc - h
            elif i == 2:
                x_offset = xc - w
                y_offset = yc
            else:
                x_offset = xc
                y_offset = yc

            x1a = max(x_offset, 0)
            y1a = max(y_offset, 0)
            x2a = min(x_offset + w, s * 2)
            y2a = min(y_offset + h, s * 2)
            x1b = max(-x_offset, 0)
            y1b = max(-y_offset, 0)
            x2b = x1b + (x2a - x1a)
            y2b = y1b + (y2a - y1a)
            if x1a >= x2a or y1a >= y2a:
                continue

            mosaic_rgb[y1a:y2a, x1a:x2a] = rgb[y1b:y2b, x1b:x2b]
            mosaic_ir[y1a:y2a, x1a:x2a] = ir[y1b:y2b, x1b:x2b]

            if boxes.shape[0]:
                boxes_c = boxes.copy()
                boxes_c[:, [0, 2]] = boxes_c[:, [0, 2]].clip(0, w)
                boxes_c[:, [1, 3]] = boxes_c[:, [1, 3]].clip(0, h)
                boxes_c[:, [0, 2]] = boxes_c[:, [0, 2]] - x1b + x1a
                boxes_c[:, [1, 3]] = boxes_c[:, [1, 3]] - y1b + y1a
                mosaic_boxes.append(boxes_c)
                mosaic_labels.append(labels.copy())

        if mosaic_boxes:
            boxes = np.concatenate(mosaic_boxes, axis=0)
            labels = np.concatenate(mosaic_labels, axis=0)
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, s * 2)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, s * 2)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)

        rgb, ir, boxes, labels = self._crop_to_size(mosaic_rgb, mosaic_ir, boxes, labels, self.imgsz)
        return rgb, ir, boxes, labels

    def _crop_to_size(self, rgb, ir, boxes, labels, target):
        h, w = rgb.shape[:2]
        if h == target and w == target:
            return rgb, ir, boxes, labels
        max_x = max(w - target, 0)
        max_y = max(h - target, 0)
        x0 = random.randint(0, max_x)
        y0 = random.randint(0, max_y)
        x1 = x0 + target
        y1 = y0 + target
        rgb_crop = rgb[y0:y1, x0:x1]
        ir_crop = ir[y0:y1, x0:x1]
        if boxes.shape[0]:
            boxes = boxes.copy()
            boxes[:, [0, 2]] -= x0
            boxes[:, [1, 3]] -= y0
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, target)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, target)
            keep = (boxes[:, 2] - boxes[:, 0] > 2) & (boxes[:, 3] - boxes[:, 1] > 2)
            boxes = boxes[keep]
            labels = labels[keep]
        return rgb_crop, ir_crop, boxes, labels

def letterbox(img: Image.Image, new_size: int) -> Tuple[Image.Image, float, Tuple[float,float]]:
    w, h = img.size
    s = min(new_size/w, new_size/h)
    nw, nh = int(w*s), int(h*s)
    img2 = img.resize((nw,nh), Image.BILINEAR)
    canvas = Image.new("RGB", (new_size,new_size), (114,114,114))
    pad = ((new_size-nw)//2, (new_size-nh)//2)
    canvas.paste(img2, pad)
    return canvas, s, pad

def collate_fn(batch):
    rgbs, irs, targets, names = zip(*batch)
    rgbs = torch.stack(rgbs,0)
    irs  = torch.stack(irs ,0)
    return rgbs, irs, list(targets), names

class ModelEMA:
    """
    Exponential Moving Average of model weights (YOLOv8-style).
    """
    def __init__(self, model: nn.Module, decay: float = 0.9999, tau: float = 2000.0, updates: int = 0):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = float(decay)
        self.tau = float(tau)
        self.updates = int(updates)

    def update(self, model: nn.Module):
        self.updates += 1
        d = self.decay * (1.0 - math.exp(-self.updates / max(self.tau, 1.0)))
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v.copy_(v * d + msd[k].detach() * (1.0 - d))
                else:
                    v.copy_(msd[k])

def build_optimizer(model: nn.Module, args):
    """
    YOLOv8-like optimizer parameter groups:
      - g0: weights with decay
      - g1: norm weights without decay
      - g2: biases without decay
    """
    g0, g1, g2 = [], [], []
    for module_name, m in model.named_modules():
        for param_name, p in m.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            if param_name == "bias":
                g2.append(p)
            elif isinstance(m, nn.BatchNorm2d) or "bn" in module_name.lower():
                g1.append(p)
            else:
                g0.append(p)

    groups = [
        {"params": g0, "weight_decay": args.weight_decay},
        {"params": g1, "weight_decay": 0.0},
        {"params": g2, "weight_decay": 0.0},
    ]

    opt = args.optimizer.lower()
    if opt == "sgd":
        optimizer = torch.optim.SGD(groups, lr=args.lr0, momentum=args.momentum, nesterov=True)
    elif opt == "adamw":
        optimizer = torch.optim.AdamW(groups, lr=args.lr0, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    return optimizer

def cosine_lr_factor(progress: float, lrf: float) -> float:
    # progress in [0,1]
    return lrf + 0.5 * (1.0 - lrf) * (1.0 + math.cos(math.pi * progress))

# -------- Model (YOLOv8-like backbone/neck: C2f) --------
class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None: p = k//2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True):
        super().__init__()
        c_ = int(c2 // 2)
        self.cv1 = ConvBNAct(c1, c_, 1, 1)
        self.cv2 = ConvBNAct(c_, c2, 3, 1)
        self.add = shortcut and c1 == c2
    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y

class C2f(nn.Module):
    """
    YOLOv8 C2f block: cv1 -> split -> n * Bottleneck(c_, c_) with gradient-friendly concat -> cv2
    """
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = ConvBNAct(c1, 2*c_, 1, 1)
        self.cv2 = ConvBNAct((2+n)*c_, c2, 1, 1)
        self.m = nn.ModuleList([Bottleneck(c_, c_, shortcut=shortcut) for _ in range(n)])
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))  # y[0], y[1] : each c_
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))

class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = ConvBNAct(c1, c_, 1, 1)
        self.cv2 = ConvBNAct(c_*4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k//2)
    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))

class MLP(nn.Module):
    """
    轻量 MLP：1x1 Conv -> SiLU -> 1x1 Conv
    可改变通道数；保持空间尺寸不变
    """
    def __init__(self, in_ch: int, out_ch: int, r: float = 0.5):
        super().__init__()
        hidden = max(1, int(in_ch * r))
        self.fc1 = ConvBNAct(in_ch, hidden, 1, 1)
        self.fc2 = ConvBNAct(hidden, out_ch, 1, 1, act=False)
    def forward(self, x):
        return self.fc2(self.fc1(x))

class BackboneV8Large(nn.Module):
    """
    YOLOv8-L 风格主干：输出 P3(1/8,256), P4(1/16,512), P5(1/32,1024)
    深度更大（n=3/6/6），通道更宽（64/128/256/512/1024）
    """
    def __init__(self, in_ch: int = 3):
        super().__init__()
        self.layer1 = nn.Sequential(
            ConvBNAct(in_ch, 64, 3, 2),     # 640 -> 320
            ConvBNAct(64, 128, 3, 2),       # 320 -> 160
            C2f(128, 128, n=3, shortcut=True)
        )
        self.layer2 = nn.Sequential(
            ConvBNAct(128, 256, 3, 2),      # 160 -> 80
            C2f(256, 256, n=6, shortcut=True)
        )
        self.layer3 = nn.Sequential(
            ConvBNAct(256, 512, 3, 2),      # 80 -> 40
            C2f(512, 512, n=6, shortcut=True)
        )
        self.layer4 = nn.Sequential(
            ConvBNAct(512, 1024, 3, 2),     # 40 -> 20
            C2f(1024, 1024, n=3, shortcut=True),
            SPPF(1024, 1024, k=5)
        )

    def forward(self, x):
        x = self.layer1(x)
        P3 = self.layer2(x)   # 1/8,  256 ch
        P4 = self.layer3(P3)  # 1/16, 512 ch
        P5 = self.layer4(P4)  # 1/32, 1024 ch
        return P3, P4, P5

class YoloV8NeckSingle(nn.Module):
    """
    YOLOv8-L 风格单模态颈部（FPN + PAN）：
      - 输入: P3(256), P4(512), P5(1024)
      - 输出: N3(256), N4(512), N5(1024)
    """
    def __init__(self, C3=256, C4=512, C5=1024):
        super().__init__()
        # 侧连降通道
        self.lat5 = ConvBNAct(C5, 512, 1, 1)
        self.lat4 = ConvBNAct(C4, 512, 1, 1)
        self.lat3 = ConvBNAct(C3, 256, 1, 1)

        # Top-down FPN
        self.fpn_p4 = C2f(512 + 512, 512, n=3, shortcut=True)
        self.fpn_p3 = C2f(256 + 512, 256, n=3, shortcut=True)

        # Bottom-up PAN
        self.pan_p4_down = ConvBNAct(256, 256, 3, 2)
        self.pan_p4 = C2f(256 + 512, 512, n=3, shortcut=True)
        self.pan_p5_down = ConvBNAct(512, 512, 3, 2)
        self.pan_p5 = C2f(512 + 512, 1024, n=3, shortcut=True)

    def forward(self, feats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        p3, p4, p5 = feats  # 来自主干

        # Top-down
        p5_td = self.lat5(p5)  # 512
        p4_td = self.fpn_p4(torch.cat([self.lat4(p4), F.interpolate(p5_td, scale_factor=2.0, mode="nearest")], dim=1))
        p3_td = self.fpn_p3(torch.cat([self.lat3(p3), F.interpolate(p4_td, scale_factor=2.0, mode="nearest")], dim=1))

        # Bottom-up
        p4_bu = self.pan_p4(torch.cat([self.pan_p4_down(p3_td), p4_td], dim=1))      # 512
        p5_bu = self.pan_p5(torch.cat([self.pan_p5_down(p4_bu), p5_td], dim=1))      # 1024

        return p3_td, p4_bu, p5_bu
    
class XAttnGate(nn.Module):
    """
    稳定版跨模态门控（按用户定义的结构）：
      1) 由 r,i,c 生成 [r, i, c, i-r] 在通道维拼接 -> (B, 4C, H, W)
      2) 3x3 Conv + BN + SiLU
      3) 1x1 Conv 将 4C 映射到 3C（零初始化）
      4) reshape 为 (B, 3, C, H, W) 后做 softmax（带温度）
    """
    def __init__(self, channels: int, temperature: float = 2.0):
        super().__init__()
        self.channels = channels
        self.temperature = float(temperature)
        self.conv3 = nn.Conv2d(4 * channels, 4 * channels, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(4 * channels)
        self.act = nn.SiLU(inplace=True)
        self.conv1 = nn.Conv2d(4 * channels, 3 * channels, kernel_size=1, padding=0, bias=True)
        nn.init.zeros_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)

    def forward(self, r: torch.Tensor, i: torch.Tensor, c: torch.Tensor):
        """
        Inputs: r, i, c: (B, C, H, W)
        Returns:
          w1, w2, w3: (B, C, H, W)
        """
        B, C, H, W = r.shape
        feat = torch.cat([r, i, c, i - r], dim=1)  # (B, 4C, H, W)
        x = self.act(self.bn3(self.conv3(feat)))
        x = self.conv1(x)  # (B, 3C, H, W)
        x = x.view(B, 3, C, H, W)
        x = x / max(self.temperature, 1e-6)
        x = torch.softmax(x, dim=1)
        w1 = x[:, 0]
        w2 = x[:, 1]
        w3 = x[:, 2]
        return w1, w2, w3


class DualFusionNeckXAttnL(nn.Module):
    """
    L 规模三路融合颈部（带 FPN + PAN，三路门控融合）：
      - 维持原始三尺度通道 (P3/P4/P5 = C3c/C4c/C5c)
      - 仅在跨尺度交互处做“逐层对齐”（例如 P5->P4, P4->P3 的投影）
      - 在融合前，RGB/IR 各自做 top-down + bottom-up（保持模态独立）
      - 每个尺度构造 (RGB, IR, Concat) 三路候选特征，经注意力门控融合
      - 输出三层通道分别为 (C3c, C4c, C5c)，供检测头直接使用
    """
    def __init__(self, C3c=256, C4c=512, C5c=1024):
        super().__init__()
        # 逐层对齐：仅用于跨尺度相加/交互的通道投影
        self.up5_to4 = ConvBNAct(C5c, C4c, 1, 1)  # P5(C5) -> P4(C4)
        self.up4_to3 = ConvBNAct(C4c, C3c, 1, 1)  # P4(C4) -> P3(C3)

        # 顶到底 FPN 平滑
        self.smooth5 = C2f(C5c, C5c, n=3, shortcut=True)
        self.smooth4 = C2f(C4c, C4c, n=3, shortcut=True)
        self.smooth3 = C2f(C3c, C3c, n=3, shortcut=True)

        # ---------- F5（20x20） ----------
        self.rgb_mlp5 = MLP(C5c, C5c, r=0.5)        # 1024 -> 1024
        self.ir_mlp5  = MLP(C5c, C5c, r=0.5)
        self.cat_mlp5 = MLP(2 * C5c, 2 * C5c, r=0.5)  # 1024+1024 -> 2048
        self.fuse5    = ConvBNAct(2 * C5c, C5c, 1, 1)  # 2048 -> 1024

        # ---------- F4（40x40） ----------
        self.rgb_mlp4 = MLP(C4c, C4c, r=0.5)        # 512 -> 512
        self.ir_mlp4  = MLP(C4c, C4c, r=0.5)
        self.cat_mlp4 = MLP(2 * C4c, 2 * C4c, r=0.5)
        self.fuse4    = ConvBNAct(2 * C4c, C4c, 1, 1)  # 1024 -> 512

        # ---------- F3（80x80） ----------
        self.rgb_mlp3 = MLP(C3c, C3c, r=0.5)        # 256 -> 256
        self.ir_mlp3  = MLP(C3c, C3c, r=0.5)
        self.cat_mlp3 = MLP(2 * C3c, 2 * C3c, r=0.5)  # 256+256 -> 512
        self.fuse3    = ConvBNAct(2 * C3c, C3c, 1, 1)  # 512 -> 256

        # 跨模态注意力门控
        self.gate5 = XAttnGate(C5c)
        self.gate4 = XAttnGate(C4c)
        self.gate3 = XAttnGate(C3c)

        # 输出整形（保持原通道数不变，便于后续检测头直接使用）
        self.out5 = ConvBNAct(C5c, C5c, 1, 1)
        self.out4 = ConvBNAct(C4c, C4c, 1, 1)
        self.out3 = ConvBNAct(C3c, C3c, 1, 1)

        # 自底向上 PAN：在融合前，保持模态独立做尺度交互（逐层对齐）
        self.pan_down4 = ConvBNAct(C3c, C4c, 3, 2)               # P3(C3) -> stride2 -> C4
        self.pan_c4 = C2f(C4c + C4c, C4c, n=3, shortcut=True)    # concat P3ds(C4) + P4(C4) -> C4
        self.pan_down5 = ConvBNAct(C4c, C5c, 3, 2)               # P4(C4) -> stride2 -> C5
        self.pan_c5 = C2f(C5c + C5c, C5c, n=3, shortcut=True)    # concat P4ds(C5) + P5(C5) -> C5

    def forward(self, rgb_feats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                      ir_feats:  Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        r3, r4, r5 = rgb_feats
        i3, i4, i5 = ir_feats

        # RGB top-down
        p5 = self.smooth5(r5)
        r5_to4 = self.up5_to4(p5)
        p4 = self.smooth4(r4 + F.interpolate(r5_to4, scale_factor=2.0, mode="nearest"))
        p4_to3 = self.up4_to3(p4)
        p3 = self.smooth3(r3 + F.interpolate(p4_to3, scale_factor=2.0, mode="nearest"))
        # IR top-down
        p5_i = self.smooth5(i5)
        i5_to4 = self.up5_to4(p5_i)
        p4_i = self.smooth4(i4 + F.interpolate(i5_to4, scale_factor=2.0, mode="nearest"))
        p4_i_to3 = self.up4_to3(p4_i)
        p3_i = self.smooth3(i3 + F.interpolate(p4_i_to3, scale_factor=2.0, mode="nearest"))

        # 自底向上 PAN（在融合前，保持模态独立）
        p4_bu = self.pan_c4(torch.cat([self.pan_down4(p3), p4], dim=1))
        p5_bu = self.pan_c5(torch.cat([self.pan_down5(p4_bu), p5], dim=1))

        p4_bu_i = self.pan_c4(torch.cat([self.pan_down4(p3_i), p4_i], dim=1))
        p5_bu_i = self.pan_c5(torch.cat([self.pan_down5(p4_bu_i), p5_i], dim=1))

        # F5
        r5_m = self.rgb_mlp5(p5_bu)
        i5_m = self.ir_mlp5(p5_bu_i)
        c5_m = self.fuse5(self.cat_mlp5(torch.cat([p5_bu, p5_bu_i], dim=1)))
        w5 = self.gate5(r5_m, i5_m, c5_m)
        F5 = r5_m * w5[0] + i5_m * w5[1] + c5_m * w5[2]

        # F4
        r4_m = self.rgb_mlp4(p4_bu)
        i4_m = self.ir_mlp4(p4_bu_i)
        c4_m = self.fuse4(self.cat_mlp4(torch.cat([p4_bu, p4_bu_i], dim=1)))
        w4 = self.gate4(r4_m, i4_m, c4_m)
        F4 = r4_m * w4[0] + i4_m * w4[1] + c4_m * w4[2]

        # F3
        r3_m = self.rgb_mlp3(p3)
        i3_m = self.ir_mlp3(p3_i)
        c3_m = self.fuse3(self.cat_mlp3(torch.cat([p3, p3_i], dim=1)))
        w3 = self.gate3(r3_m, i3_m, c3_m)
        F3 = r3_m * w3[0] + i3_m * w3[1] + c3_m * w3[2]

        P3 = self.out3(F3)   # C3c
        P4 = self.out4(F4)   # C4c
        P5 = self.out5(F5)   # C5c

        return P3, P4, P5

class DetectHead(nn.Module):
    """
    Anchor-free 解耦头 + DFL（对齐 YOLOv8 思路）
      - reg 分支输出 4*reg_max 的离散距离分布（bins=reg_max，Ultralytics/YOLOv8 风格）
      - cls 分支输出 nc 类概率（无 obj 分支）
    """
    def __init__(self, nc: int, ch: List[int], reg_max: int = 16,
                 strides=(8, 16, 32), img_size: int = 640):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        self.nl = len(ch)
        self.img_size = img_size
        self.strides = list(strides)

        # DFL integral (distribution -> distance expectation)
        self.register_buffer("dfl_proj", torch.arange(reg_max, dtype=torch.float32).view(1, 1, reg_max, 1, 1),
                             persistent=False)

        self.stems = nn.ModuleList([ConvBNAct(c, c, k=1, s=1) for c in ch])
        self.cls_conv = nn.ModuleList([nn.Sequential(
            ConvBNAct(c, c, 3, 1),
            ConvBNAct(c, c, 3, 1)
        ) for c in ch])
        self.reg_conv = nn.ModuleList([nn.Sequential(
            ConvBNAct(c, c, 3, 1),
            ConvBNAct(c, c, 3, 1)
        ) for c in ch])

        self.cls_pred = nn.ModuleList([nn.Conv2d(c, nc, 1) for c in ch])
        self.reg_pred = nn.ModuleList([nn.Conv2d(c, 4 * reg_max, 1) for c in ch])

        self._bias_init()

    def forward(self, feats: List[torch.Tensor]):
        outputs = []
        for i, x in enumerate(feats):
            x = self.stems[i](x)
            cls_feat = self.cls_conv[i](x)
            reg_feat = self.reg_conv[i](x)
            cls_out = self.cls_pred[i](cls_feat)          # (B, nc, H, W)
            reg_out = self.reg_pred[i](reg_feat)          # (B, 4*reg_max, H, W)
            # Ultralytics-like: merge per-level outputs to a single tensor
            # (B, nc + 4*reg_max, H, W)
            outputs.append(torch.cat([reg_out, cls_out], dim=1))
        return outputs

    def _bias_init(self, prior_prob: float = 0.01):
        """
        Ultralytics-style bias init:
          - cls bias set for low foreground prior and scale-aware (depends on stride & img_size)
          - reg bias gently positive to help optimization stability
        """
        import math

        for i in range(self.nl):
            # classification bias: encourage low initial confidence
            # scale-aware term similar to YOLOv5/8 practice
            stride = float(self.strides[i]) if i < len(self.strides) else 8.0 * (2 ** i)
            # approximate number of locations per image at this level
            nloc = (self.img_size / stride) ** 2
            # 5 positives per image heuristic (common practice)
            cls_bias = math.log(5.0 / max(self.nc, 1) / max(nloc, 1.0))
            nn.init.constant_(self.cls_pred[i].bias, cls_bias)

            # regression bias: small positive value
            nn.init.constant_(self.reg_pred[i].bias, 1.0)

    def dfl_integral(self, reg_out: torch.Tensor) -> torch.Tensor:
        """
        reg_out: (B, 4*reg_max, H, W) logits
        returns: (B, 4, H, W) distances (in bins, not multiplied by stride)
        """
        B, _, H, W = reg_out.shape
        x = reg_out.view(B, 4, self.reg_max, H, W)
        x = x.softmax(2)
        # expectation over bins
        dist = (x * self.dfl_proj.to(x.dtype)).sum(2)
        return dist

    def decode(self, preds: List[torch.Tensor], conf_thr: float = 0.25) -> torch.Tensor:
        """
        Decode merged head outputs to a unified prediction tensor:
          preds: list of (B, nc + 4*reg_max, H, W)
        returns:
          y: (B, N, 4 + nc) where boxes are xyxy in input-pixel coordinates
        """
        device = preds[0].device
        bs = preds[0].shape[0]
        out = []
        reg_ch = 4 * self.reg_max

        for i, p in enumerate(preds):
            reg_out = p[:, :reg_ch]
            cls_out = p[:, reg_ch:]
            _, _, H, W = cls_out.shape
            stride = self.strides[i]

            # grid centers in pixels
            gy, gx = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing="ij"
            )
            cx = (gx + 0.5) * stride
            cy = (gy + 0.5) * stride
            cx = cx.view(-1)
            cy = cy.view(-1)

            # DFL integral -> distances (pixels)
            dist = self.dfl_integral(reg_out) * stride  # (B,4,H,W)
            dist = dist.permute(0, 2, 3, 1).reshape(bs, -1, 4)  # (B, HW, 4)

            x1 = cx[None, :] - dist[:, :, 0]
            y1 = cy[None, :] - dist[:, :, 1]
            x2 = cx[None, :] + dist[:, :, 2]
            y2 = cy[None, :] + dist[:, :, 3]
            boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # (B, HW, 4)

            cls = cls_out.permute(0, 2, 3, 1).reshape(bs, -1, self.nc).sigmoid()  # (B, HW, nc)
            out.append(torch.cat([boxes, cls], dim=-1))  # (B, HW, 4+nc)

        y = torch.cat(out, dim=1)  # (B, N, 4+nc)
        return y

    def decode_dets(self, preds: List[torch.Tensor], conf_thr: float = 0.25) -> List[torch.Tensor]:
        """
        Decode merged head outputs to per-image detections:
          returns List[Tensor(N,6)] with columns [x1,y1,x2,y2,conf,cls]
        """
        y = self.decode(preds, conf_thr=0.0)  # (B, N, 4+nc)
        bs = y.shape[0]
        dets = []
        for b in range(bs):
            boxes = y[b, :, :4]
            cls_prob = y[b, :, 4:]
            conf, cls_id = cls_prob.max(dim=-1)
            mask = conf > conf_thr
            if mask.any():
                det = torch.cat([boxes[mask], conf[mask].unsqueeze(1), cls_id[mask].float().unsqueeze(1)], dim=1)
            else:
                det = torch.zeros((0, 6), device=y.device)
            dets.append(det)
        return dets

def bbox_iou(box1: torch.Tensor, box2: torch.Tensor, *, CIoU: bool = False, eps: float = 1e-7) -> torch.Tensor:
    """
    IoU / CIoU for xyxy boxes. Supports broadcasting.
    Args:
        box1: (...,4) xyxy
        box2: (...,4) xyxy
    """
    # IoU
    inter = (torch.min(box1[..., 2:], box2[..., 2:]) - torch.max(box1[..., :2], box2[..., :2])).clamp(min=0)
    inter_area = inter[..., 0] * inter[..., 1]
    w1 = (box1[..., 2] - box1[..., 0]).clamp(min=0)
    h1 = (box1[..., 3] - box1[..., 1]).clamp(min=0)
    w2 = (box2[..., 2] - box2[..., 0]).clamp(min=0)
    h2 = (box2[..., 3] - box2[..., 1]).clamp(min=0)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter_area + eps
    iou = inter_area / union
    if not CIoU:
        return iou

    # CIoU
    cx1 = (box1[..., 0] + box1[..., 2]) * 0.5
    cy1 = (box1[..., 1] + box1[..., 3]) * 0.5
    cx2 = (box2[..., 0] + box2[..., 2]) * 0.5
    cy2 = (box2[..., 1] + box2[..., 3]) * 0.5
    rho2 = (cx2 - cx1) ** 2 + (cy2 - cy1) ** 2

    c_x1 = torch.min(box1[..., 0], box2[..., 0])
    c_y1 = torch.min(box1[..., 1], box2[..., 1])
    c_x2 = torch.max(box1[..., 2], box2[..., 2])
    c_y2 = torch.max(box1[..., 3], box2[..., 3])
    c2 = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2 + eps

    v = (4 / (math.pi ** 2)) * (torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps))) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    ciou = iou - (rho2 / c2 + alpha * v)
    return ciou


class TaskAlignedAssigner(nn.Module):
    """
    YOLOv8/PP-YOLOE-style TaskAlignedAssigner (anchor-free).
      metric = (cls_score ** alpha) * (iou ** beta)
      - candidates: anchor points inside GT
      - select topk per GT by metric
      - resolve conflicts by highest IoU
      - classification target score: (metric / metric_max_per_gt) * iou
    """
    def __init__(self, topk: int = 10, alpha: float = 0.5, beta: float = 6.0):
        super().__init__()
        self.topk = int(topk)
        self.alpha = float(alpha)
        self.beta = float(beta)

    @torch.no_grad()
    def forward(self,
                pred_scores: torch.Tensor,   # (N, nc) sigmoid probs
                pred_bboxes: torch.Tensor,   # (N, 4)  xyxy (pixels)
                anchor_points: torch.Tensor, # (N, 2)  (x,y) pixels
                gt_labels: torch.Tensor,     # (M,)
                gt_bboxes: torch.Tensor,     # (M,4) xyxy pixels
                num_classes: int):
        device = pred_bboxes.device
        eps = 1e-9
        N = pred_bboxes.shape[0]
        M = gt_bboxes.shape[0]

        target_bboxes = torch.zeros((N, 4), device=device, dtype=pred_bboxes.dtype)
        target_scores = torch.zeros((N, num_classes), device=device, dtype=pred_scores.dtype)
        fg_mask = torch.zeros((N,), device=device, dtype=torch.bool)
        target_gt_idx = torch.zeros((N,), device=device, dtype=torch.long)

        if M == 0 or N == 0:
            return target_bboxes, target_scores, fg_mask, target_gt_idx

        gt_labels = gt_labels.to(device=device, dtype=torch.long).clamp(min=0, max=num_classes - 1)
        gt_bboxes = gt_bboxes.to(device=device, dtype=pred_bboxes.dtype)

        # anchors inside GT boxes
        x, y = anchor_points[:, 0], anchor_points[:, 1]
        l = x[:, None] - gt_bboxes[None, :, 0]
        t = y[:, None] - gt_bboxes[None, :, 1]
        r = gt_bboxes[None, :, 2] - x[:, None]
        b = gt_bboxes[None, :, 3] - y[:, None]
        in_gts = (l > 0) & (t > 0) & (r > 0) & (b > 0)  # (N,M)

        # IoU between predicted boxes and GT
        ious = bbox_iou(pred_bboxes[:, None, :], gt_bboxes[None, :, :], CIoU=False)  # (N,M)

        # classification score aligned with GT class
        cls_score = pred_scores[:, gt_labels]  # (N,M)
        align_metric = (cls_score.clamp(min=0) ** self.alpha) * (ious.clamp(min=0) ** self.beta)

        # only consider anchors inside GT
        align_metric = align_metric.clone()
        align_metric[~in_gts] = -1e8

        k = min(self.topk, N)
        topk_idx = torch.topk(align_metric, k=k, dim=0, largest=True).indices  # (k,M)
        mask_topk = torch.zeros((N, M), device=device, dtype=torch.bool)
        mask_topk[topk_idx, torch.arange(M, device=device)] = True
        mask_pos = mask_topk & in_gts

        # resolve conflicts: assign each anchor to GT with highest IoU among its positives
        ious_pos = ious.clone()
        ious_pos[~mask_pos] = -1.0
        max_iou, gt_idx = ious_pos.max(dim=1)  # (N,)
        fg_mask = max_iou > -0.5
        target_gt_idx = gt_idx.clamp(min=0)

        if fg_mask.any():
            # per-GT max metric for normalization
            metric_pos = align_metric.clone()
            metric_pos[~mask_pos] = -1e8
            metric_max = metric_pos.max(dim=0).values.clamp(min=eps)  # (M,)

            fg_inds = fg_mask.nonzero(as_tuple=False).squeeze(1)
            gti = target_gt_idx[fg_inds]
            target_bboxes[fg_inds] = gt_bboxes[gti]

            # score: (metric / max_metric_per_gt) * iou
            score = (align_metric[fg_inds, gti] / metric_max[gti]) * ious[fg_inds, gti]
            score = score.clamp(min=0.0, max=1.0)
            tcls = gt_labels[gti].clamp(min=0, max=num_classes - 1)
            target_scores[fg_inds, tcls] = score

        return target_bboxes, target_scores, fg_mask, target_gt_idx


class YoloLoss(nn.Module):
    """
    YOLOv8-style loss:
      - TaskAlignedAssigner for matching (cls-aligned + IoU-aligned)
      - quality-guided cls targets (soft labels)
      - IoU/CIoU box loss + DFL, normalized by sum(target_scores)
    """
    def __init__(self, num_classes: int, reg_max: int = 16,
                 lambda_box: float = 7.5, lambda_cls: float = 0.5, lambda_dfl: float = 1.5,
                 tal_topk: int = 10, tal_alpha: float = 0.5, tal_beta: float = 6.0):
        super().__init__()
        self.nc = int(num_classes)
        self.reg_max = int(reg_max)
        self.l_box = float(lambda_box)
        self.l_cls = float(lambda_cls)
        self.l_dfl = float(lambda_dfl)

        self.assigner = TaskAlignedAssigner(topk=tal_topk, alpha=tal_alpha, beta=tal_beta)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

        # bins projection for DFL decode (0..reg_max-1)
        self.register_buffer("dfl_proj", torch.arange(self.reg_max, dtype=torch.float32), persistent=False)

    def _make_anchors(self, preds: List[torch.Tensor], strides: List[int], offset: float = 0.5):
        device = preds[0].device
        dtype = preds[0].dtype
        anchor_points = []
        stride_tensor = []
        for i, p in enumerate(preds):
            _, _, h, w = p.shape
            stride = float(strides[i])
            gy, gx = torch.meshgrid(
                torch.arange(h, device=device),
                torch.arange(w, device=device),
                indexing="ij"
            )
            points = torch.stack([(gx + offset) * stride, (gy + offset) * stride], dim=-1).view(-1, 2)  # (HW,2)
            anchor_points.append(points)
            stride_tensor.append(torch.full((h * w, 1), stride, device=device, dtype=dtype))
        return torch.cat(anchor_points, dim=0), torch.cat(stride_tensor, dim=0)

    def _decode_bboxes(self,
                       anchor_points: torch.Tensor,  # (N,2) pixels
                       pred_dist: torch.Tensor,      # (B,N,4,reg_max) logits
                       stride_tensor: torch.Tensor): # (N,1) pixels
        B, N, _, _ = pred_dist.shape
        proj = self.dfl_proj.to(device=pred_dist.device, dtype=pred_dist.dtype)  # (reg_max,)
        prob = pred_dist.softmax(-1)
        dist = (prob * proj).sum(-1)  # (B,N,4) in bins (grid units)
        dist = dist * stride_tensor.view(1, N, 1)  # pixels
        x = anchor_points[:, 0].view(1, N)
        y = anchor_points[:, 1].view(1, N)
        x1 = x - dist[:, :, 0]
        y1 = y - dist[:, :, 1]
        x2 = x + dist[:, :, 2]
        y2 = y + dist[:, :, 3]
        return torch.stack([x1, y1, x2, y2], dim=-1)  # (B,N,4) pixels

    def _bbox2dist(self,
                   anchor_points: torch.Tensor,  # (n,2) pixels
                   target_bboxes: torch.Tensor,  # (n,4) pixels
                   stride: torch.Tensor):        # (n,1) pixels
        # distances in bins (divide by stride)
        x = anchor_points[:, 0]
        y = anchor_points[:, 1]
        l = (x - target_bboxes[:, 0]) / stride.squeeze(1)
        t = (y - target_bboxes[:, 1]) / stride.squeeze(1)
        r = (target_bboxes[:, 2] - x) / stride.squeeze(1)
        b = (target_bboxes[:, 3] - y) / stride.squeeze(1)
        dist = torch.stack([l, t, r, b], dim=-1)
        # clamp to representable DFL bin range [0, reg_max-1]
        # (distances larger than reg_max-1 saturate to the last bin)
        return dist.clamp(min=0.0, max=float(self.reg_max - 1))

    def _df_loss(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred_dist: (n,4,reg_max) logits
        target:    (n,4) float in [0, reg_max)
        returns:   (n,) per-anchor DFL loss (sum over 4 edges)
        """
        n = target.shape[0]
        if n == 0:
            return torch.zeros((0,), device=pred_dist.device, dtype=pred_dist.dtype)
        # Keep targets in [0, reg_max-1] and handle the right-edge case (== reg_max-1)
        # so DFL still trains the last bin with weight 1.0 rather than degenerating to 0.
        target = target.clamp(min=0.0, max=float(self.reg_max - 1))
        left = target.floor().long()
        right = (left + 1).clamp(max=self.reg_max - 1)
        wl = right.float() - target
        wr = target - left.float()
        same_bin = right == left  # happens at the upper boundary (and only there with reg_max>1)
        wl = torch.where(same_bin, torch.ones_like(wl), wl)
        wr = torch.where(same_bin, torch.zeros_like(wr), wr)

        pred = pred_dist.reshape(-1, self.reg_max)  # (n*4, reg_max)
        loss_l = F.cross_entropy(pred, left.reshape(-1), reduction="none")
        loss_r = F.cross_entropy(pred, right.reshape(-1), reduction="none")
        loss = loss_l * wl.reshape(-1) + loss_r * wr.reshape(-1)
        return loss.view(n, 4).sum(dim=1)

    def forward(self, preds: List[torch.Tensor], targets: List[Dict], strides: List[int], device=None):
        device = preds[0].device
        B = preds[0].shape[0]

        reg_ch = 4 * self.reg_max
        # flatten per-level outputs
        cls_logits_all = []
        reg_logits_all = []
        for p in preds:
            reg = p[:, :reg_ch]
            cls = p[:, reg_ch:]
            cls_logits_all.append(cls.permute(0, 2, 3, 1).reshape(B, -1, self.nc))
            reg_logits_all.append(reg.permute(0, 2, 3, 1).reshape(B, -1, 4, self.reg_max))
        cls_logits = torch.cat(cls_logits_all, dim=1)  # (B,N,nc)
        reg_logits = torch.cat(reg_logits_all, dim=1)  # (B,N,4,reg_max)

        anchor_points, stride_tensor = self._make_anchors(preds, strides, offset=0.5)  # (N,2), (N,1)
        pred_scores = cls_logits.sigmoid()  # (B,N,nc)
        pred_bboxes = self._decode_bboxes(anchor_points, reg_logits, stride_tensor)  # (B,N,4) pixels

        N = anchor_points.shape[0]
        target_bboxes = torch.zeros((B, N, 4), device=device, dtype=pred_bboxes.dtype)
        target_scores = torch.zeros((B, N, self.nc), device=device, dtype=pred_scores.dtype)
        fg_mask = torch.zeros((B, N), device=device, dtype=torch.bool)

        for b in range(B):
            gt_bboxes = targets[b]["boxes"].to(device)
            gt_labels = targets[b]["labels"].to(device)
            if gt_bboxes.numel() == 0:
                continue
            t_bboxes, t_scores, t_fg, _ = self.assigner(
                pred_scores[b], pred_bboxes[b], anchor_points, gt_labels, gt_bboxes, self.nc
            )
            target_bboxes[b] = t_bboxes
            target_scores[b] = t_scores
            fg_mask[b] = t_fg

        target_scores_sum = target_scores.sum().clamp(min=1.0)

        # cls loss over all anchors (targets are soft scores)
        loss_cls = self.bce(cls_logits, target_scores).sum() / target_scores_sum

        # box + dfl loss over positives
        if fg_mask.any():
            bbox_weight = target_scores.sum(dim=-1)[fg_mask]  # (n_pos,)
            pred_b = pred_bboxes[fg_mask]
            tgt_b = target_bboxes[fg_mask]
            iou = bbox_iou(pred_b, tgt_b, CIoU=True)
            loss_box = ((1.0 - iou) * bbox_weight).sum() / target_scores_sum

            pred_d = reg_logits[fg_mask]  # (n_pos,4,reg_max)
            pts = anchor_points.view(1, N, 2).expand(B, N, 2)[fg_mask]  # (n_pos,2)
            st = stride_tensor.view(1, N, 1).expand(B, N, 1)[fg_mask]   # (n_pos,1)
            tgt_dist = self._bbox2dist(pts, tgt_b, st)  # (n_pos,4)
            dfl = self._df_loss(pred_d, tgt_dist)       # (n_pos,)
            loss_dfl = (dfl * bbox_weight).sum() / target_scores_sum
        else:
            loss_box = torch.zeros((), device=device)
            loss_dfl = torch.zeros((), device=device)

        loss = self.l_box * loss_box + self.l_cls * loss_cls + self.l_dfl * loss_dfl
        parts = {"l_box": float(loss_box.detach()), "l_cls": float(loss_cls.detach()), "l_dfl": float(loss_dfl.detach())}
        return loss, parts

class DualYoloV8L(nn.Module):
    def __init__(self, num_classes: int, imgsz: int = 640):
        super().__init__()
        self.rgb_backbone = BackboneV8Large(3)
        self.ir_backbone  = BackboneV8Large(3)

        # 独立的 YOLOv8-L 颈部（参数不共享）
        self.rgb_neck = YoloV8NeckSingle(C3=256, C4=512, C5=1024)
        self.ir_neck  = YoloV8NeckSingle(C3=256, C4=512, C5=1024)

        self.fusion_neck = DualFusionNeckXAttnL(
            C3c=256, C4c=512, C5c=1024
        )

        self.strides = [8, 16, 32]
        # anchor-free 检测头（融合颈部输出保持 256/512/1024 三尺度通道）
        self.detect = DetectHead(num_classes, ch=[256, 512, 1024], reg_max=16, strides=self.strides, img_size=imgsz)

    def forward(self, rgb, ir):
        # 双模态各自独立的 backbone + neck
        r3, r4, r5 = self.rgb_backbone(rgb)
        i3, i4, i5 = self.ir_backbone(ir)

        r_feats = self.rgb_neck((r3, r4, r5))
        i_feats = self.ir_neck((i3, i4, i5))

        # 融合颈部在 neck 输出处进行跨模态门控
        F3, F4, F5 = self.fusion_neck(r_feats, i_feats)
        outs = self.detect([F3, F4, F5])
        return outs

def decode_predictions(preds, strides, conf_thr=0.25, img_size=640, reg_max=16):
    """
    Anchor-free 解码：preds: List[(B, 4*reg_max+nc, H, W)] per level
    返回 [B] -> (N,6) [x1,y1,x2,y2,conf,cls]
    """
    device = preds[0].device
    bs = preds[0].shape[0]
    out_per_im = [[] for _ in range(bs)]
    reg_ch = 4 * reg_max

    for l, p in enumerate(preds):
        reg_out = p[:, :reg_ch]
        cls_out = p[:, reg_ch:]
        B, _, H, W = cls_out.shape
        stride = strides[l]
        # 构造网格中心
        gy, gx = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij"
        )
        cx = (gx + 0.5) * stride
        cy = (gy + 0.5) * stride
        cx = cx.view(-1)
        cy = cy.view(-1)

        # 回归距离
        reg_out = reg_out.permute(0, 2, 3, 1).reshape(B, -1, 4, reg_max)
        reg_prob = reg_out.softmax(-1)
        proj = reg_prob * torch.arange(reg_max, device=device)
        dist = proj.sum(-1) * stride  # (B, N, 4)

        x1 = cx[None, :] - dist[:, :, 0]
        y1 = cy[None, :] - dist[:, :, 1]
        x2 = cx[None, :] + dist[:, :, 2]
        y2 = cy[None, :] + dist[:, :, 3]

        cls_prob = cls_out.permute(0, 2, 3, 1).reshape(B, -1, cls_out.size(1)).sigmoid()
        conf, cls_id = cls_prob.max(dim=-1)

        for b in range(B):
            mask = conf[b] > conf_thr
            if not mask.any():
                continue
            det = torch.stack([
                x1[b, mask], y1[b, mask], x2[b, mask], y2[b, mask],
                conf[b, mask],
                cls_id[b, mask].float()
            ], dim=-1)
            out_per_im[b].append(det)

    for b in range(bs):
        if len(out_per_im[b]) > 0:
            out_per_im[b] = torch.cat(out_per_im[b], dim=0)
        else:
            out_per_im[b] = torch.zeros((0, 6), device=device)
    return out_per_im

def nms_axis_aligned(dets: torch.Tensor, iou_thr=0.5, topk=300):
    if dets.numel()==0: return dets
    keep = []
    for cls in dets[:,5].unique():
        D = dets[dets[:,5]==cls]
        order = torch.argsort(D[:,4], descending=True)
        D = D[order]
        while D.size(0) > 0 and len(keep) < topk:
            best = D[0:1]
            keep.append(best)
            if D.size(0) == 1: break
            ious = box_iou_axis_aligned(best[:,:4], D[1:,:4]).squeeze(0)
            D = D[1:][ious < iou_thr]
    return torch.cat(keep, dim=0) if keep else dets[:0]

def compute_pr_map(preds_all, gts_all, iou_thr=0.5, num_classes=len(CANONICAL_CLASSES), max_det=300):
    """
    preds_all: list[Tensor(M,6)] per image, [x1,y1,x2,y2,conf,cls]
    gts_all:   list[dict{'boxes':(N,4), 'labels':(N,)}] per image

    Returns:
        P:   macro-averaged precision over classes (float)
        R:   macro-averaged recall over classes   (float)
        mAP: mean AP at the given IoU threshold (COCO 101-pt)  (float)
        APs: list[float] per-class AP at the given IoU thr   (len = num_classes)
    """
    import numpy as np
    assert len(preds_all) == len(gts_all)
    all_scores = [[] for _ in range(num_classes)]
    all_tp = [[] for _ in range(num_classes)]
    all_fp = [[] for _ in range(num_classes)]
    total_positives = [0 for _ in range(num_classes)]

    for img_idx, (dets, gt) in enumerate(zip(preds_all, gts_all)):
        if dets.numel() == 0:
            for c in range(num_classes):
                total_positives[c] += int((gt["labels"] == c).sum().item())
            continue

        if dets.size(0) > max_det:
            order = torch.argsort(dets[:,4], descending=True)[:max_det]
            dets = dets[order]

        for c in range(num_classes):
            m_det = dets[:,5] == float(c)
            det_c = dets[m_det]
            gt_mask = (gt["labels"] == c)
            gt_c = gt["boxes"][gt_mask]
            total_positives[c] += int(gt_mask.sum().item())

            if det_c.numel() == 0:
                continue

            order = torch.argsort(det_c[:,4], descending=True)
            det_c = det_c[order]
            scores = det_c[:,4].tolist()

            if gt_c.numel() == 0:
                all_scores[c].extend(scores)
                all_tp[c].extend([0]*len(scores))
                all_fp[c].extend([1]*len(scores))
            else:
                used = torch.zeros((gt_c.size(0),), dtype=torch.bool, device=gt_c.device)
                tps = []
                fps = []
                for k in range(det_c.size(0)):
                    ious = box_iou_axis_aligned(det_c[k:k+1,:4], gt_c).squeeze(0)
                    iou_val, idx = ious.max(dim=0)
                    if iou_val.item() >= iou_thr and not used[idx]:
                        tps.append(1); fps.append(0); used[idx] = True
                    else:
                        tps.append(0); fps.append(1)
                all_scores[c].extend(scores)
                all_tp[c].extend(tps)
                all_fp[c].extend(fps)

    APs, precs, recs = [], [], []
    for c in range(num_classes):
        if len(all_scores[c]) == 0:
            APs.append(0.0); precs.append(0.0); recs.append(0.0); continue
        scores = np.array(all_scores[c])
        tp = np.array(all_tp[c], dtype=np.int32)
        fp = np.array(all_fp[c], dtype=np.int32)

        order = np.argsort(-scores)
        tp = tp[order]; fp = fp[order]

        tp_c = np.cumsum(tp)
        fp_c = np.cumsum(fp)
        denom = np.maximum(tp_c + fp_c, 1e-9)
        prec = tp_c / denom
        Npos = max(total_positives[c], 1)
        rec = tp_c / Npos

        # COCO-style 101-point interpolated AP
        ap = 0.0
        for t in np.linspace(0.0, 1.0, 101):
            mask = rec >= t
            ap += np.max(prec[mask]) if mask.any() else 0.0
        ap /= 101.0
        APs.append(float(ap))
        precs.append(float(prec[-1]))
        recs.append(float(rec[-1]))

    mAP = float(np.mean(APs)) if APs else 0.0
    P = float(np.mean(precs)) if precs else 0.0
    R = float(np.mean(recs)) if recs else 0.0
    return P, R, mAP, APs

def train(args):
    os.makedirs(args.logdir, exist_ok=True)

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if getattr(args, "local_rank", -1) >= 0:
            local_rank = int(args.local_rank)

        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        dist.init_process_group(backend=backend, init_method="env://")

        rank = get_rank()
        world_size = get_world_size()
        log_file = os.path.join(args.logdir, "training.log" if rank == 0 else f"training_rank{rank}.log")
        logger = setup_logger(log_file, is_main=(rank == 0))
        logger.info(f"DDP enabled | backend={backend} rank={rank}/{world_size} local_rank={local_rank} device={device}")
    else:
        rank = 0
        world_size = 1
        logger = setup_logger(os.path.join(args.logdir, "training.log"), is_main=True)

        # ----- GPU selection (single process) -----
        gpus = parse_gpus_arg(getattr(args, "gpus", ""))
        if len(gpus) > 0:
            if not torch.cuda.is_available():
                logger.warning("`--gpus` 已设置但 CUDA 不可用，将回退到 CPU。")
                device = torch.device("cpu")
            else:
                dev_cnt = torch.cuda.device_count()
                for gid in gpus:
                    if gid < 0 or gid >= dev_cnt:
                        raise RuntimeError(f"--gpus 包含无效 GPU id: {gid}（共 {dev_cnt} 张）。")
                torch.cuda.set_device(gpus[0])
                device = torch.device(f"cuda:{gpus[0]}")
                if len(gpus) > 1:
                    logger.info(f"Using DataParallel on GPUs {gpus} (primary cuda:{gpus[0]})")
                else:
                    logger.info(f"Using CUDA device cuda:{gpus[0]} — {torch.cuda.get_device_name(device)}")
        elif args.gpu < 0 or not torch.cuda.is_available():
            device = torch.device("cpu")
            logger.info("Using CPU (either --gpu < 0 or CUDA not available).")
        else:
            if args.gpu >= torch.cuda.device_count():
                raise RuntimeError(f"--gpu={args.gpu} 超出可用 GPU 数量（共 {torch.cuda.device_count()} 张）。")
            torch.cuda.set_device(args.gpu)
            device = torch.device(f"cuda:{args.gpu}")
            logger.info(f"Using CUDA device cuda:{args.gpu} — {torch.cuda.get_device_name(device)}")

    # Seeds
    seed = int(args.seed) + int(rank)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    # Data
    train_ds = RGBIRDetDataset(args.train_csv, imgsz=args.imgsz, augment=True)
    val_ds   = RGBIRDetDataset(args.val_csv,   imgsz=args.imgsz, augment=False)

    # Model
    model = DualYoloV8L(num_classes=len(CANONICAL_CLASSES), imgsz=args.imgsz).to(device)
    optim = build_optimizer(model, args)
    criterion = YoloLoss(len(CANONICAL_CLASSES), reg_max=16)
    ema = ModelEMA(model, decay=args.ema_decay, tau=args.ema_tau) if args.ema else None

    # ---------- 自动批大小估计，尽量吃满显存但不 OOM ----------
    if device.type == "cuda" and args.auto_batch and (not distributed or is_main_process()):
        torch.cuda.empty_cache()
        try:
            free_mem, total_mem = torch.cuda.mem_get_info(device)
            logger.info(f"Auto batch | free={free_mem/1024**3:.2f} GiB total={total_mem/1024**3:.2f} GiB, target frac={args.max_vram_frac}")

            sample_rgb, sample_ir, sample_tgt, _ = train_ds[0]
            base_b = max(1, min(args.batch, 4))
            rgb_b = sample_rgb.unsqueeze(0).repeat(base_b, 1, 1, 1).to(device)
            ir_b  = sample_ir.unsqueeze(0).repeat(base_b, 1, 1, 1).to(device)
            tgt_b = []
            for _ in range(base_b):
                tgt_b.append({
                    "boxes": sample_tgt["boxes"].to(device),
                    "labels": sample_tgt["labels"].to(device),
                    "orig_size": sample_tgt["orig_size"].to(device),
                })

            model.train()
            optim.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats(device)
            out_tmp = model(rgb_b, ir_b)
            loss_tmp, _ = criterion(out_tmp, tgt_b, model.strides)
            loss_tmp.backward()
            peak = torch.cuda.max_memory_allocated(device)
            mem_per_sample = peak / base_b
            torch.cuda.empty_cache()

            target_mem = total_mem * args.max_vram_frac
            est_batch = int(target_mem // max(mem_per_sample, 1))
            est_batch = max(1, est_batch)
            est_batch = min(est_batch, args.batch * 3)  # 避免倍数过大
            logger.info(f"Auto batch | peak={peak/1024**3:.2f} GiB for {base_b} -> {mem_per_sample/1024**3:.3f} GiB/sample, suggested batch={est_batch}")
            args.batch = est_batch
        except Exception as e:
            logger.warning(f"Auto batch failed: {e}. 使用用户设定 batch={args.batch}")
            torch.cuda.empty_cache()

    # DDP 下把 batch 同步到所有进程（global batch = args.batch * world_size）
    if distributed:
        b = torch.tensor([int(args.batch)], device=device, dtype=torch.int64)
        dist.broadcast(b, src=0)
        args.batch = int(b.item())

    # DataLoaders (放到 auto-batch 之后，确保 batch_size 生效)
    if distributed:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch,
            shuffle=False,
            sampler=train_sampler,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True if device.type == "cuda" else False,
        )
        # val_loader 仅 rank0 用于评估：不使用 DistributedSampler，避免只评估子集
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True if device.type == "cuda" else False,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True if device.type == "cuda" else False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True if device.type == "cuda" else False,
        )

    # Wrap model for multi-GPU
    if distributed:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    else:
        gpus = parse_gpus_arg(getattr(args, "gpus", ""))
        if device.type == "cuda" and len(gpus) > 1:
            model = nn.DataParallel(model, device_ids=gpus)

    nb = len(train_loader)
    warmup_iters = max(1, int(args.warmup_epochs * nb)) if args.warmup_epochs > 0 else 0
    if (not distributed) or is_main_process():
        logger.info(
            f"Recipe | opt={args.optimizer} lr0={args.lr0} lrf={args.lrf} "
            f"momentum={args.momentum} wd={args.weight_decay} "
            f"warmup_epochs={args.warmup_epochs} warmup_iters={warmup_iters} "
            f"ema={'on' if ema else 'off'}"
        )

    best_map = 0.0
    mosaic_closed = False
    for epoch in range(1, args.epochs+1):
        # YOLOv8-style: disable mosaic/mixup in the last N epochs
        if (not mosaic_closed) and getattr(args, "close_mosaic", 0) and epoch >= (args.epochs - int(args.close_mosaic) + 1):
            train_ds.mosaic = False
            train_ds.mixup_prob = 0.0
            mosaic_closed = True
            if (not distributed) or is_main_process():
                logger.info(f"Close mosaic/mixup for last {args.close_mosaic} epochs (epoch {epoch} -> {args.epochs})")
        model.train()
        if distributed:
            train_sampler.set_epoch(epoch)
        raw_model = unwrap_model(model)
        t0 = time.time()
        loss_meter = 0.0; meter = {"l_box":0.0,"l_cls":0.0,"l_dfl":0.0}
        nb_iter = 0
        for i, (rgbs, irs, targets, names) in enumerate(train_loader):
            # ---- YOLOv8-style warmup + cosine LR ----
            if warmup_iters > 0 and (epoch - 1) * nb + i < warmup_iters:
                ni = (epoch - 1) * nb + i
                xi = ni / warmup_iters
                warmup_lr = args.lr0 * cosine_lr_factor(0.0, args.lrf)
                for gi, pg in enumerate(optim.param_groups):
                    if gi == 2:
                        pg["lr"] = args.warmup_bias_lr + xi * (warmup_lr - args.warmup_bias_lr)
                    else:
                        pg["lr"] = xi * warmup_lr
                    if "momentum" in pg:
                        pg["momentum"] = args.warmup_momentum + xi * (args.momentum - args.warmup_momentum)
            else:
                progress = (epoch - 1 + i / max(nb, 1)) / max(args.epochs, 1)
                lr = args.lr0 * cosine_lr_factor(progress, args.lrf)
                for pg in optim.param_groups:
                    pg["lr"] = lr

            rgbs = rgbs.to(device, non_blocking=True if device.type=='cuda' else False)
            irs  = irs.to(device,  non_blocking=True if device.type=='cuda' else False)
            tgts = [{k:v.to(device) if torch.is_tensor(v) else v for k,v in t.items()} for t in targets]

            preds = model(rgbs, irs)
            loss, parts = criterion(preds, tgts, raw_model.strides)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optim.step()
            if ema is not None:
                ema.update(unwrap_model(model))

            loss_meter += loss.item(); nb_iter += 1
            for k in meter: meter[k]+= parts[k]

        if distributed:
            # reduce metrics for logging
            t = torch.tensor(
                [loss_meter, meter["l_box"], meter["l_cls"], meter["l_dfl"], float(nb_iter)],
                device=device,
                dtype=torch.float32,
            )
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            denom = max(t[4].item(), 1.0)
            loss_meter = t[0].item() / denom
            meter["l_box"] = t[1].item() / denom
            meter["l_cls"] = t[2].item() / denom
            meter["l_dfl"] = t[3].item() / denom
        dt = time.time()-t0
        if not distributed:
            loss_meter /= max(nb_iter,1)
            for k in meter: meter[k] /= max(nb_iter,1)
        if (not distributed) or is_main_process():
            logger.info(f"Epoch {epoch}/{args.epochs} | loss={loss_meter:.4f} (box {meter['l_box']:.2f} cls {meter['l_cls']:.2f} dfl {meter['l_dfl']:.2f}) | {dt:.1f}s")

        # ---- Evaluation (从第20个epoch开始) ----
        if distributed:
            dist.barrier()

        if epoch >= 20 and ((not distributed) or is_main_process()):
            raw_model = unwrap_model(model)
            eval_model = ema.ema if ema is not None else raw_model
            eval_model.eval()
            preds_all = []; gts_all = []
            with torch.no_grad():
                for rgbs, irs, targets, names in val_loader:
                    rgbs = rgbs.to(device); irs = irs.to(device)
                    outs = eval_model(rgbs, irs)
                    dets = eval_model.detect.decode_dets(outs, conf_thr=args.conf_thres)
                    dets = [nms_axis_aligned(d, iou_thr=args.nms_iou, topk=args.max_det) for d in dets]
                    preds_all.extend(dets)
                    for t in targets:
                        g = { "boxes": t["boxes"].to(device), "labels": t["labels"].to(device) }
                        gts_all.append(g)

            P, R, mAP50, APs = compute_pr_map(
                preds_all, gts_all, iou_thr=0.5,
                num_classes=len(CANONICAL_CLASSES),
                max_det=args.max_det
            )

            idx_map = {c: i for i, c in enumerate(CANONICAL_CLASSES)}
            ap_table = {c: APs[idx_map[c]] for c in CANONICAL_CLASSES}
            ap_log = " ".join([f"{c.replace('_','-').title()}={ap_table[c]:.4f}" for c in CANONICAL_CLASSES])

            logger.info(f"Epoch {epoch} VAL | mAP50={mAP50:.4f} | AP50: {ap_log} | P={P:.4f} R={R:.4f}")

            if mAP50 > best_map:
                best_map = mAP50
                save_path = os.path.join(args.logdir, "best.pt")
                state = {
                    "model": (ema.ema.state_dict() if ema is not None else raw_model.state_dict()),
                    "classes": CANONICAL_CLASSES,
                    "epoch": epoch,
                    "metrics": {
                        "P": P, "R": R, "mAP50": mAP50,
                        "AP50_per_class": ap_table
                    }
                }
                if ema is not None:
                    state["model_raw"] = raw_model.state_dict()
                torch.save(state, save_path)
                logger.info(f"Saved best to {save_path} (mAP50={mAP50:.4f})")
        elif (not distributed) or is_main_process():
            logger.info(f"Epoch {epoch}/{args.epochs} | Skipping evaluation (will start from epoch 20)")

        if distributed:
            dist.barrier()

    if distributed:
        dist.barrier()

    if (not distributed) or is_main_process():
        final_path = os.path.join(args.logdir, "last.pt")
        raw_model = unwrap_model(model)
        state = {"model": (ema.ema.state_dict() if ema is not None else raw_model.state_dict()),
                 "classes": CANONICAL_CLASSES}
        if ema is not None:
            state["model_raw"] = raw_model.state_dict()
        torch.save(state, final_path)
        logger.info(f"Training finished. Final weights saved to {final_path}")

    if distributed and is_dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()

def get_args():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, default="../LLVIP/llvip_train_paths.csv")
    ap.add_argument("--val_csv",   type=str, default="../LLVIP/llvip_test_paths.csv")
    ap.add_argument("--imgsz",     type=int, default=640)
    ap.add_argument("--epochs",    type=int, default=200)
    ap.add_argument("--batch",     type=int, default=20)
    ap.add_argument("--num_workers", type=int, default=4)
    # ---- YOLOv8-like recipe ----
    ap.add_argument("--lr0",       type=float, default=0.01, help="初始学习率 (YOLOv8 默认 0.01)")
    ap.add_argument("--lrf",       type=float, default=0.01, help="最终学习率比例 (lr_final = lr0 * lrf)")
    ap.add_argument("--momentum",  type=float, default=0.937, help="SGD momentum (YOLOv8 默认 0.937)")
    ap.add_argument("--weight_decay", type=float, default=5e-4, help="权重衰减 (YOLOv8 默认 5e-4)")
    ap.add_argument("--warmup_epochs", type=float, default=3.0, help="warmup 轮数 (YOLOv8 默认 3.0)")
    ap.add_argument("--warmup_momentum", type=float, default=0.8, help="warmup 初始 momentum")
    ap.add_argument("--warmup_bias_lr", type=float, default=0.1, help="bias warmup 起始学习率")
    ap.add_argument("--optimizer", type=str, default="SGD", help="优化器: SGD 或 AdamW")
    ap.add_argument("--ema", action="store_true", default=True, help="启用 EMA (默认开启)")
    ap.add_argument("--no-ema", dest="ema", action="store_false", help="关闭 EMA")
    ap.add_argument("--ema_decay", type=float, default=0.9999, help="EMA decay (YOLOv8 默认 0.9999)")
    ap.add_argument("--ema_tau", type=float, default=2000.0, help="EMA tau (YOLOv8 默认 2000)")
    ap.add_argument("--lr",        type=float, default=None, help="兼容旧参数：等价于 --lr0")
    ap.add_argument("--conf_thres",type=float, default=0.001)
    ap.add_argument("--nms_iou",   type=float, default=0.5)
    ap.add_argument("--logdir",    type=str, default="./runs/0126_warm")
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--gpu",       type=int, default=0, help="使用第几张 GPU（0 开始）。设为 -1 强制使用 CPU。")
    ap.add_argument("--gpus",      type=str, default="", help="单进程多 GPU（DataParallel），例如：--gpus 0,1,2。推荐优先用 DDP(torchrun)。")
    ap.add_argument("--local_rank", type=int, default=-1, help=argparse.SUPPRESS)
    ap.add_argument("--max_det",       type=int, default=300)
    ap.add_argument("--close_mosaic",  type=int, default=10, help="训练最后 N 个 epoch 关闭 mosaic/mixup（YOLOv8 默认 close_mosaic=10）")
    ap.add_argument("--auto_batch", action="store_true", help="自动估算批大小以尽量占满显存")
    ap.add_argument("--max_vram_frac", type=float, default=0.9, help="自动批大小的目标显存占用比例 (0-1)")
    args = ap.parse_args()
    if args.lr is not None:
        args.lr0 = float(args.lr)
    return args

if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.logdir, exist_ok=True)
    train(args)
