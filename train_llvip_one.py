import os
import sys
import math
import time
import random
import logging
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False

# -------- Logging --------
def setup_logger(log_file: str):
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    fh = logging.FileHandler(log_file, encoding="utf-8"); fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger

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
                 mosaic: bool = True, mosaic_prob: float = 0.8, mixup_prob: float = 0.15,
                 mosaic_scale: Tuple[float, float] = (0.5, 1.5),
                 hsv_gain: Tuple[float, float, float] = (0.015, 0.7, 0.4),
                 spectral_prob: float = 0.5, spectral_strength: float = 0.2,
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

        rgb_t = torch.from_numpy(np.array(rgb)).permute(2,0,1).float()/255.0
        ir_t  = torch.from_numpy(np.array(ir )).permute(2,0,1).float()/255.0

        target = {
            "boxes": torch.from_numpy(boxes_scaled),
            "labels": torch.from_numpy(labels_np),
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
            rgb, ir, boxes = self._random_flip_np(rgb, ir, boxes)
        rgb, ir = self._apply_hsv(rgb, ir)
        rgb, ir = self._spectral_augment(rgb, ir)
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
            SPPF(1024, 1024, k=5)
        )

    def forward(self, x):
        x = self.layer1(x)
        P3 = self.layer2(x)   # 1/8,  256 ch
        P4 = self.layer3(P3)  # 1/16, 512 ch
        P5 = self.layer4(P4)  # 1/32, 1024 ch
        return P3, P4, P5
    
class XAttnGate(nn.Module):
    """
    轻量跨模态注意力门控：
      - 输入三路特征 (r, i, c)，shape=(B,C,H,W)，C 为该尺度目标通道
      - 先做 GAP 得到 tokens: t_r, t_i, t_c ∈ R^{B×C}
      - 线性映射到 d_model 维；Query=concat token，Key/Value=[r,i,c] 三个 token
      - 得到注意力权重 α∈R^{B×3}，经 softmax 作为三路门控

    gate_mode:
      - 'scalar'  : 返回三路标量权重 (B,3,1,1)
      - 'channel' : 先用注意力输出 O ∈ R^{B×d} 经过 MLP -> 3C，再按路径维 softmax，返回 (B,3,C,1,1)
    """
    def __init__(self, channels: int, d_model: int = 64, gate_mode: str = "scalar",
                 reduction: int = 16, dropout: float = 0.0):
        super().__init__()
        assert gate_mode in ("scalar", "channel")
        self.channels = channels
        self.d_model = d_model
        self.gate_mode = gate_mode

        # 三个路径的可学习路径嵌入（帮助区分 RGB/IR/Concat）
        self.path_embed = nn.Parameter(torch.zeros(3, d_model))

        # token 映射
        self.proj_q = nn.Linear(channels, d_model, bias=True)  # for concat token
        self.proj_k = nn.Linear(channels, d_model, bias=True)  # shared for [r,i,c]
        self.proj_v = nn.Linear(channels, d_model, bias=True)

        self.scale = d_model ** -0.5
        self.dp = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if gate_mode == "channel":
            hid = max(8, channels // reduction)
            # 由注意力输出 O(d_model) 生成三路逐通道权重 (3C)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, hid, bias=True),
                nn.SiLU(inplace=True),
                nn.Linear(hid, 3 * channels, bias=True),
            )
            # 让初始权重尽量接近均分
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

        # 让起始注意力更接近均匀
        nn.init.normal_(self.path_embed, std=1e-4)

    def forward(self, r: torch.Tensor, i: torch.Tensor, c: torch.Tensor):
        """
        Inputs: r, i, c: (B, C, H, W)
        Returns:
          - scalar: (wr, wi, wc) each (B,1,1,1)
          - channel: (Wr, Wi, Wc) each (B, C, 1, 1)
        """
        B, C, H, W = r.shape
        # GAP tokens: (B, C)
        tr = F.adaptive_avg_pool2d(r, 1).flatten(1)
        ti = F.adaptive_avg_pool2d(i, 1).flatten(1)
        tc = F.adaptive_avg_pool2d(c, 1).flatten(1)

        # 线性映射到 d_model
        q = self.proj_q(tc)                   # (B, d)
        k = torch.stack([self.proj_k(tr),     # (B, 3, d)
                         self.proj_k(ti),
                         self.proj_k(tc)], dim=1)
        v = torch.stack([self.proj_v(tr),     # (B, 3, d)
                         self.proj_v(ti),
                         self.proj_v(tc)], dim=1)

        # 加路径嵌入
        k = k + self.path_embed.view(1, 3, self.d_model)
        v = v + self.path_embed.view(1, 3, self.d_model)

        # 注意力 logits: (B, 1, d) x (B, d, 3) -> (B, 1, 3)
        attn_logits = (q.unsqueeze(1) @ k.transpose(1, 2)) * self.scale
        attn = torch.softmax(attn_logits, dim=-1).squeeze(1)  # (B, 3)
        attn = self.dp(attn)

        if self.gate_mode == "scalar":
            # 标量权重 (wr, wi, wc) -> (B,1,1,1)
            wr = attn[:, 0].view(B, 1, 1, 1)
            wi = attn[:, 1].view(B, 1, 1, 1)
            wc = attn[:, 2].view(B, 1, 1, 1)
            return wr, wi, wc
        else:
            # channel 权重：先得到注意力输出 O = attn @ V ∈ (B, d)
            O = (attn.unsqueeze(1) @ v).squeeze(1)  # (B, d)
            g = self.mlp(O)                         # (B, 3C)
            g = g.view(B, 3, C)                     # (B,3,C)
            g = torch.softmax(g, dim=1)             # 沿路径维 softmax
            Wr = g[:, 0].unsqueeze(-1).unsqueeze(-1)  # (B,C,1,1)
            Wi = g[:, 1].unsqueeze(-1).unsqueeze(-1)
            Wc = g[:, 2].unsqueeze(-1).unsqueeze(-1)
            return Wr, Wi, Wc


class DualFusionNeckXAttnL(nn.Module):
    """
    L 规模三路融合颈部：
      - lateral 统一到 512
      - smooth 用 C2f(512->512)
      - F3/F4/F5 目标通道 = 256 / 512 / 1024
    """
    def __init__(self, C3c=256, C4c=512, C5c=1024,
                 gate_mode: str = "channel",  # 或 'scalar'
                 d_model: int = 96,
                 reduction: int = 16):
        super().__init__()
        # 侧连统一到 512
        self.lat5 = ConvBNAct(C5c, 512, 1, 1)  # 1024 -> 512
        self.lat4 = ConvBNAct(C4c, 512, 1, 1)  #  512 -> 512
        self.lat3 = ConvBNAct(C3c, 512, 1, 1)  #  256 -> 512

        # 顶到底 FPN 平滑
        self.smooth4 = C2f(512, 512, n=3, shortcut=True)
        self.smooth3 = C2f(512, 512, n=3, shortcut=True)

        # ---------- F5（20x20） ----------
        self.rgb_mlp5 = MLP(512, C5c, r=0.5)        # 512 -> 1024
        self.ir_mlp5  = MLP(512, C5c, r=0.5)        # 512 -> 1024
        self.cat_mlp5 = MLP(1024, 1024, r=0.5)      # 512+512 -> 1024
        self.fuse5    = ConvBNAct(1024, C5c, 1, 1)  # 1024 -> 1024

        # ---------- F4（40x40） ----------
        self.rgb_mlp4 = MLP(512, C4c, r=0.5)        # 512 -> 512
        self.ir_mlp4  = MLP(512, C4c, r=0.5)
        self.cat_mlp4 = MLP(1024, 1024, r=0.5)
        self.fuse4    = ConvBNAct(1024, C4c, 1, 1)  # 1024 -> 512

        # ---------- F3（80x80） ----------
        self.rgb_mlp3 = MLP(512, C3c, r=0.5)        # 512 -> 256
        self.ir_mlp3  = MLP(512, C3c, r=0.5)
        self.cat_mlp3 = MLP(1024, 1024, r=0.5)
        self.fuse3    = ConvBNAct(1024, C3c, 1, 1)  # 1024 -> 256

        # 跨模态注意力门控
        self.gate5 = XAttnGate(C5c, d_model=d_model, gate_mode=gate_mode, reduction=reduction)
        self.gate4 = XAttnGate(C4c, d_model=d_model, gate_mode=gate_mode, reduction=reduction)
        self.gate3 = XAttnGate(C3c, d_model=d_model, gate_mode=gate_mode, reduction=reduction)

    def forward(self, rgb_feats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                      ir_feats:  Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        r3, r4, r5 = rgb_feats
        i3, i4, i5 = ir_feats

        # lateral to 512
        R5 = self.lat5(r5); I5 = self.lat5(i5)
        R4 = self.lat4(r4); I4 = self.lat4(i4)
        R3 = self.lat3(r3); I3 = self.lat3(i3)

        # RGB top-down
        p4 = self.smooth4(R4 + F.interpolate(R5, scale_factor=2.0, mode="nearest"))
        p3 = self.smooth3(R3 + F.interpolate(p4, scale_factor=2.0, mode="nearest"))
        # IR top-down
        p4_i = self.smooth4(I4 + F.interpolate(I5, scale_factor=2.0, mode="nearest"))
        p3_i = self.smooth3(I3 + F.interpolate(p4_i, scale_factor=2.0, mode="nearest"))

        # F5
        r5_m = self.rgb_mlp5(R5)
        i5_m = self.ir_mlp5(I5)
        c5_m = self.fuse5(self.cat_mlp5(torch.cat([R5, I5], dim=1)))
        w5 = self.gate5(r5_m, i5_m, c5_m)
        F5 = r5_m * w5[0] + i5_m * w5[1] + c5_m * w5[2]

        # F4
        r4_m = self.rgb_mlp4(p4)
        i4_m = self.ir_mlp4(p4_i)
        c4_m = self.fuse4(self.cat_mlp4(torch.cat([p4, p4_i], dim=1)))
        w4 = self.gate4(r4_m, i4_m, c4_m)
        F4 = r4_m * w4[0] + i4_m * w4[1] + c4_m * w4[2]

        # F3
        r3_m = self.rgb_mlp3(p3)
        i3_m = self.ir_mlp3(p3_i)
        c3_m = self.fuse3(self.cat_mlp3(torch.cat([p3, p3_i], dim=1)))
        w3 = self.gate3(r3_m, i3_m, c3_m)
        F3 = r3_m * w3[0] + i3_m * w3[1] + c3_m * w3[2]

        return F3, F4, F5

class DetectHead(nn.Module):
    """
    Anchor-free 解耦头 + DFL（对齐 YOLOv8 思路）
      - reg 分支输出 4*(reg_max+1) 的离散距离分布
      - cls 分支输出 nc 类概率（无 obj 分支）
    """
    def __init__(self, nc: int, ch: List[int], reg_max: int = 16):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        self.nl = len(ch)

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
        self.reg_pred = nn.ModuleList([nn.Conv2d(c, 4 * (reg_max + 1), 1) for c in ch])

    def forward(self, feats: List[torch.Tensor]):
        outputs = []
        for i, x in enumerate(feats):
            x = self.stems[i](x)
            cls_feat = self.cls_conv[i](x)
            reg_feat = self.reg_conv[i](x)
            cls_out = self.cls_pred[i](cls_feat)          # (B, nc, H, W)
            reg_out = self.reg_pred[i](reg_feat)          # (B, 4*(reg_max+1), H, W)
            outputs.append((reg_out, cls_out))
        return outputs

class YoloLoss(nn.Module):
    """
    更贴近 YOLOv8 的 anchor-free + DFL 损失：
      - 正样本：中心落入 GT 且在中心半径范围内，若无则取距中心最近 topk
      - cls：Focal BCE，软标签为 IoU（TaskAligned 风格）
      - reg：IoU 损失；DFL：离散距离分布
    """
    def __init__(self, num_classes: int, reg_max: int = 16,
                 lambda_box=7.5, lambda_cls=1.0, lambda_dfl=1.5,
                 center_radius: float = 2.5, topk: int = 10):
        super().__init__()
        self.nc = num_classes
        self.reg_max = reg_max
        self.l_box = lambda_box
        self.l_cls = lambda_cls
        self.l_dfl = lambda_dfl
        self.center_radius = center_radius
        self.topk = topk

    @staticmethod
    def bbox_iou(box1, box2):
        inter = (torch.min(box1[..., 2:], box2[..., 2:]) - torch.max(box1[..., :2], box2[..., :2])).clamp(min=0)
        inter_area = inter[..., 0] * inter[..., 1]
        area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
        area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
        union = area1 + area2 - inter_area + 1e-6
        return inter_area / union

    def dfl_loss(self, pred, target):
        n, _, bins = pred.shape
        t = target.clamp(min=0, max=self.reg_max - 1e-4)
        l = t.floor().long()
        r = (l + 1).clamp(max=self.reg_max)
        wl = r.float() - t
        wr = t - l.float()
        pred = pred.reshape(n * 4, bins)
        l_loss = F.cross_entropy(pred, l.reshape(-1), reduction="none")
        r_loss = F.cross_entropy(pred, r.reshape(-1), reduction="none")
        return (l_loss * wl.view(-1) + r_loss * wr.view(-1)).mean()

    def focal_bce(self, logits, targets, alpha=0.25, gamma=2.0):
        prob = logits.sigmoid()
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        return ce * alpha_t * ((1 - p_t) ** gamma)

    def forward(self, preds, targets, strides, device=None):
        device = preds[0][0].device
        bs = len(targets)
        total_box = torch.zeros(1, device=device)
        total_cls = torch.zeros(1, device=device)
        total_dfl = torch.zeros(1, device=device)

        for b in range(bs):
            tboxes = targets[b]["boxes"].to(device)
            tcls   = targets[b]["labels"].to(device)
            if tboxes.numel() == 0:
                for _, cls_out in preds:
                    total_cls += F.binary_cross_entropy_with_logits(cls_out[b], torch.zeros_like(cls_out[b]), reduction="sum")
                continue

            for l, (reg_out, cls_out) in enumerate(preds):
                stride = strides[l]
                _, _, h, w = cls_out.shape
                gy, gx = torch.meshgrid(
                    torch.arange(h, device=device),
                    torch.arange(w, device=device),
                    indexing="ij"
                )
                centers = torch.stack([gx + 0.5, gy + 0.5], dim=-1).view(-1, 2)  # (HW,2)

                gt = tboxes / stride
                gtc = tcls
                gt_xy1 = gt[:, :2]
                gt_xy2 = gt[:, 2:4]
                gt_ctr = (gt_xy1 + gt_xy2) * 0.5

                left = centers[:, None, 0] - gt_xy1[None, :, 0]
                top = centers[:, None, 1] - gt_xy1[None, :, 1]
                right = gt_xy2[None, :, 0] - centers[:, None, 0]
                bottom = gt_xy2[None, :, 1] - centers[:, None, 1]
                in_gt = (left > 0) & (top > 0) & (right > 0) & (bottom > 0)

                radius = self.center_radius
                in_ctr = (centers[:, None, 0] >= (gt_ctr[None, :, 0] - radius)) & \
                         (centers[:, None, 0] <= (gt_ctr[None, :, 0] + radius)) & \
                         (centers[:, None, 1] >= (gt_ctr[None, :, 1] - radius)) & \
                         (centers[:, None, 1] <= (gt_ctr[None, :, 1] + radius))

                mask = in_gt & in_ctr

                if not mask.any():
                    dist = (centers[:, None, :] - gt_ctr[None, :, :]).pow(2).sum(-1).sqrt()
                    topk = min(self.topk, centers.size(0))
                    _, idx = dist.topk(topk, dim=0, largest=False)
                    mask = torch.zeros_like(dist, dtype=torch.bool)
                    mask.scatter_(0, idx, True)

                pos_idx = mask.nonzero(as_tuple=False)
                if pos_idx.numel() == 0:
                    total_cls += F.binary_cross_entropy_with_logits(cls_out[b], torch.zeros_like(cls_out[b]), reduction="sum")
                    continue

                point_idx = pos_idx[:, 0]
                gt_idx = pos_idx[:, 1]
                cxcy = centers[point_idx]
                assigned_gt = gt[gt_idx]
                assigned_cls = gtc[gt_idx]

                ltrb = torch.stack([
                    cxcy[:, 0] - assigned_gt[:, 0],
                    cxcy[:, 1] - assigned_gt[:, 1],
                    assigned_gt[:, 2] - cxcy[:, 0],
                    assigned_gt[:, 3] - cxcy[:, 1],
                ], dim=1).clamp(min=0)

                cls_flat = cls_out[b].permute(1, 2, 0).reshape(-1, self.nc)
                reg_flat = reg_out[b].permute(1, 2, 0).reshape(-1, 4, self.reg_max + 1)
                cls_sel = cls_flat[point_idx]
                reg_sel = reg_flat[point_idx]

                total_dfl += self.dfl_loss(reg_sel, ltrb)

                reg_prob = reg_sel.softmax(-1)
                proj = reg_prob * torch.arange(self.reg_max + 1, device=device)
                dist = proj.sum(-1)
                x1 = cxcy[:, 0] - dist[:, 0]
                y1 = cxcy[:, 1] - dist[:, 1]
                x2 = cxcy[:, 0] + dist[:, 2]
                y2 = cxcy[:, 1] + dist[:, 3]
                pred_box = torch.stack([x1, y1, x2, y2], dim=1)
                iou = self.bbox_iou(pred_box, assigned_gt).detach()
                total_box += (1 - iou).sum()

                cls_target = torch.zeros_like(cls_sel)
                cls_target[torch.arange(cls_sel.size(0)), assigned_cls] = iou
                total_cls += self.focal_bce(cls_sel, cls_target).sum()

                neg_mask = torch.ones((h * w,), device=device, dtype=torch.bool)
                neg_mask[point_idx] = False
                if neg_mask.any():
                    cls_neg = cls_flat[neg_mask]
                    total_cls += self.focal_bce(cls_neg, torch.zeros_like(cls_neg)).sum()

        loss = self.l_box * total_box + self.l_cls * total_cls + self.l_dfl * total_dfl
        return loss, {"l_box": total_box.item(), "l_cls": total_cls.item(), "l_dfl": total_dfl.item()}

class DualYoloV8L(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.rgb_backbone = BackboneV8Large(3)
        self.ir_backbone  = BackboneV8Large(3)

        # 注意: d_model 可设 96/128；'scalar' 更省显存，'channel' 表达力更强
        self.neck = DualFusionNeckXAttnL(
            C3c=256, C4c=512, C5c=1024,
            gate_mode='channel',
            d_model=96,
            reduction=16
        )

        # anchor-free 检测头
        self.detect = DetectHead(num_classes, ch=[256, 512, 1024], reg_max=16)
        self.strides = [8, 16, 32]

    def forward(self, rgb, ir):
        r3, r4, r5 = self.rgb_backbone(rgb)
        i3, i4, i5 = self.ir_backbone(ir)
        F3, F4, F5 = self.neck((r3, r4, r5), (i3, i4, i5))
        outs = self.detect([F3, F4, F5])
        return outs

def decode_predictions(preds, strides, conf_thr=0.25, img_size=640, reg_max=16):
    """
    Anchor-free 解码：preds: List[(reg_out, cls_out)] per level
    返回 [B] -> (N,6) [x1,y1,x2,y2,conf,cls]
    """
    device = preds[0][0].device
    bs = preds[0][0].shape[0]
    out_per_im = [[] for _ in range(bs)]

    for l, (reg_out, cls_out) in enumerate(preds):
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
        reg_out = reg_out.permute(0, 2, 3, 1).reshape(B, -1, 4, reg_max + 1)
        reg_prob = reg_out.softmax(-1)
        proj = reg_prob * torch.arange(reg_max + 1, device=device)
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

def nms_axis_aligned(dets: torch.Tensor, iou_thr=0.5, topk=100):
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

def compute_pr_map(preds_all, gts_all, iou_thr=0.5, num_classes=len(CANONICAL_CLASSES), max_det=100):
    """
    preds_all: list[Tensor(M,6)] per image, [x1,y1,x2,y2,conf,cls]
    gts_all:   list[dict{'boxes':(N,4), 'labels':(N,)}] per image

    Returns:
        P:   macro-averaged precision over classes (float)
        R:   macro-averaged recall over classes   (float)
        mAP: mean AP at the given IoU threshold (VOC-11pt)  (float)
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

        # VOC-2007 11点插值 AP
        ap = 0.0
        for t in np.linspace(0,1,11):
            mask = rec >= t
            ap += np.max(prec[mask]) if mask.any() else 0.0
        ap /= 11.0
        APs.append(float(ap))
        precs.append(float(prec[-1]))
        recs.append(float(rec[-1]))

    mAP = float(np.mean(APs)) if APs else 0.0
    P = float(np.mean(precs)) if precs else 0.0
    R = float(np.mean(recs)) if recs else 0.0
    return P, R, mAP, APs

def train(args):
    os.makedirs(args.logdir, exist_ok=True)
    logger = setup_logger(os.path.join(args.logdir, "training.log"))

    # ----- GPU selection -----
    if args.gpu < 0 or not torch.cuda.is_available():
        device = torch.device("cpu")
        logger.info("Using CPU (either --gpu < 0 or CUDA not available).")
    else:
        if args.gpu >= torch.cuda.device_count():
            raise RuntimeError(f"--gpu={args.gpu} 超出可用 GPU 数量（共 {torch.cuda.device_count()} 张）。")
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
        logger.info(f"Using CUDA device cuda:{args.gpu} — {torch.cuda.get_device_name(device)}")

    # Seeds
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    # Data
    train_ds = RGBIRDetDataset(args.train_csv, imgsz=args.imgsz, augment=True)
    val_ds   = RGBIRDetDataset(args.val_csv,   imgsz=args.imgsz, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers,
                              collate_fn=collate_fn, pin_memory=True if device.type=='cuda' else False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=args.num_workers,
                              collate_fn=collate_fn, pin_memory=True if device.type=='cuda' else False)

    # Model
    model = DualYoloV8L(num_classes=len(CANONICAL_CLASSES)).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    criterion = YoloLoss(len(CANONICAL_CLASSES), reg_max=16)

    best_map = 0.0
    for epoch in range(1, args.epochs+1):
        model.train()
        t0 = time.time()
        loss_meter = 0.0; meter = {"l_box":0.0,"l_cls":0.0,"l_dfl":0.0}
        nb = 0
        for rgbs, irs, targets, names in train_loader:
            rgbs = rgbs.to(device, non_blocking=True if device.type=='cuda' else False)
            irs  = irs.to(device,  non_blocking=True if device.type=='cuda' else False)
            tgts = [{k:v.to(device) if torch.is_tensor(v) else v for k,v in t.items()} for t in targets]

            preds = model(rgbs, irs)
            loss, parts = criterion(preds, tgts, model.strides)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optim.step()

            loss_meter += loss.item(); nb += 1
            for k in meter: meter[k]+= parts[k]

        scheduler.step()
        dt = time.time()-t0
        loss_meter /= max(nb,1)
        for k in meter: meter[k] /= max(nb,1)
        logger.info(f"Epoch {epoch}/{args.epochs} | loss={loss_meter:.4f} (box {meter['l_box']:.2f} cls {meter['l_cls']:.2f} dfl {meter['l_dfl']:.2f}) | {dt:.1f}s")

        # ---- Evaluation (从第20个epoch开始) ----
        if epoch >= 20:
            model.eval()
            preds_all = []; gts_all = []
            with torch.no_grad():
                for rgbs, irs, targets, names in val_loader:
                    rgbs = rgbs.to(device); irs = irs.to(device)
                    outs = model(rgbs, irs)
                    dets = decode_predictions(outs, model.strides, conf_thr=args.conf_thres, img_size=args.imgsz, reg_max=16)
                    dets = [nms_axis_aligned(d, iou_thr=args.nms_iou) for d in dets]
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
                torch.save({
                    "model": model.state_dict(),
                    "classes": CANONICAL_CLASSES,
                    "epoch": epoch,
                    "metrics": {
                        "P": P, "R": R, "mAP50": mAP50,
                        "AP50_per_class": ap_table
                    }
                }, save_path)
                logger.info(f"Saved best to {save_path} (mAP50={mAP50:.4f})")
        else:
            logger.info(f"Epoch {epoch}/{args.epochs} | Skipping evaluation (will start from epoch 20)")

    final_path = os.path.join(args.logdir, "last.pt")
    torch.save({"model": model.state_dict(), "classes": CANONICAL_CLASSES}, final_path)
    logger.info(f"Training finished. Final weights saved to {final_path}")

def get_args():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, default="../LLVIP/llvip_train_paths.csv")
    ap.add_argument("--val_csv",   type=str, default="../LLVIP/llvip_test_paths.csv")
    ap.add_argument("--imgsz",     type=int, default=640)
    ap.add_argument("--epochs",    type=int, default=100)
    ap.add_argument("--batch",     type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr",        type=float, default=1e-3)
    ap.add_argument("--conf_thres",type=float, default=0.001)
    ap.add_argument("--nms_iou",   type=float, default=0.5)
    ap.add_argument("--logdir",    type=str, default="./runs/1226_free2")
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--gpu",       type=int, default=0, help="使用第几张 GPU（0 开始）。设为 -1 强制使用 CPU。")
    ap.add_argument("--max_det",       type=int, default=100)
    return ap.parse_args()

if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.logdir, exist_ok=True)
    train(args)
