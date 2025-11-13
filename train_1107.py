import os
import sys
import math
import time
import json
import random
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET

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
    "truvk": "truck",
    "feright": "freight_car",
    "feright car": "freight_car",
    "feright_car": "freight_car",
    "freight": "freight_car",
    "freight car": "freight_car",
    "freight_car": "freight_car",
    "*": None,
    "": None,
    "unknown": None,
}

CANONICAL_CLASSES: List[str] = [
    "car", "truck", "bus", "van", "freight_car"
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

def iou_rotated(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    精确旋转矩形 IoU（AABB 过滤 + SAT 判定 + 精确交集多边形）
    输入:
        boxes1: (N,5) [cx, cy, w, h, angle(rad)]
        boxes2: (M,5) [cx, cy, w, h, angle(rad)]
    返回:
        (N,M) IoU 张量
    说明:
        - 完全 torch 实现，可跑在 CPU/GPU；
        - AABB 粗过滤 + SAT 细过滤，大幅减少精确求交次数；
        - 精确部分：两矩形交集多边形由“包含角点 + 16 组边-边交点”构成，再用凸包顺序与鞋带公式求面积；
        - 角度单位: 弧度。
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    device, dtype = boxes1.device, boxes1.dtype
    N, M = boxes1.shape[0], boxes2.shape[0]
    EPS = torch.tensor(1e-9, device=device, dtype=dtype)
    ROUND = 1e-4  # 去重用的量化步长（防数值噪声导致重复点）

    # ---------- 基础工具 ----------
    def _rbox_corners(bx: torch.Tensor) -> torch.Tensor:
        # bx: (B,5) -> (B,4,2) CCW: (+x,+y)->(-x,+y)->(-x,-y)->(+x,-y)
        cx, cy, w, h, a = bx.unbind(-1)
        hw, hh = w * 0.5, h * 0.5
        cosa, sina = torch.cos(a), torch.sin(a)
        # 局部四角
        xs = torch.stack([ hw, -hw, -hw,  hw], dim=-1)  # (B,4)
        ys = torch.stack([ hh,  hh, -hh, -hh], dim=-1)  # (B,4)
        xr = xs * cosa.unsqueeze(-1) - ys * sina.unsqueeze(-1)
        yr = xs * sina.unsqueeze(-1) + ys * cosa.unsqueeze(-1)
        x = xr + cx.unsqueeze(-1)
        y = yr + cy.unsqueeze(-1)
        return torch.stack([x, y], dim=-1)  # (B,4,2)

    def _aabb_from_corners(P: torch.Tensor) -> torch.Tensor:
        # P: (B,4,2) -> (B,4) [x1,y1,x2,y2]
        x_min = P[..., 0].min(dim=1).values
        y_min = P[..., 1].min(dim=1).values
        x_max = P[..., 0].max(dim=1).values
        y_max = P[..., 1].max(dim=1).values
        return torch.stack([x_min, y_min, x_max, y_max], dim=-1)

    def _aabb_overlap_mask(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # A,B: (B,4) -> bool (N,M)
        Ax1, Ay1, Ax2, Ay2 = A.unbind(-1)
        Bx1, By1, Bx2, By2 = B.unbind(-1)
        # 交集宽高 > 0 判定
        inter_w = torch.clamp(torch.minimum(Ax2[:, None], Bx2[None, :]) -
                              torch.maximum(Ax1[:, None], Bx1[None, :]), min=0)
        inter_h = torch.clamp(torch.minimum(Ay2[:, None], By2[None, :]) -
                              torch.maximum(Ay1[:, None], By1[None, :]), min=0)
        return (inter_w > 0) & (inter_h > 0)

    def _edges_from_corners(P: torch.Tensor):
        # P: (K,4,2) -> e0=(K,4,2), e1=(K,4,2) 四条边起点与终点
        return P, torch.roll(P, shifts=-1, dims=1)

    def _cross(u: torch.Tensor, v: torch.Tensor):
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

    def _points_in_convex_poly(pts: torch.Tensor, poly: torch.Tensor) -> torch.Tensor:
        # pts: (K,4,2)  poly: (K,4,2)  -> (K,4) bool
        b0, b1 = _edges_from_corners(poly)        # (K,4,2)
        e = b1 - b0                               # (K,4,2)
        # 对每个点，计算它相对每条边的“在左侧”判定（CCW 多边形，左侧为内侧）
        # 广播: (K,4_pts,1,2) 与 (K,1,4_edges,2) -> (K,4,4)
        v = pts.unsqueeze(2) - b0.unsqueeze(1)    # (K,4,4,2)
        c = _cross(e.unsqueeze(1), v)             # (K,4,4)
        inside_all_edges = (c >= -1e-9).all(dim=-1)  # (K,4)
        return inside_all_edges

    def _segment_intersections(a0, a1, b0, b1):
        """
        a0,a1: (K,4,1,2)  b0,b1: (K,1,4,2)
        返回:
            inter: (K,4,4,2)
            valid: (K,4,4) bool
        """
        r = a1 - a0   # (K,4,1,2)
        s = b1 - b0   # (K,1,4,2)
        denom = _cross(r, s)  # (K,4,4)
        d = b0 - a0           # (K,4,4,2)
        t = _cross(d, s) / (denom + EPS)
        u = _cross(d, r) / (denom + EPS)
        inter = a0 + t.unsqueeze(-1) * r  # (K,4,4,2)
        valid = (denom.abs() > 1e-9) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
        return inter, valid

    def _polygon_area(pts: torch.Tensor) -> torch.Tensor:
        # pts: (P,2) 已按凸包角度排序
        x, y = pts[:, 0], pts[:, 1]
        x2, y2 = torch.roll(x, -1), torch.roll(y, -1)
        return 0.5 * torch.abs(torch.sum(x * y2 - y * x2))

    # ---------- 预计算 ----------
    P1 = _rbox_corners(boxes1)  # (N,4,2)
    P2 = _rbox_corners(boxes2)  # (M,4,2)
    A1 = _aabb_from_corners(P1) # (N,4)
    A2 = _aabb_from_corners(P2) # (M,4)
    area1 = (boxes1[:, 2] * boxes1[:, 3]).clamp(min=0)
    area2 = (boxes2[:, 2] * boxes2[:, 3]).clamp(min=0)

    # ---------- AABB 粗过滤 ----------
    mask_aabb = _aabb_overlap_mask(A1, A2)  # (N,M)
    cand = mask_aabb.nonzero(as_tuple=False)  # (K,2)
    if cand.numel() == 0:
        return boxes1.new_zeros((N, M))

    # ---------- SAT 进一步过滤（矩形-矩形常数时间判定） ----------
    def _sat_overlap(b1: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
        # b1,b2: (K,5)
        cx1, cy1, w1, h1, a1 = b1.unbind(-1)
        cx2, cy2, w2, h2, a2 = b2.unbind(-1)
        hw1, hh1 = w1 * 0.5, h1 * 0.5
        hw2, hh2 = w2 * 0.5, h2 * 0.5
        c1, s1 = torch.cos(a1), torch.sin(a1)
        c2, s2 = torch.cos(a2), torch.sin(a2)
        # A 的轴 (u1, v1), B 的轴 (u2, v2)
        # u = (cos, sin), v = (-sin, cos)
        u1 = torch.stack([ c1, s1], dim=-1)  # (K,2)
        v1 = torch.stack([-s1, c1], dim=-1)
        u2 = torch.stack([ c2, s2], dim=-1)
        v2 = torch.stack([-s2, c2], dim=-1)
        # t = c2 - c1
        t = torch.stack([cx2 - cx1, cy2 - cy1], dim=-1)  # (K,2)
        # R 矩阵 = A 轴到 B 轴的投影
        R00 = (u1 * u2).sum(-1).abs(); R01 = (u1 * v2).sum(-1).abs()
        R10 = (v1 * u2).sum(-1).abs(); R11 = (v1 * v2).sum(-1).abs()
        # t 在 A 与 B 轴上的投影
        tA0 = (t * u1).sum(-1).abs()
        tA1 = (t * v1).sum(-1).abs()
        tB0 = (t * u2).sum(-1).abs()
        tB1 = (t * v2).sum(-1).abs()
        # 四个轴的分离判据
        cond1 = tA0 <= (hw1 + hw2 * R00 + hh2 * R01)
        cond2 = tA1 <= (hh1 + hw2 * R10 + hh2 * R11)
        cond3 = tB0 <= (hw2 + hw1 * R00 + hh1 * R10)
        cond4 = tB1 <= (hh2 + hw1 * R01 + hh1 * R11)
        return cond1 & cond2 & cond3 & cond4  # (K,)

    # 分块处理，避免一次性占用过大显存
    CHUNK = 65536
    ious = boxes1.new_zeros((N, M))
    for st in range(0, cand.shape[0], CHUNK):
        ed = min(st + CHUNK, cand.shape[0])
        idx = cand[st:ed]                  # (k,2)
        i_idx, j_idx = idx[:, 0], idx[:, 1]
        b1 = boxes1[i_idx]
        b2 = boxes2[j_idx]

        # SAT 过滤
        sat_ok = _sat_overlap(b1, b2)      # (k,)
        if sat_ok.sum() == 0:
            continue
        i_idx = i_idx[sat_ok]
        j_idx = j_idx[sat_ok]
        k = i_idx.numel()

        # 取对应顶点
        pa = P1[i_idx]     # (k,4,2)
        pb = P2[j_idx]     # (k,4,2)

        # A 的角在 B 内 / B 的角在 A 内
        mask_a_in_b = _points_in_convex_poly(pa, pb)  # (k,4)
        mask_b_in_a = _points_in_convex_poly(pb, pa)  # (k,4)

        # 边-边交点（16 组）
        a0, a1 = _edges_from_corners(pa)  # (k,4,2)
        b0, b1_ = _edges_from_corners(pb)
        # 为广播 (k,4,1,2) & (k,1,4,2)
        a0_ex = a0.unsqueeze(2); a1_ex = a1.unsqueeze(2)
        b0_ex = b0.unsqueeze(1); b1_ex = b1_.unsqueeze(1)
        inter_all, valid_all = _segment_intersections(a0_ex, a1_ex, b0_ex, b1_ex)  # (k,4,4,2), (k,4,4)

        # 逐对拼接候选点 -> 凸包排序 -> 面积
        for t_i in range(k):
            pts_list = []

            # A 的内点
            if mask_a_in_b[t_i].any():
                pts_list.append(pa[t_i][mask_a_in_b[t_i]])

            # B 的内点
            if mask_b_in_a[t_i].any():
                pts_list.append(pb[t_i][mask_b_in_a[t_i]])

            # 线段交点
            if valid_all[t_i].any():
                Pints = inter_all[t_i][valid_all[t_i]]  # (m,2)
                pts_list.append(Pints)

            if len(pts_list) == 0:
                inter = torch.tensor(0.0, device=device, dtype=dtype)
            else:
                pts = torch.cat(pts_list, dim=0)  # (T,2)
                # 去重（量化到 ROUND 网格），防止重复点/平行边导致的抖动
                pts_q = torch.round(pts / ROUND) * ROUND
                if pts_q.shape[0] > 1:
                    pts_q = torch.unique(pts_q, dim=0)
                if pts_q.shape[0] < 3:
                    inter = torch.tensor(0.0, device=device, dtype=dtype)
                else:
                    # 以质心排序（凸多边形），然后鞋带求面积
                    c = pts_q.mean(dim=0, keepdim=True)     # (1,2)
                    ang = torch.atan2(pts_q[:, 1] - c[0, 1], pts_q[:, 0] - c[0, 0])
                    order = torch.argsort(ang)
                    inter = _polygon_area(pts_q[order])

            a1_ = area1[i_idx[t_i]]
            a2_ = area2[j_idx[t_i]]
            union = torch.clamp(a1_ + a2_ - inter, min=1e-6)
            ious[i_idx[t_i], j_idx[t_i]] = torch.clamp(inter / union, 0.0, 1.0)

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
def read_csv(path: str) -> List[Dict[str,str]]:
    rows = []
    if HAS_PANDAS:
        df = pd.read_csv(path)
        cols = list(df.columns)
        if cols and cols[0].startswith("\ufeff"):
            df = df.rename(columns={cols[0]: cols[0].replace("\ufeff","")})
        rename = {}
        for c in df.columns:
            lc = c.strip().lower()
            if lc in ["rgb_img","ir_img","rgb_label","ir_label"]:
                rename[c] = lc
        df = df.rename(columns=rename)
        if "rgb_img" not in df.columns or "ir_img" not in df.columns or \
           "rgb_label" not in df.columns or "ir_label" not in df.columns:
            raise ValueError(f"CSV 缺少列：需要['rgb_img','ir_img','rgb_label','ir_label']，实际列：{list(df.columns)}")
        for _, r in df.iterrows():
            rows.append({k: str(r[k]) for k in ["rgb_img","ir_img","rgb_label","ir_label"]})
    else:
        # ---- 修复 ②：csv 回退先读 header，并正规化映射 ----
        import csv
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
            norm = [h.strip() for h in header]
            lower_map = {h.lower(): h for h in norm}
            need = ["rgb_img","ir_img","rgb_label","ir_label"]
            if not all(k in lower_map for k in need):
                raise ValueError(f"CSV 缺少列：需要{need}，实际列：{header}")
            for row in reader:
                rows.append({k: str(row[lower_map[k]]) for k in need})
    return rows

def parse_ir_xml(xml_path: str) -> Tuple[List[List[float]], List[int]]:
    rboxes = []
    clses = []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for obj in root.findall("object"):
        name = obj.findtext("name", default="").strip()
        name = clean_class_name(name)
        if name is None: continue
        if name not in CLS2ID: continue
        poly = obj.find("polygon")
        pts = []
        if poly is not None:
            for i in range(1,5):
                xi = poly.findtext(f"x{i}")
                yi = poly.findtext(f"y{i}")
                if xi is None or yi is None:
                    pts = []; break
                pts.extend([float(xi), float(yi)])
        else:
            rbb = obj.find("robndbox")
            if rbb is not None:
                cx = float(rbb.findtext("cx", "0"))
                cy = float(rbb.findtext("cy", "0"))
                w  = float(rbb.findtext("w", "1"))
                h  = float(rbb.findtext("h", "1"))
                a  = float(rbb.findtext("angle", "0"))
                rboxes.append([cx,cy,w,h,a]); clses.append(CLS2ID[name]); continue
        if len(pts)==8:
            cx,cy,w,h,ang = polygon_to_rbox(pts)
            rboxes.append([cx,cy,w,h,ang]); clses.append(CLS2ID[name])
    return rboxes, clses

class RGBIRDetDataset(Dataset):
    def __init__(self, csv_path: str, imgsz: int = 640, augment: bool = True):
        super().__init__()
        self.items = read_csv(csv_path)
        self.imgsz = imgsz
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def _load_image(self, path: str) -> Image.Image:
        img = Image.open(path).convert("RGB")
        return img

    def __getitem__(self, idx: int):
        it = self.items[idx]
        rgb_path = it["rgb_img"]
        ir_path  = it["ir_img"]
        ir_xml   = it["ir_label"]

        rgb = self._load_image(rgb_path)
        ir  = self._load_image(ir_path)

        rboxes, clses = parse_ir_xml(ir_xml)
        rboxes = np.array(rboxes, dtype=np.float32) if rboxes else np.zeros((0,5), np.float32)
        clses  = np.array(clses, dtype=np.int64) if clses else np.zeros((0,), np.int64)

        rgb, scale, pad = letterbox(rgb, self.imgsz)
        ir,  _,    _    = letterbox(ir, self.imgsz)
        if rboxes.shape[0] > 0:
            rboxes_scaled = rboxes.copy()
            rboxes_scaled[:,0] = rboxes[:,0] * scale + pad[0]
            rboxes_scaled[:,1] = rboxes[:,1] * scale + pad[1]
            rboxes_scaled[:,2] = rboxes[:,2] * scale
            rboxes_scaled[:,3] = rboxes[:,3] * scale
        else:
            rboxes_scaled = rboxes

        rgb_t = torch.from_numpy(np.array(rgb)).permute(2,0,1).float()/255.0
        ir_t  = torch.from_numpy(np.array(ir )).permute(2,0,1).float()/255.0

        target = {
            "boxes": torch.from_numpy(rboxes_scaled),
            "labels": torch.from_numpy(clses),
            "orig_size": torch.tensor([rgb.size[1], rgb.size[0]], dtype=torch.float32)
        }
        return rgb_t, ir_t, target, os.path.basename(rgb_path)

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
    def __init__(self, nc: int, anchors: List[List[Tuple[int,int]]], ch: List[int]):
        super().__init__()
        self.nc = nc
        self.no = 6 + nc
        self.nl = len(anchors)
        self.na = len(anchors[0])
        self.register_buffer("anchors", torch.tensor(anchors).float())
        self.m = nn.ModuleList([nn.Conv2d(c, self.no*self.na, 1) for c in ch])

    def forward(self, feats: List[torch.Tensor]):
        outs = []
        for i, x in enumerate(feats):
            bs, _, h, w = x.shape
            x = self.m[i](x)
            x = x.view(bs, self.na, self.no, h, w).permute(0,1,3,4,2).contiguous()
            outs.append(x)
        return outs

class YoloLoss(nn.Module):
    def __init__(self, anchors, num_classes: int, img_size: int = 640,
                 lambda_box=0.05, lambda_obj=1.0, lambda_cls=0.5, iou_t=0.5):
        super().__init__()
        self.anchors = anchors
        self.nc = num_classes
        self.img_size = img_size
        self.l_box = lambda_box
        self.l_obj = lambda_obj
        self.l_cls = lambda_cls
        self.iou_t = iou_t

    def forward(self, preds, targets, strides):
        device = preds[0].device
        bs = preds[0].shape[0]
        obj_loss = torch.zeros(1, device=device)
        cls_loss = torch.zeros(1, device=device)
        box_loss = torch.zeros(1, device=device)

        for b in range(bs):
            tboxes = targets[b]["boxes"].to(device)
            tcls   = targets[b]["labels"].to(device)
            if tboxes.numel() == 0:
                for l, p in enumerate(preds):
                    obj = p[b, ..., 5]
                    obj_loss = obj_loss + F.binary_cross_entropy_with_logits(obj, torch.zeros_like(obj), reduction="sum")
                continue

            for l, p in enumerate(preds):
                na = p.shape[1]; h = p.shape[2]; w = p.shape[3]; no = p.shape[-1]
                stride = strides[l]
                gxy = tboxes[:, :2] / stride
                gwh = tboxes[:, 2:4] / stride
                gang= tboxes[:, 4:5]

                anc = torch.tensor(self.anchors[l], device=device) / stride
                ratios = gwh[:,None,:] / anc[None,:,:]
                ar = torch.max(ratios, 1.0/ratios).max(dim=2).values
                best = ar.argmin(dim=1)

                gij = gxy.long()
                gi = gij[:,0].clamp(0, w-1); gj = gij[:,1].clamp(0, h-1)

                sel = p[b, best, gj, gi]
                px,py,pw,ph,pa,pobj = sel[:,0],sel[:,1],sel[:,2],sel[:,3],sel[:,4],sel[:,5]
                pcls = sel[:,6:]

                tx = gxy[:,0] - gi.float()
                ty = gxy[:,1] - gj.float()
                # ---- 修复 ④：稳定性 —— clamp log 尺寸，并角度 wrap ----
                tw_tgt = torch.log(gwh[:,0] / anc[best,0] + 1e-6).clamp(min=-4.0, max=4.0)
                th_tgt = torch.log(gwh[:,1] / anc[best,1] + 1e-6).clamp(min=-4.0, max=4.0)
                tw_pred = pw.clamp(min=-4.0, max=4.0)
                th_pred = ph.clamp(min=-4.0, max=4.0)
                ta = gang[:,0]
                ang_diff = torch.remainder(pa - ta + math.pi, 2*math.pi) - math.pi

                box_loss = box_loss + F.smooth_l1_loss(px, tx, reduction="sum")
                box_loss = box_loss + F.smooth_l1_loss(py, ty, reduction="sum")
                box_loss = box_loss + F.smooth_l1_loss(tw_pred, tw_tgt, reduction="sum")
                box_loss = box_loss + F.smooth_l1_loss(th_pred, th_tgt, reduction="sum")
                box_loss = box_loss + F.smooth_l1_loss(ang_diff, torch.zeros_like(ang_diff), reduction="sum")

                obj_loss = obj_loss + F.binary_cross_entropy_with_logits(pobj, torch.ones_like(pobj), reduction="sum")
                tgt = torch.zeros_like(pcls)
                tgt[torch.arange(tcls.numel(), device=device), tcls] = 1.0
                cls_loss = cls_loss + F.binary_cross_entropy_with_logits(pcls, tgt, reduction="sum")

                K = min(1000, (h*w*na))
                if K > 0:
                    ri = torch.randint(0, na, (K,), device=device)
                    rj = torch.randint(0, h,  (K,), device=device)
                    rk = torch.randint(0, w,  (K,), device=device)
                    neg = p[b, ri, rj, rk, 5]
                    obj_loss = obj_loss + F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg), reduction="sum")

        loss = self.l_box*box_loss + self.l_obj*obj_loss + self.l_cls*cls_loss
        return loss, {"l_box": box_loss.item(),"l_obj": obj_loss.item(),"l_cls": cls_loss.item()}

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

        # 可保持你原来的锚框；若数据分布差别较大，建议重聚类
        anchors = [
            [(10,13), (16,30), (33,23)],
            [(30,61), (62,45), (59,119)],
            [(116,90), (156,198), (373,326)]
        ]
        self.detect = DetectHead(num_classes, anchors, ch=[256, 512, 1024])
        self.strides = [8, 16, 32]
        self.anchors = anchors

    def forward(self, rgb, ir):
        r3, r4, r5 = self.rgb_backbone(rgb)
        i3, i4, i5 = self.ir_backbone(ir)
        F3, F4, F5 = self.neck((r3, r4, r5), (i3, i4, i5))
        outs = self.detect([F3, F4, F5])
        return outs

def decode_predictions(preds, strides, conf_thr=0.001, img_size=640, anchors=None):
    """
    Convert raw head outputs to absolute coords for NMS
    Return list per image of detections: [cx,cy,w,h,angle, conf, cls_id]
    anchors: optional List[List[Tuple[w,h]]] per level; if None,从模型的 detect.anchors 取
    """
    device = preds[0].device
    bs = preds[0].shape[0]
    out_per_im = [[] for _ in range(bs)]

    if anchors is None:
        anchors = [
            [(10,13), (16,30), (33,23)],     # P3
            [(30,61), (62,45), (59,119)],    # P4
            [(116,90), (156,198), (373,326)] # P5
        ]

    for l, p in enumerate(preds):
        na = int(p.size(1)); h = int(p.size(2)); w = int(p.size(3)); no = int(p.size(4))
        stride = strides[l]

        gy, gx = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij",
        )

        anc_lvl = torch.tensor(anchors[l], device=device, dtype=torch.float32)
        anc_w = anc_lvl[:, 0].view(na, 1, 1).expand(na, h, w)
        anc_h = anc_lvl[:, 1].view(na, 1, 1).expand(na, h, w)

        for b in range(bs):
            x = p[b]  # (na,h,w,no)

            obj = x[..., 5].sigmoid()
            cls = x[..., 6:].sigmoid()
            conf, cls_id = (obj[..., None] * cls).max(dim=-1)

            mask = conf > conf_thr
            if not mask.any():
                continue

            sel = x[mask]
            aidx_full = torch.arange(na, device=device).view(na, 1, 1).repeat(1, h, w)
            gi = gx.unsqueeze(0).expand(na, -1, -1)[mask]
            gj = gy.unsqueeze(0).expand(na, -1, -1)[mask]
            aidx = aidx_full[mask]

            aw = anc_w[mask]
            ah = anc_h[mask]

            # 解码（训练预测为相对偏移/对数缩放；角度为弧度）
            cx = (sel[:, 0] + gi.float()) * stride
            cy = (sel[:, 1] + gj.float()) * stride
            w_abs = aw * torch.exp(sel[:, 2])
            h_abs = ah * torch.exp(sel[:, 3])
            ang = sel[:, 4]

            c = conf[mask]
            # ---- 修复 ③：类别 id 转 long，然后为拼接统一转回 float ----
            cid_long = cls_id[mask].to(dtype=torch.long)
            cid = cid_long.to(dtype=torch.float32)

            det = torch.stack([cx, cy, w_abs, h_abs, ang, c, cid], dim=-1)
            out_per_im[b].append(det)

    for b in range(bs):
        if len(out_per_im[b]) > 0:
            out_per_im[b] = torch.cat(out_per_im[b], dim=0)
        else:
            out_per_im[b] = torch.zeros((0, 7), device=device)

    return out_per_im

def nms_rotated_simple(dets: torch.Tensor, iou_thr=0.5, topk=300):
    if dets.numel()==0: return dets
    keep = []
    for cls in dets[:,6].unique():
        D = dets[dets[:,6]==cls]
        order = torch.argsort(D[:,5], descending=True)
        D = D[order]
        while D.size(0) > 0 and len(keep) < topk:
            best = D[0:1]
            keep.append(best)
            if D.size(0) == 1: break
            ious = iou_rotated(best[:,:5], D[1:,:5]).squeeze(0)
            D = D[1:][ious < iou_thr]
    return torch.cat(keep, dim=0) if keep else dets[:0]

def compute_pr_map(preds_all, gts_all, iou_thr=0.5, num_classes=len(CANONICAL_CLASSES), max_det=300):
    """
    preds_all: list[Tensor(M,7)] per image, [cx,cy,w,h,angle,conf,cls]
    gts_all:   list[dict{'boxes':(N,5), 'labels':(N,)}] per image

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
            order = torch.argsort(dets[:,5], descending=True)[:max_det]
            dets = dets[order]

        for c in range(num_classes):
            m_det = dets[:,6] == float(c)
            det_c = dets[m_det]
            gt_mask = (gt["labels"] == c)
            gt_c = gt["boxes"][gt_mask]
            total_positives[c] += int(gt_mask.sum().item())

            if det_c.numel() == 0:
                continue

            order = torch.argsort(det_c[:,5], descending=True)
            det_c = det_c[order]
            scores = det_c[:,5].tolist()

            if gt_c.numel() == 0:
                all_scores[c].extend(scores)
                all_tp[c].extend([0]*len(scores))
                all_fp[c].extend([1]*len(scores))
            else:
                used = torch.zeros((gt_c.size(0),), dtype=torch.bool, device=gt_c.device)
                tps = []
                fps = []
                for k in range(det_c.size(0)):
                    ious = iou_rotated(det_c[k:k+1,:5], gt_c).squeeze(0)
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
    criterion = YoloLoss(model.anchors, len(CANONICAL_CLASSES), img_size=args.imgsz)

    best_map = 0.0
    for epoch in range(1, args.epochs+1):
        model.train()
        t0 = time.time()
        loss_meter = 0.0; meter = {"l_box":0.0,"l_obj":0.0,"l_cls":0.0}
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
        logger.info(f"Epoch {epoch}/{args.epochs} | loss={loss_meter:.4f} (box {meter['l_box']:.2f} obj {meter['l_obj']:.2f} cls {meter['l_cls']:.2f}) | {dt:.1f}s")

        # ---- Evaluation ----
                # ---- Evaluation ----
        model.eval()
        preds_all = []; gts_all = []
        with torch.no_grad():
            for rgbs, irs, targets, names in val_loader:
                rgbs = rgbs.to(device); irs = irs.to(device)
                outs = model(rgbs, irs)
                dets = decode_predictions(outs, model.strides, conf_thr=0.001, img_size=args.imgsz, anchors=model.anchors)
                dets = [nms_rotated_simple(d, iou_thr=args.nms_iou) for d in dets]
                preds_all.extend(dets)
                for t in targets:
                    g = { "boxes": t["boxes"].to(device), "labels": t["labels"].to(device) }
                    gts_all.append(g)

        P, R, mAP50, APs = compute_pr_map(
            preds_all, gts_all, iou_thr=0.5,
            num_classes=len(CANONICAL_CLASSES),
            max_det=args.max_det
        )

        # 将 AP50 按顺序输出：Car / Bus / Truck / Freight-car / Van
        idx_map = {c: i for i, c in enumerate(CANONICAL_CLASSES)}
        order = ["car", "bus", "truck", "freight_car", "van"]
        name_map = {"car":"Car", "bus":"Bus", "truck":"Truck", "freight_car":"Freight-car", "van":"Van"}

        ap_table2 = {}
        for cname in order:
            ap_table2[name_map[cname]] = APs[idx_map[cname]] if cname in idx_map else 0.0

        logger.info(
            "Epoch %d VAL | mAP50=%.4f | AP50: Car=%.4f Bus=%.4f Truck=%.4f Freight-car=%.4f Van=%.4f | P=%.4f R=%.4f"
            % (epoch, mAP50,
               ap_table2["Car"], ap_table2["Bus"], ap_table2["Truck"], ap_table2["Freight-car"], ap_table2["Van"],
               P, R)
        )

        if mAP50 > best_map:
            best_map = mAP50
            save_path = os.path.join(args.logdir, "best.pt")
            torch.save({
                "model": model.state_dict(),
                "classes": CANONICAL_CLASSES,
                "epoch": epoch,
                "metrics": {
                    "P": P, "R": R, "mAP50": mAP50,
                    "AP50_per_class": ap_table2
                }
            }, save_path)
            logger.info(f"Saved best to {save_path} (mAP50={mAP50:.4f})")

    final_path = os.path.join(args.logdir, "last.pt")
    torch.save({"model": model.state_dict(), "classes": CANONICAL_CLASSES}, final_path)
    logger.info(f"Training finished. Final weights saved to {final_path}")

def get_args():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, default="train.csv")
    ap.add_argument("--val_csv",   type=str, default="val.csv")
    ap.add_argument("--imgsz",     type=int, default=640)
    ap.add_argument("--epochs",    type=int, default=100)
    ap.add_argument("--batch",     type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr",        type=float, default=1e-3)
    ap.add_argument("--conf_thres",type=float, default=0.001)
    ap.add_argument("--nms_iou",   type=float, default=0.5)
    ap.add_argument("--logdir",    type=str, default="./runs/1111")
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--gpu",       type=int, default=0, help="使用第几张 GPU（0 开始）。设为 -1 强制使用 CPU。")
    ap.add_argument("--max_det",       type=int, default=300)
    return ap.parse_args()

if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.logdir, exist_ok=True)
    train(args)