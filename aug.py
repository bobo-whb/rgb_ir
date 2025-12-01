import math
import random
from typing import Tuple

import numpy as np
from PIL import Image, ImageOps, ImageEnhance

def _wrap_angle_rad(a: np.ndarray) -> np.ndarray:
    """
    将角度归一化到 [-pi, pi)
    """
    return ((a + math.pi) % (2 * math.pi)) - math.pi


def _rotated_bbox_half_extents(boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算旋转矩形在 x/y 方向上的半轴长度，用于判断是否完全落在裁剪区域内。
    """
    if boxes.shape[0] == 0:
        zero = np.zeros((0,), dtype=np.float32)
        return zero, zero
    half_w = boxes[:, 2] * 0.5
    half_h = boxes[:, 3] * 0.5
    ang = boxes[:, 4]
    cos_a = np.abs(np.cos(ang))
    sin_a = np.abs(np.sin(ang))
    half_extent_x = np.abs(half_w * cos_a) + np.abs(half_h * sin_a)
    half_extent_y = np.abs(half_w * sin_a) + np.abs(half_h * cos_a)
    return half_extent_x, half_extent_y


def random_color_jitter_rgb(
    rgb: Image.Image,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    prob: float = 0.8,
) -> Image.Image:
    """
    只对 RGB 图像做颜色增强；IR 不动
    brightness / contrast / saturation 因子范围：
        factor in [1 - x, 1 + x]
    """
    if random.random() < prob:
        # Brightness
        if brightness > 0:
            factor = 1.0 + random.uniform(-brightness, brightness)
            rgb = ImageEnhance.Brightness(rgb).enhance(factor)
        # Contrast
        if contrast > 0:
            factor = 1.0 + random.uniform(-contrast, contrast)
            rgb = ImageEnhance.Contrast(rgb).enhance(factor)
        # Saturation
        if saturation > 0:
            factor = 1.0 + random.uniform(-saturation, saturation)
            rgb = ImageEnhance.Color(rgb).enhance(factor)
    return rgb


def random_scale_pair(
    rgb: Image.Image,
    ir: Image.Image,
    boxes: np.ndarray,
    scale_range: Tuple[float, float] = (0.8, 1.2),
) -> Tuple[Image.Image, Image.Image, np.ndarray]:
    """
    等比例缩放 RGB/IR + 旋转框 (cx,cy,w,h,angle)
    boxes: (N,5) or (0,5)
    """
    s = random.uniform(scale_range[0], scale_range[1])
    if abs(s - 1.0) < 1e-3:
        return rgb, ir, boxes

    w, h = rgb.size
    new_w = max(1, int(w * s))
    new_h = max(1, int(h * s))

    rgb = rgb.resize((new_w, new_h), Image.BILINEAR)
    ir = ir.resize((new_w, new_h), Image.BILINEAR)

    if boxes.shape[0] > 0:
        boxes = boxes.copy()
        boxes[:, 0] *= s  # cx
        boxes[:, 1] *= s  # cy
        boxes[:, 2] *= s  # w
        boxes[:, 3] *= s  # h
    return rgb, ir, boxes


def random_rotate_pair(
    rgb: Image.Image,
    ir: Image.Image,
    boxes: np.ndarray,
    labels: np.ndarray,
    max_deg: float = 10.0,
    prob: float = 0.5,
) -> Tuple[Image.Image, Image.Image, np.ndarray, np.ndarray]:
    """
    以图像中心为原点小角度旋转（保持尺寸不变 expand=False）
    旋转后：
      - 图像按角度旋转
      - 旋转框中心 (cx,cy) 应用同样的旋转
      - 框自身角度 angle += theta_rad
      - 若中心离开图像范围则连同标签一起丢弃
    """
    if random.random() >= prob:
        return rgb, ir, boxes, labels

    angle_deg = random.uniform(-max_deg, max_deg)
    if abs(angle_deg) < 1e-3:
        return rgb, ir, boxes, labels

    w, h = rgb.size
    # 旋转图像（使用灰色填充）
    rgb = rgb.rotate(angle_deg, resample=Image.BILINEAR, expand=False, fillcolor=(114, 114, 114))
    ir = ir.rotate(angle_deg, resample=Image.BILINEAR, expand=False, fillcolor=(114, 114, 114))

    if boxes.shape[0] == 0:
        return rgb, ir, boxes, labels

    boxes = boxes.copy()
    labels = labels.copy()
    theta = math.radians(angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    cx0, cy0 = w / 2.0, h / 2.0

    x = boxes[:, 0] - cx0
    y = boxes[:, 1] - cy0

    # PIL 的 rotate 以屏幕坐标系 (x 右 / y 下) 逆时针旋转；
    # 因此中心坐标也要使用相同的变换矩阵。
    x_new = x * cos_t + y * sin_t + cx0
    y_new = -x * sin_t + y * cos_t + cy0

    boxes[:, 0] = x_new
    boxes[:, 1] = y_new
    boxes[:, 4] = _wrap_angle_rad(boxes[:, 4] + theta)

    # 丢弃中心跑到图像外的框
    keep = (
        (boxes[:, 0] > 0.0)
        & (boxes[:, 0] < w)
        & (boxes[:, 1] > 0.0)
        & (boxes[:, 1] < h)
    )
    boxes = boxes[keep]
    labels = labels[keep]

    return rgb, ir, boxes, labels


def random_crop_pair(
    rgb: Image.Image,
    ir: Image.Image,
    boxes: np.ndarray,
    labels: np.ndarray,
    min_scale: float = 0.7,
    prob: float = 0.5,
) -> Tuple[Image.Image, Image.Image, np.ndarray, np.ndarray]:
    """
    随机裁剪：
      - 随机选择一个宽高都在 [min_scale, 1.0] * 原尺寸 范围的裁剪框
      - 裁剪 RGB/IR
      - 框中心减去裁剪起点 (x0,y0)
      - 只保留中心和整框都仍在裁剪图像内的目标
    """
    if random.random() >= prob:
        return rgb, ir, boxes, labels

    w, h = rgb.size
    if w < 32 or h < 32:
        return rgb, ir, boxes, labels

    scale = random.uniform(min_scale, 1.0)
    new_w = int(w * scale)
    new_h = int(h * scale)
    if new_w <= 0 or new_h <= 0:
        return rgb, ir, boxes, labels
    if new_w == w and new_h == h:
        return rgb, ir, boxes, labels

    if w == new_w:
        x0 = 0
    else:
        x0 = random.randint(0, w - new_w)
    if h == new_h:
        y0 = 0
    else:
        y0 = random.randint(0, h - new_h)

    # 裁剪图像
    rgb = rgb.crop((x0, y0, x0 + new_w, y0 + new_h))
    ir = ir.crop((x0, y0, x0 + new_w, y0 + new_h))

    if boxes.shape[0] == 0:
        return rgb, ir, boxes, labels

    boxes = boxes.copy()
    labels = labels.copy()

    boxes[:, 0] -= x0
    boxes[:, 1] -= y0

    cx = boxes[:, 0]
    cy = boxes[:, 1]

    keep = (cx > 0) & (cx < new_w) & (cy > 0) & (cy < new_h)
    if keep.any():
        half_x, half_y = _rotated_bbox_half_extents(boxes)
        keep = keep & (cx - half_x >= 0) & (cx + half_x <= new_w) \
                    & (cy - half_y >= 0) & (cy + half_y <= new_h)
    boxes = boxes[keep]
    labels = labels[keep]

    return rgb, ir, boxes, labels


def random_flip_pair(
    rgb: Image.Image,
    ir: Image.Image,
    boxes: np.ndarray,
    hflip_prob: float = 0.5,
    vflip_prob: float = 0.5,
) -> Tuple[Image.Image, Image.Image, np.ndarray]:
    """
    随机水平/垂直翻转：
      水平翻转：
        x' = W - x
        angle' = pi - angle
      垂直翻转：
        y' = H - y
        angle' = -angle
    若两者都发生，相当于旋转 180°（angle += pi）。
    """
    w, h = rgb.size

    do_h = random.random() < hflip_prob
    do_v = random.random() < vflip_prob

    if do_h:
        rgb = ImageOps.mirror(rgb)
        ir = ImageOps.mirror(ir)
    if do_v:
        rgb = ImageOps.flip(rgb)
        ir = ImageOps.flip(ir)

    if boxes.shape[0] == 0:
        return rgb, ir, boxes

    boxes = boxes.copy()

    # 需要注意：水平/垂直可能一起发生，所以要用当前的 w,h
    if do_h:
        boxes[:, 0] = w - boxes[:, 0]
        boxes[:, 4] = math.pi - boxes[:, 4]

    if do_v:
        boxes[:, 1] = h - boxes[:, 1]
        boxes[:, 4] = -boxes[:, 4]

    boxes[:, 4] = _wrap_angle_rad(boxes[:, 4])

    return rgb, ir, boxes


def augment_pair(
    rgb: Image.Image,
    ir: Image.Image,
    boxes: np.ndarray,
    labels: np.ndarray,
) -> Tuple[Image.Image, Image.Image, np.ndarray, np.ndarray]:
    """
    综合增强管线：
      1. 颜色增强（仅 RGB）
      2. 随机缩放
      3. 小角度随机旋转
      4. 随机裁剪
      5. 随机翻转
    """
    # 1) 颜色增强（RGB）
    rgb = random_color_jitter_rgb(rgb)

    # 2) 随机缩放
    rgb, ir, boxes = random_scale_pair(rgb, ir, boxes, scale_range=(0.8, 1.2))

    # 3) 轻微随机旋转
    rgb, ir, boxes, labels = random_rotate_pair(rgb, ir, boxes, labels, max_deg=10.0, prob=0.5)

    # 4) 随机裁剪
    rgb, ir, boxes, labels = random_crop_pair(rgb, ir, boxes, labels, min_scale=0.7, prob=0.5)

    # 5) 随机翻转
    rgb, ir, boxes = random_flip_pair(rgb, ir, boxes, hflip_prob=0.5, vflip_prob=0.5)

    # 保证 dtype 正确
    if boxes.shape[0] > 0:
        boxes = boxes.astype(np.float32)
    else:
        boxes = np.zeros((0, 5), dtype=np.float32)

    if labels.shape[0] > 0:
        labels = labels.astype(np.int64)
    else:
        labels = np.zeros((0,), dtype=np.int64)

    return rgb, ir, boxes, labels
