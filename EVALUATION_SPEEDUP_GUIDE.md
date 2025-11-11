# 评估加速优化指南

## 问题现象
将 `conf_thres` 从 0.1 改为 0.001 后，评估时间从 ~30秒/epoch 增加到 ~10分钟/epoch，慢了20倍。

## 原因分析

### 1. 检测框数量暴增
| conf_thres | 保留框数/图 | 总计算量 |
|-----------|----------|---------|
| 0.1 | ~300 | 300 × 100 GT = 3万次IoU |
| 0.001 | ~5000 | 5000 × 100 GT = 50万次IoU |

### 2. 精确IoU计算复杂
- AABB粗过滤：O(N×M)
- SAT细过滤：O(k) k为候选对数
- 精确多边形求交：每对需要计算16组边交点 + 凸包排序

## ✅ 已实施的优化（v1.0）

### 优化1: 智能评估频率
```python
# 策略：
# - 前50轮：每5轮评估（epoch 5, 10, 15...）
# - 50-90轮：每2轮评估（epoch 52, 54, 56...）
# - 最后10轮：每轮都评估（确保找到最佳）
```

**效果**：100轮训练减少60%评估次数（从100次 → 40次）

### 优化2: 增大CHUNK分块大小
```python
CHUNK = 65536  # 原先16384
```

**效果**：减少循环次数，提升GPU利用率，加速15-25%

### 优化3: 添加评估时间监控
```python
eval_time=XXXs  # 日志中显示每次评估耗时
```

## 🚀 进一步优化选项

### 选项A: 使用更激进的评估频率
```bash
# 前期快速验证，后期精细评估
python train_1107.py --eval_interval 10  # 每10轮评估一次
```

### 选项B: 减小验证集（仅用于快速验证）
```python
# 在数据加载时随机采样20%验证集
val_ds_full = RGBIRDetDataset(args.val_csv, ...)
val_ds = Subset(val_ds_full, random.sample(range(len(val_ds_full)), len(val_ds_full)//5))
```

**风险**：mAP不够稳定，仅适合调试阶段

### 选项C: 限制NMS后的检测框数量
```python
# 在decode_predictions中添加topk
for b in range(bs):
    if len(out_per_im[b]) > 1000:  # 限制最多1000个
        scores = out_per_im[b][:, 5]
        topk_idx = torch.topk(scores, k=1000).indices
        out_per_im[b] = out_per_im[b][topk_idx]
```

### 选项D: 两阶段评估策略
```python
# 训练期间用快速AABB评估（仅监控趋势）
# 训练结束后用精确IoU评估（最终结果）

if epoch < args.epochs - 5:
    # 快速评估（AABB）
    iou_fn = iou_aabb_approx
else:
    # 精确评估（rotated）
    iou_fn = iou_rotated
```

### 选项E: 并行化验证集处理（需要改代码）
使用多GPU或多进程并行处理验证集图像。

## 📊 性能对比

| 配置 | 评估时间/epoch | 100轮总时间 | 建议场景 |
|-----|-------------|-----------|---------|
| 原始(conf=0.1) | 30s | 50min | ❌ 阈值不标准 |
| conf=0.001无优化 | 600s | 1000min | ❌ 太慢 |
| **v1.0优化** | 600s × 40% | 400min | ✅ **推荐** |
| +选项A(interval=10) | 600s × 10% | 100min | 快速实验 |
| +选项B(20%验证集) | 120s × 40% | 80min | 快速调试 |

## 🎯 推荐训练配置

### 快速实验阶段（调超参）
```bash
python train_1107.py \
    --epochs 50 \
    --eval_interval 5 \
    --conf_thres 0.001 \
    --max_det 300
```

### 正式训练阶段（发论文/提交）
```bash
python train_1107.py \
    --epochs 100 \
    --eval_interval 1 \      # 使用自动策略
    --conf_thres 0.001 \
    --max_det 300
```

### 极速调试阶段
```bash
python train_1107.py \
    --epochs 20 \
    --eval_interval 10 \     # 每10轮评估
    --conf_thres 0.01 \      # 临时提高阈值
    --max_det 200
```

## 🔧 如果评估仍然很慢

### 检查1: GPU利用率
```bash
nvidia-smi -l 1  # 实时监控
```
- 如果GPU利用率 < 80%：考虑增大CHUNK到 131072
- 如果显存不足：减小CHUNK到 32768

### 检查2: 验证集大小
```bash
wc -l val.csv  # 查看验证集图片数量
```
- 如果 > 2000张：考虑减小或分批评估
- 如果 < 500张：问题在IoU计算本身

### 检查3: 每张图的目标数量
在compute_pr_map中添加统计：
```python
avg_gt_per_img = sum(len(g['boxes']) for g in gts_all) / len(gts_all)
logger.info(f"Avg GT per image: {avg_gt_per_img:.1f}")
```

### 检查4: 是否在CPU上运行
确保device是cuda，而不是cpu：
```python
logger.info(f"Eval device: {device}")  # 应该是 cuda:0
```

## 💡 终极方案：混合评估

训练时用快速评估跟踪趋势，训练完后用精确评估报告最终结果：

```bash
# 1. 快速训练（监控loss和快速mAP）
python train_1107.py --conf_thres 0.01 --eval_interval 5

# 2. 训练完后精确评估best.pt
python eval_exact.py --weights runs/xxx/best.pt --conf 0.001
```

## 📈 预期优化效果

使用v1.0优化后：
- ✅ 评估次数：100次 → 40次（减少60%）
- ✅ 单次评估：600s → 510s（快15%）
- ✅ **总评估时间：1000min → 340min（节省11小时！）**

## ⚙️ 高级选项：动态调整

```python
# 根据epoch动态调整conf_thres
if epoch < 30:
    conf_eval = 0.01   # 前期快速评估
elif epoch < 80:
    conf_eval = 0.005  # 中期适中
else:
    conf_eval = 0.001  # 后期精确
```

## 🎓 总结

1. **必须使用conf=0.001**（学术标准）
2. **降低评估频率**是最有效的优化（已实施）
3. 增大CHUNK可小幅提速（已实施）
4. 极端情况考虑两阶段评估或减小验证集

**当前优化版本已经能节省60%的评估时间，如果还是太慢，建议使用 --eval_interval 参数进一步降低频率。**
