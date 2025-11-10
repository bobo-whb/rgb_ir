#!/bin/bash

# 验证mAP差异的对照实验脚本

echo "==========================================="
echo "验证train_1007 vs train_1107的mAP差异"
echo "==========================================="

# 假设你已经训练好了两个模型
MODEL_1007="./runs/train_1007/best.pt"
MODEL_1107="./runs/train_1107/best.pt"

echo ""
echo "实验1: 统一conf_thr=0.1，对比IoU计算方法的影响"
echo "-------------------------------------------"
echo "预期结果: 1007的mAP仍会高10-20%（因为AABB虚高）"
echo ""
echo "TODO: 修改train_1007中的conf_thr为0.1后重新评估"
echo "建议: 在train_1007_attmoe_v8l.py第907行改为conf_thr=0.1"

echo ""
echo "实验2: 都用AABB评估，看模型本身差异"
echo "-------------------------------------------"
echo "预期结果: 如果训练一样，mAP应该接近"
echo ""
echo "TODO: 修改train_1107使用AABB IoU评估"

echo ""
echo "实验3: 都用精确IoU + conf=0.1评估"
echo "-------------------------------------------"
echo "预期结果: 这是最公平的比较"
echo ""
echo "TODO: 修改train_1007使用精确IoU评估"

echo ""
echo "实验4: 可视化检测结果"
echo "-------------------------------------------"
echo "在相同测试图上对比:"
echo "  - train_1007 (conf=0.001): 大量低质量框"
echo "  - train_1107 (conf=0.1):   少量高质量框"
echo ""

echo ""
echo "==========================================="
echo "快速验证命令（需要你实现eval脚本）"
echo "==========================================="
echo ""
echo "# 1. train_1007模型用高阈值重新评估"
echo "python eval.py --weights $MODEL_1007 --conf 0.1 --iou-method aabb"
echo ""
echo "# 2. train_1107模型用AABB评估（看会涨多少）"
echo "python eval.py --weights $MODEL_1107 --conf 0.1 --iou-method aabb"
echo ""
echo "# 3. 两个模型都用精确IoU评估"
echo "python eval.py --weights $MODEL_1007 --conf 0.1 --iou-method exact"
echo "python eval.py --weights $MODEL_1107 --conf 0.1 --iou-method exact"
