"""
YOLO26x-sem 语义分割推理脚本
参考: https://docs.ultralytics.com/zh/modes/predict/
"""

import numpy as np
from ultralytics import YOLO

# 加载训练好的语义分割模型
model = YOLO("yolo26x-sem.pt")

# 推理数据源（支持：图片路径 / 目录 / 视频 / RTSP流 / YouTube / webcam=0）
source = "dataset_predict/QQ2026727-212554.mp4"  # TODO: 替换为实际图片/视频路径

# 推理
results = model.predict(
    source=source,

    # ========== 基础配置 ==========
    imgsz=640,                     # 推理图片尺寸
    conf=0.25,                     # 置信度阈值（语义分割中可能不适用，保留兼容）
    device=0,                      # GPU 设备号；CPU 填 "cpu"
    batch=1,                       # 推理 batch size
    half=True,                     # FP16 半精度推理（省显存加速）

    # ========== 类别过滤 ==========
    classes=None,                  # 只检测指定类别；None=全部；示例: [0, 1]

    # ========== 保存与可视化 ==========
    save=True,                     # 保存标注结果图片/视频
    show=False,                    # 实时窗口显示结果
    line_width=None,               # 边框线宽；None 自动计算

    # ========== 视频/流专用 ==========
    vid_stride=1,                  # 视频帧间隔（1=每帧都处理，N=每隔N帧）
    stream=False,                  # True 时返回生成器，避免视频/流场景内存溢出

    # ========== 输出 ==========
    project="./runs",              # 输出根目录
    name="predict_semantic",       # 实验名称（子目录）
    exist_ok=True,                 # 覆盖已有输出目录
    verbose=True,                  # 打印详细信息
)

# 遍历结果
for i, r in enumerate(results):
    if r.semantic_mask is not None:
        mask = r.semantic_mask.cpu().numpy() if hasattr(r.semantic_mask, 'cpu') else np.array(r.semantic_mask)
        # mask 是 [H, W] 的类别 ID 图，每个像素值代表该位置的类别
        unique_classes = np.unique(mask)
        # 统计每个类别的像素占比
        total_pixels = mask.size
        print(f"\n图片 {i}: 语义分割结果 (尺寸: {mask.shape})")
        for cls_id in unique_classes:
            cls_name = model.names.get(int(cls_id), f"class_{cls_id}")
            pixel_count = (mask == cls_id).sum()
            ratio = pixel_count / total_pixels * 100
            print(f"  {cls_name:<15s}  pixels={pixel_count:>8d}  ratio={ratio:.2f}%")
    else:
        print(f"\n图片 {i}: 无分割结果")
