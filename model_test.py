"""
YOLO26x-sem 语义分割验证脚本
数据集配置文件: data.yaml
参考: https://docs.ultralytics.com/zh/modes/val/
"""

from ultralytics import YOLO
from ultralytics.data.utils import add_polygon_background
from ultralytics.utils import YAML

# 加载训练好的语义分割模型
model = YOLO("best.pt")

# 数据集
data_source = "data.yaml"

# 修复: 多边形标注数据集需要同步 background 类到 model.names
data_dict = add_polygon_background(YAML.load(data_source))
model.model.names = data_dict["names"]

# 验证
metrics = model.val(
    # ========== 基础配置 ==========
    data=data_source,
    imgsz=640,                     # 输入图片尺寸
    batch=16,                      # batch size，根据显存调整
    device="cpu",                  # GPU 设备号；CPU 填 "cpu"
    workers=8,                     # 数据加载线程数
    split="test",                  # 数据集划分: "val" / "test" / "train"

    # ========== 保存与可视化 ==========
    save_json=False,               # 保存预测mask为PNG文件
    plots=True,                    # 绘制混淆矩阵等分析图表

    # ========== 数据集加载 ==========
    rect=True,                     # 矩形推理：按宽高比分组批处理，减少 padding 开销

    # ========== 输出 ==========
    project="./runs",              # 输出根目录
    name="val_semantic",           # 实验名称（子目录）
    exist_ok=True,                 # 覆盖已有输出目录
)

# 打印语义分割指标
print(f"mIoU:         {metrics.miou:.4f}")        # 平均交并比
print(f"Pixel Acc:    {metrics.pixel_accuracy:.4f}")  # 像素准确率
print(f"fitness:      {metrics.fitness:.4f}")   # 综合适应度分数 (= mIoU)
