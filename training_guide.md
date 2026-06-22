# YOLOv8-pose NPU 适配 —— 可执行训练步骤

> 面向非专业边缘 AI 工程师。每一步都可独立执行和验证，出问题时方便定位。

---

## 前置环境确认

在开始前，在 PowerShell 中执行以下命令，确认环境就绪：

```powershell
# 1. 确认在正确目录
Set-Location e:\yolo\ultralytics_yolov8

# 2. 确认 PyTorch + XPU (A770) 可用
python -c "import torch; print('PyTorch:', torch.__version__); print('XPU:', torch.xpu.is_available())"
# 期望输出: PyTorch: 2.7.0+xpu / XPU: True

# 3. 确认模型能构建
python -c "from ultralytics.nn.tasks import PoseModel; m = PoseModel('ultralytics/cfg/models/v8/yolov8n-pose-npu-s4.yaml'); print('OK'); print(f'Params: {sum(p.numel() for p in m.parameters()):,}')"
# 期望输出: OK / Params: 3,267,649

# 4. 确认 ONNX 能导出
python -c "import torch; from ultralytics.nn.tasks import PoseModel; m = PoseModel('ultralytics/cfg/models/v8/yolov8n-pose-npu-s4.yaml'); m.eval(); torch.onnx.export(m, torch.randn(1,3,640,640), 'test.onnx', opset_version=12, do_constant_folding=True, input_names=['images'], output_names=['output0']); import os; print(f'ONNX: {os.path.getsize(\"test.onnx\")/1024/1024:.1f} MB')"
# 期望输出: ONNX: 12.8 MB
```

---

## 第一步：准备训练数据

### 1.1 确认数据格式

你的标注数据需要是以下两种格式之一：

**格式 A：COCO JSON（推荐）**
```
dataset/
├── images/
│   ├── train/
│   │   ├── 0001.jpg
│   │   ├── 0002.jpg
│   │   └── ...
│   └── val/
│       └── ...
└── annotations/
    ├── person_keypoints_train.json
    └── person_keypoints_val.json
```

**格式 B：YOLO txt 格式**
```
dataset/
├── images/
│   ├── train/
│   │   ├── 0001.jpg
│   │   ├── 0001.txt    ← 同名 txt
│   │   └── ...
│   └── val/
└── labels/
    ├── train/
    │   └── ...
    └── val/
```

### 1.2 创建数据集配置文件

在 `ultralytics/cfg/datasets/` 下新建一个 YAML，例如 `golf_swing.yaml`：

```yaml
# Golf Swing Keypoint Detection Dataset
path: E:/datasets/golf_swing   # ← 改成你数据的实际路径
train: images/train
val: images/val

# 关键点定义
kpt_shape: [17, 3]   # 17个关键点, 3维 (x, y, visible)
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

# 类别
names:
  0: person
```

### 1.3 验证数据能读取

```powershell
python -c "
from ultralytics.data.utils import check_det_dataset
data = check_det_dataset('ultralytics/cfg/datasets/golf_swing.yaml')
print(f'Train images: {data[\"train\"].shape if hasattr(data[\"train\"], \"shape\") else \"loaded\"}')
print('Data OK')
"
```

---

## 第二步：验证损失函数（30 秒）

在正式训练前，跑一次快速验证确认损失函数不出错：

```powershell
python -c "
from ultralytics.utils.loss import BoneLoss, TemporalConsistencyLoss, ConfidenceWeightedKptLoss, COCO_SKELETON_0IDX
import torch

# 三个 loss 各跑一次
bl = BoneLoss(COCO_SKELETON_0IDX)
tl = TemporalConsistencyLoss()
x = torch.randn(4, 17, 3)
mask = torch.ones(4, 17, dtype=torch.bool)
print(f'BoneLoss:      {bl(x, x+0.1, mask).item():.4f}  (期望: ~0.1)')
print(f'TemporalLoss:  {tl(x, x+0.1, mask).item():.4f}  (期望: ~0.07)')
print('All loss checks passed!')
"
# 期望: 三个数值都输出，无报错
```

---

## 第三步：试跑 1 epoch（干跑验证）

用最小配置跑 1 个 epoch，确认训练管线完整无 bug：

```powershell
yolo train `
    model=ultralytics/cfg/models/v8/yolov8n-pose-npu-s4.yaml `
    data=ultralytics/cfg/datasets/golf_swing.yaml `
    epochs=1 `
    imgsz=640 `
    batch=8 `
    device=xpu `
    pretrained=True `
    bone=0.0 `
    kpt_conf=0.0 `
    temporal=0.0 `
    workers=2
```

**这一步的目的**：不是看效果，是确认代码全链路不出错（数据读取→前向→loss计算→反向→优化器）。

**如果报错**：
- 显存不够 → 减小 `batch=4` 或 `imgsz=320`
- `device=xpu` 报 CUDA 错误 → 改为 `device=0`（Intel 环境可能直接映射）
- 数据加载报错 → 回到 1.3 检查数据格式

---

## 第四步：第 1 轮训练（激活函数验证）

用 **仅 s1.yaml**（只改 LeakyReLU，不加其他模块），验证激活函数替换不影响收敛：

```powershell
yolo train `
    model=ultralytics/cfg/models/v8/yolov8n-pose-npu-s1.yaml `
    data=ultralytics/cfg/datasets/golf_swing.yaml `
    epochs=100 `
    imgsz=640 `
    batch=16 `
    device=xpu `
    pretrained=True `
    workers=4 `
    name=golf_s1_baseline
```

**期望结果**：
- mAP 与原始 YOLOv8n-pose 在同一数据集上相差不超过 5%
- 如果明显变差，说明你的数据分布对 SiLU→LeakyReLU 敏感，需要调学习率（降 50%）或 warmup 更长

---

## 第五步：第 2 轮训练（全特性 + 基础 loss）

用 **s4.yaml + BoneLoss + ConfidenceLoss** 正式训练：

```powershell
yolo train `
    model=ultralytics/cfg/models/v8/yolov8n-pose-npu-s4.yaml `
    data=ultralytics/cfg/datasets/golf_swing.yaml `
    epochs=200 `
    imgsz=640 `
    batch=16 `
    device=xpu `
    pretrained=True `
    bone=3.0 `
    kpt_conf=1.0 `
    temporal=0.0 `
    workers=4 `
    name=golf_s4_full `
    lr0=0.01 `
    warmup_epochs=3 `
    cos_lr=True
```

**参数解释**：

| 参数 | 值 | 为什么 |
|------|-----|--------|
| `bone=3.0` | 骨骼损失权重 | 让模型关注肢体比例，不瞎猜遮挡点 |
| `kpt_conf=1.0` | 置信度加权权重 | 让模型学会"不确定就低置信"，而不是乱猜 |
| `temporal=0.0` | 关闭时序 | 你的 DataLoader 不是时序的，开启会出错 |
| `lr0=0.01` | 学习率 | 比默认稍低，防止新模块梯度爆炸 |
| `cos_lr=True` | 余弦退火 | 训练末期平滑收敛 |
| `warmup_epochs=3` | 预热 3 epoch | 新层随机初始化，需要预热期 |

**监控指标**：
- 训练过程中观察 `metrics/kpt_loss` 是否持续下降
- 观察 `val/pose_mAP50-95` 的收敛趋势
- 如果 `loss[5]` (bone) 一直为 0，说明你的数据中关键点标注有问题

---

## 第六步：训练后验证

### 6.1 在验证集上看效果

```powershell
yolo val `
    model=runs/pose/golf_s4_full/weights/best.pt `
    data=ultralytics/cfg/datasets/golf_swing.yaml `
    imgsz=640 `
    device=xpu
```

记录下 `pose_mAP50` 和 `pose_mAP50-95`，与第四步的 s1 基线对比。

### 6.2 可视化几个样本

```powershell
yolo predict `
    model=runs/pose/golf_s4_full/weights/best.pt `
    source=E:/datasets/golf_swing/images/val `
    imgsz=640 `
    device=xpu `
    save=True `
    project=runs/predict_golf_s4
```

打开 `runs/predict_golf_s4/` 中的图片，肉眼检查关键点是否合理。

---

## 第七步：ONNX 导出（NPU 验证准备）

```powershell
yolo export `
    model=runs/pose/golf_s4_full/weights/best.pt `
    format=onnx `
    imgsz=640 `
    opset=12 `
    simplify=False
```

输出文件：`runs/pose/golf_s4_full/weights/best.onnx`

验证 ONNX 能正常推理：

```powershell
python -c "
import onnxruntime as ort
import numpy as np
session = ort.InferenceSession('runs/pose/golf_s4_full/weights/best.onnx')
out = session.run(None, {'images': np.random.randn(1,3,640,640).astype(np.float32)})
print(f'Output shape: {out[0].shape}')
# 期望: 有 shape 输出，无报错
"
```

---

## 第八步：RKNN 转换 + 模拟器验证

这一步需要 `rknn-toolkit2`。如果你还没安装：

```powershell
# 安装 rknn-toolkit2（需要 Python 3.8 环境）
pip install rknn-toolkit2==1.6.0
```

### 8.1 RKNN 转换脚本

新建 `convert_to_rknn.py`：

```python
from rknn.api import RKNN

rknn = RKNN()

# 配置
rknn.config(
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
    target_platform="rv1126",
    quantized_dtype="asymmetric_quantized-8",
    optimized_level=0,
)

# 加载 ONNX
ret = rknn.load_onnx(model="runs/pose/golf_s4_full/weights/best.onnx")
if ret != 0:
    print(f"Load ONNX failed: {ret}")
    exit(1)

# 构建 RKNN
ret = rknn.build(do_quantization=False)
if ret != 0:
    print(f"Build RKNN failed: {ret}")
    exit(1)

# 导出
ret = rknn.export_rknn("yolov8n-pose-s4.rknn")
if ret != 0:
    print(f"Export failed: {ret}")
    exit(1)

print("RKNN exported OK!")

# 模拟器推理验证
import numpy as np
dummy = np.random.randn(1, 3, 640, 640).astype(np.float32)
outputs = rknn.inference(inputs=[dummy])
print(f"Simulator output shape: {outputs[0].shape}")
print("Simulator inference OK!")

rknn.release()
```

```powershell
python convert_to_rknn.py
# 期望: RKNN exported OK! / Simulator inference OK!
```

**如果这一步报某个算子不支持**——记录下报错信息，这是关键反馈，我会帮你替换不支持的算子。

### 8.2 实机部署（后续）

模拟器通过后，把 `yolov8n-pose-s4.rknn` 部署到 RV1126B 实机上测试端到端耗时。

```powershell
# 实机推理耗时测试（需要在 RV1126B 上执行，这里仅做模板）
# 目标: 100帧总耗时 ≤ 5秒（单帧 ≤ 50ms）
```

---

## 完整训练命令速查

```powershell
# === 第三步：干跑验证 ===
yolo train model=ultralytics/cfg/models/v8/yolov8n-pose-npu-s4.yaml data=ultralytics/cfg/datasets/golf_swing.yaml epochs=1 imgsz=640 batch=8 device=xpu pretrained=True workers=2

# === 第四步：s1 基线 ===
yolo train model=ultralytics/cfg/models/v8/yolov8n-pose-npu-s1.yaml data=ultralytics/cfg/datasets/golf_swing.yaml epochs=100 imgsz=640 batch=16 device=xpu pretrained=True workers=4 name=golf_s1_baseline

# === 第五步：s4 全特性 ===
yolo train model=ultralytics/cfg/models/v8/yolov8n-pose-npu-s4.yaml data=ultralytics/cfg/datasets/golf_swing.yaml epochs=200 imgsz=640 batch=16 device=xpu pretrained=True bone=3.0 kpt_conf=1.0 temporal=0.0 workers=4 name=golf_s4_full lr0=0.01 warmup_epochs=3 cos_lr=True

# === 第六步：验证 ===
yolo val model=runs/pose/golf_s4_full/weights/best.pt data=ultralytics/cfg/datasets/golf_swing.yaml imgsz=640 device=xpu

# === 第七步：ONNX 导出 ===
yolo export model=runs/pose/golf_s4_full/weights/best.pt format=onnx imgsz=640 opset=12 simplify=False
```

---

## 问题排查速查表

| 现象 | 可能原因 | 解决 |
|------|----------|------|
| `torch.xpu.is_available() = False` | IPEX 未正确安装 | `pip install intel-extension-for-pytorch` |
| `Out of memory` | batch 太大 | 减小 `batch=4` 或 `imgsz=320` |
| `loss[5]` (bone) 始终为 0 | 数据中关键点不可见标记全 0 | 检查 `keypoints[..., 2]` 是否被正确标注 |
| ONNX 导出报 `Unsupported operator` | 模型中有不支持的 op | 把报错贴给我 |
| RKNN 转换报 `xxx op not supported` | NPU 不支持该算子 | 记录算子名，需要替换 |
| 训练 loss 不降 | 学习率不合适 | 先降 `lr0=0.001`，再升 `lr0=0.1` 二分查找 |
| A770 训练报 CUDA 错误 | xpu 后端兼容问题 | 尝试 `device=0` 或 `device=cpu` 先确认非 GPU 问题 |
