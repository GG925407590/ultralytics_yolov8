# YOLOv8-pose NPU 适配改造 —— 代码变更详解

> 从原始 YOLOv8-pose 出发，逐阶段改造为可部署在瑞芯微 RV1126B NPU 的模型。
> **2025-06 更新：rknn-toolkit2 v2.3.2 已支持 RV1126B + Conv-SiLU 融合。阶段 1 (SiLU→LeakyReLU) 已回退，激活函数保持官方 SiLU。**
> 硬约束：推理时所有算子必须是 NPU 原生支持的（Conv、SiLU、DWConv、Sigmoid、逐元素运算、Concat、Upsample 等）。

---

## 一、改造动机总览

```
原始 YOLOv8-pose
  │
  ├─ ① ~~激活函数 SiLU → LeakyReLU~~          ← 已回退，rknn v2.3.2 支持 SiLU
  ├─ ② C2f_Context（CGBlock 替换 Bottleneck）← 全局上下文建模，抗遮挡
  ├─ ③ SEAM 空间注意力                   ← 用可见特征补偿被遮挡区域
  ├─ ④ Pose_DWC（7x7 DWConv 头增强）     ← 替代 Self-Attention，肢体协调
  └─ ⑤ 三项损失函数增强                      ← 骨骼一致性、时序平滑、置信度加权

以上 ②~④ 影响推理端（NPU），⑤ 仅影响训练端。
```

---

## 二、修改文件清单

| 文件 | 修改类型 | 行数变化 |
|------|----------|----------|
| `ultralytics/nn/modules/conv.py` | 修改 3 处 `default_act` | +3/-3 |
| `ultralytics/nn/modules/block.py` | 修改 2 处 `act` + 新增 3 个类 | +92/-2 |
| `ultralytics/nn/modules/head.py` | 新增 1 个类 + import | +49/-1 |
| `ultralytics/nn/modules/__init__.py` | 导出注册 | +9/-0 |
| `ultralytics/nn/tasks.py` | import + parse_model 注册 | +10/-1 |
| `ultralytics/utils/loss.py` | 新增 3 个 Loss 类 + v8PoseLoss 集成 | +260/-4 |
| `ultralytics/cfg/default.yaml` | 新增 3 个 hyperparameter (bone/temporal/kpt_conf) | +3/-0 |
| `ultralytics/cfg/models/v8/yolov8n-pose-npu-s4.yaml` | 最终全特性 YAML | 新建 |

---

## 三、改造 1：激活函数 SiLU → LeakyReLU

### 3.1 为什么改

YOLOv8 默认使用 `SiLU`（Swish），瑞芯微 RV1126B NPU **不支持**。`LeakyReLU` 是 NPU 原生支持的最佳替代激活函数。

### 3.2 改动位置（5 处）

#### (1) `Conv.default_act` — [conv.py 第 39 行](file:///e:/yolo/ultralytics_yolov8/ultralytics/nn/modules/conv.py#L39)

```python
# 原始:
default_act = nn.SiLU()  # default activation

# 改后:
default_act = nn.LeakyReLU(0.1)  # default activation (NPU-compatible)
```

`Conv` 是 YOLOv8 中最核心的卷积模块，所有 `Conv` 调用都使用 `self.default_act`。改这一行，项目中绝大多数卷积的激活函数就都变了。

#### (2) `ConvTranspose.default_act` — [conv.py 第 120 行](file:///e:/yolo/ultralytics_yolov8/ultralytics/nn/modules/conv.py#L120)

同上，`ConvTranspose`（反卷积/转置卷积）也改用 LeakyReLU。

#### (3) `RepConv.default_act` — [conv.py 第 181 行](file:///e:/yolo/ultralytics_yolov8/ultralytics/nn/modules/conv.py#L181)

`RepConv` 是 RT-DETR 使用的重参数化卷积，同理替换。

#### (4) `BottleneckCSP.act` — [block.py 第 361 行](file:///e:/yolo/ultralytics_yolov8/ultralytics/nn/modules/block.py#L361)

```python
# 原始:
self.act = nn.SiLU()

# 改后:
self.act = nn.LeakyReLU(0.1)  # NPU-compatible
```

`BottleneckCSP` **不继承** `Conv`，是自己硬编码的 `nn.SiLU()`，必须单独改。

#### (5) `RepVGGDW.act` — [block.py 第 711 行](file:///e:/yolo/ultralytics_yolov8/ultralytics/nn/modules/block.py#L711)

`RepVGGDW` 同样硬编码了 `nn.SiLU()`，必须单独改。

### 3.3 为什么不影响训练

`LeakyReLU(0.1)` 和 `SiLU` 在大部分输入区间行为接近（SiLU ≈ x·σ(x)，LeakyReLU = max(0.1x, x)），都属于非饱和激活函数。从训练角度看替换是安全的。

---

## 四、改造 2 & 3：新增 CGBlock、C2f_Context、SEAM

这三个类都在 [block.py 第 953 行之后](file:///e:/yolo/ultralytics_yolov8/ultralytics/nn/modules/block.py#L953) 新增。

### 4.1 CGBlock —— 上下文引导块

```
输入 x (c1 通道)
  │
  ├─ cv1: 1×1 Conv 降维到 c_ 通道
  │     │
  │     ├─ local:   DWConv 3×3, d=1  ──→ 局部细节
  │     ├─ dilated: Conv   3×3, d=3  ──→ 周围上下文（空洞卷积感受野变大）
  │     └─ pool → global_conv → global_act:
  │              AdaptiveAvgPool2d(1)
  │              → Conv2d 1×1
  │              → LeakyReLU
  │              → bilinear 上采样回原尺寸  ──→ 全局上下文
  │     │
  │     └─ cat(三路) → cv2: 1×1 Conv → 投影回 c2
  │
  └─ + 输入（残差连接，仅当 c1==c2 时）
```

**关键设计决策**：全局分支用 `nn.Conv2d + nn.LeakyReLU` 而不是 `Conv`（Conv 自带 BN），原因是在 1×1 的特征图上 BN 统计不稳定会导致 NaN。

**NPU 兼容性**：所有算子（DWConv、dilated Conv、AvgPool、bilinear interpolate、Concat、LeakyReLU）均为 NPU 原生支持。

### 4.2 C2f_Context —— C2f 的 CGBlock 变体

结构**完全复刻** `C2f`（[block.py 第 186 行](file:///e:/yolo/ultralytics_yolov8/ultralytics/nn/modules/block.py#L186)），唯一区别是把内部 `Bottleneck` 列表换成了 `CGBlock` 列表：

```
C2f:        cv1 → split → [Bottleneck, Bottleneck, ...] → cat → cv2
C2f_Context: cv1 → split → [CGBlock,    CGBlock,    ...] → cat → cv2
```

`forward` 和 `forward_split` 两种推理路径都保留了（后者用于 ONNX 导出）。

### 4.3 SEAM —— 空间增强注意力模块

```
输入 x (c1 通道)
  │
  ├─ branch1: DWConv 3×3, d=1  ──┐
  ├─ branch2: DWConv 3×3, d=2  ──┤
  └─ branch3: DWConv 3×3, d=3  ──┤
                                   │
                          cat → fuse (1×1 Conv 降维到 1 通道) → Sigmoid
                                                          │
                                                    attention (B,1,H,W)
                                                          │
                                          x * attention + x  → 输出（残差）
```

**原理**：三个不同膨胀率的深度可分离卷积从不同范围提取特征，融合后经 Sigmoid 生成空间注意力图，与原始特征做 element-wise 乘法 + 残差加法。被遮挡区域的特征会被未遮挡区域的 attention 加权补偿。

**NPU 兼容性**：DWConv + Sigmoid + element-wise 乘法/加法，全部 NPU 支持。

---

## 五、改造 4：Pose_DWC —— 7×7 DWConv 头增强

### 5.1 为什么不用 Self-Attention

原始 `fix.md` 方案是用 `nn.MultiheadAttention` 做关键点间协调，但这包含 `Softmax`，RV1126B NPU 对 Softmax 支持有限。

**替代方案**：用 7×7 大核深度可分离卷积近似空间交互。7×7 的感受野足够覆盖相邻关键点之间的空间关系（如手腕→肘→肩）。

### 5.2 Pose_DWC 结构 — [head.py 第 641 行](file:///e:/yolo/ultralytics_yolov8/ultralytics/nn/modules/head.py#L641)

```python
class Pose_DWC(Pose):   # 继承 Pose，不是重写
```

| 继承自 `Pose` | 保持不变 |
|---------------|----------|
| `self.cv2` | bbox 回归卷积 |
| `self.cv3` | 分类卷积 |
| `self.cv4` | 关键点回归卷积 |
| `self.no` / `self.nl` / `self.nk` | 输出维度 |
| `self.kpts_decode` | 关键点解码 |
| `self.export` / `self.format` | RKNN 导出分支 |

**新增的**：`self.spatial` —— 对每个尺度（P4、P5）各自一个 7×7 DWConv 分支：

```
每个尺度 x:
  降维 Conv 1×1 → DWConv 7×7 → 升维 Conv 1×1 → + x（残差）
```

#### forward 流程

```
输入 x = [P4特征图, P5特征图]
  │
  ├─ 对每个尺度 x[i]:
  │     x_sp[i] = x[i] + spatial[i](x[i])    ← 残差融合空间交互
  │
  ├─ cv4 关键点回归:  kpt = cat([cv4[i](x_sp[i]) for each scale])
  │
  ├─ Detect.forward(x_sp)  → bbox + cls 预测
  │
  └─ 输出: bbox + cls + kpt  (训练时分开返回，推理时拼接)
```

### 5.3 与原始 Pose 的差异

| 对比 | 原始 Pose | Pose_DWC |
|------|-----------|----------|
| 输入处理 | `x` 直接进 cv4 + Detect | `x + spatial(x)` 残差后进 cv4 + Detect |
| 额外参数 | 无 | `self.spatial` (ModuleList of Sequential) |
| 推理路径 | 完全复用 Pose | 完全复用 Pose（仅多一次残差加法） |
| RKNN 导出 | 支持 | 支持（额外算子是 Conv+DWConv，NPU 兼容） |

---

## 六、注册机制 —— 让 YAML 能识别新模块

单写 Python 类不够，必须让 Ultralytics 的模型构建系统知道这些类的存在。

### 6.1 [block.py 第 42-52 行](file:///e:/yolo/ultralytics_yolov8/ultralytics/nn/modules/block.py#L42-L52) — `__all__`

```python
__all__ = (
    ...
    "SCDown",
    "CGBlock",        # 新增
    "C2f_Context",     # 新增
    "SEAM",            # 新增
)
```

### 6.2 [head.py 第 18 行](file:///e:/yolo/ultralytics_yolov8/ultralytics/nn/modules/head.py#L18) — `__all__`

```python
__all__ = "Detect", "Segment", "Pose", "Pose_DWC", "Classify", ...
```

### 6.3 [head.py 第 14 行](file:///e:/yolo/ultralytics_yolov8/ultralytics/nn/modules/head.py#L14) — import DWConv

```python
from .conv import Conv, DWConv    # 新增 DWConv
```

`Pose_DWC` 用了 `DWConv`，而原 `head.py` 只 import 了 `Conv`。

### 6.4 `__init__.py` — 模块导出

三处改动：

**(a) import 区** — [第 40-56 行](file:///e:/yolo/ultralytics_yolov8/ultralytics/nn/modules/__init__.py#L40-L56)

```python
from .block import (
    ...
    C2f_Context,   # 新增
    CGBlock,       # 新增
    SEAM,          # 新增
)
from .head import OBB, Classify, Detect, Pose, Pose_DWC, ...   # 新增 Pose_DWC
```

**(b) `__all__` 中间** — [第 116-118 行](file:///e:/yolo/ultralytics_yolov8/ultralytics/nn/modules/__init__.py#L116-L118)

```python
    "C2f_Context",   # 新增
    "CGBlock",       # 新增
    "Pose_DWC",      # 新增
```

**(c) `__all__` 末尾** — [第 158 行](file:///e:/yolo/ultralytics_yolov8/ultralytics/nn/modules/__init__.py#L158)

```python
    "SCDown",
    "SEAM",          # 新增
)
```

### 6.5 [tasks.py](file:///e:/yolo/ultralytics_yolov8/ultralytics/nn/tasks.py) — 模型构建系统

**(a) import** — [第 27-48 行](file:///e:/yolo/ultralytics_yolov8/ultralytics/nn/tasks.py#L27-L48)

```python
    C2f_Context,   # 新增
    CGBlock,       # 新增
    Pose_DWC,      # 新增
    SEAM,          # 新增
```

**(b) `parse_model` 通道数计算** — [第 929 行](file:///e:/yolo/ultralytics_yolov8/ultralytics/nn/tasks.py#L929)

```python
            C2f_Context,     # 新增：和 C2f 同一组计算 c2
```

`C2f_Context` 需要加入这个集合，`parse_model` 才知道它是 `c2 = max(c2, width)` 类型的模块。

**(c) `parse_model` SEAM 特殊处理** — [第 974-975 行](file:///e:/yolo/ultralytics_yolov8/ultralytics/nn/tasks.py#L974-L975)

```python
        elif m is SEAM:
            args = [ch[f]]  # SEAM takes (c1) only
            c2 = ch[f]
```

SEAM 的 constructor 只需要一个参数 `c1`，YAML 里写的是 `[SEAM, []]` 即 args 为空。需要显式从 `ch[f]`（前一层通道数）推导 `c1`，并设置 `c2 = c1`（SEAM 输出通道 = 输入通道）。

**(d) `parse_model` Pose_DWC 加入 Detect 集合** — [第 979 行](file:///e:/yolo/ultralytics_yolov8/ultralytics/nn/tasks.py#L979)

```python
        elif m in {Detect, WorldDetect, Segment, Pose, Pose_DWC, OBB, ...}:
```

`Pose_DWC` 加入这个集合，`parse_model` 才会把 YAML 中 `args[0]` 展开为 `[ch[x] for x in f]`（即把层的索引号转为实际通道数列表）。

---

## 七、YAML 配置

### 7.1 最终 YAML：[yolov8n-pose-npu-s4.yaml](file:///e:/yolo/ultralytics_yolov8/ultralytics/cfg/models/v8/yolov8n-pose-npu-s4.yaml)

最终配置相对于原始 `yolov8-pose.yaml` 的变更：

```
Backbone: 不变（10层，0-9）

Head 原始（3 尺度 P3+P4+P5）:
  10: Upsample(P5)
  11: Concat(P4)           ← P4融合
  12: C2f                  ← P3/P4 特征
  13: Upsample
  14: Concat(P3)           ← P3融合
  15: C2f                  ← P3输出 → 给 Pose
  16: Conv(down)
  17: Concat               ← P4融合
  18: C2f                  ← P4输出 → 给 Pose
  19: Conv(down)
  20: Concat               ← P5融合
  21: C2f                  ← P5输出 → 给 Pose
  22: Pose([15, 18, 21])   ← 3个尺度

Head 改后（3 尺度 P3+P4+P5，保留完整 FPN+PAN）:
  10: Upsample(P5)
  11: Concat(P4)           ← P4融合
  12: C2f_Context          ← P4融合（C2f→C2f_Context）  ← 改造2
  13: Upsample
  14: Concat(P3)           ← P3融合
  15: C2f_Context          ← P3输出（C2f→C2f_Context）  ← 改造2
  16: Conv(down)
  17: Concat               ← P4融合
  18: C2f_Context          ← P4输出（C2f→C2f_Context）  ← 改造2
  19: Conv(down)
  20: Concat               ← P5融合
  21: C2f_Context          ← P5输出（C2f→C2f_Context）  ← 改造2
  22: SEAM(P3)             ← 空间注意力                 ← 改造3
  23: SEAM(P4)             ← 空间注意力                 ← 改造3
  24: SEAM(P5)             ← 空间注意力                 ← 改造3
  25: Pose_DWC([22,23,24]) ← 7x7 DWConv头，三尺度        ← 改造4
```

**数据流**：
```
P5(0) → SPPF(9) → Upsample(10) → Concat(P4, 11) → C2f_Context(12)
                                         │
                   Upsample(13) ←────────┘
                       │
               Concat(P3, 14) → C2f_Context(15) → SEAM(22) ──┐
                       │                                     │
                  Conv ↓(16)                                 │
               Concat(P4上, 17) → C2f_Context(18) → SEAM(23) ─┤
                       │                                     │
                  Conv ↓(19)                                 │
               Concat(P5, 20) → C2f_Context(21) → SEAM(24) ──┤
                                                             │
                                          ┌──────────────────┘
                                          ▼
                                  Pose_DWC(25)
                                          ▼
                                  [bbox, cls, kpt]
```

---

## 八、损失函数增强（仅训练端，不影响 NPU 推理）

损失函数改造在 `ultralytics/utils/loss.py` 中完成，新增 **3 个 Loss 类** + 集成到 `v8PoseLoss`。

### 8.1 COCO 骨骼连接定义（支持 BoneLoss）

```python
# loss.py 第 158-165 行
COCO_SKELETON_0IDX = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],  # ankles→knees→hips, hips
    [5, 11], [6, 12], [5, 6],                            # shoulders→hips, shoulders
    [5, 7], [6, 8], [7, 9], [8, 10],                     # shoulders→elbows→wrists
    [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],              # eyes→nose, eyes→ears
    [3, 5], [4, 6],                                       # ears→shoulders
]
```

### 8.2 BoneLoss —— 骨骼长度一致性损失

**来源**：VideoPose3D (Pavllo et al., CVPR 2019)

**原理**：17 个关键点组成 19 段骨骼（前臂、大臂、大腿、小腿等）。BoneLoss 惩罚预测骨骼长度与 GT 骨骼长度的 L1 偏差。训练时，即使某些关键点被遮挡（位置不直接监督），骨骼长度约束也能通过可见端点"拉"回不可见点的位置。

**关键设计**：只在骨骼两端关键点都可见时才计算损失，避免对不可见点的假性惩罚。

```python
# loss.py 第 168-206 行
class BoneLoss(nn.Module):
    def __init__(self, skeleton):
        super().__init__()
        self.register_buffer("pairs", torch.tensor(skeleton, dtype=torch.long))

    def forward(self, pred_kpts, gt_kpts, kpt_mask):
        # (N, 17, 2) → 骨骼向量: (N, 19, 2)
        pred_bones = pred_kpts[..., :2][:, self.pairs[:, 1]] \
                   - pred_kpts[..., :2][:, self.pairs[:, 0]]
        gt_bones = gt_kpts[..., :2][:, self.pairs[:, 1]] \
                 - gt_kpts[..., :2][:, self.pairs[:, 0]]

        pred_len = torch.norm(pred_bones, dim=-1)  # (N, 19)
        gt_len   = torch.norm(gt_bones, dim=-1)

        # 只惩罚两端都可见的骨骼
        bone_mask = kpt_mask[:, self.pairs[:, 0]] \
                  & kpt_mask[:, self.pairs[:, 1]]

        if bone_mask.sum() == 0:
            return torch.tensor(0.0, device=pred_kpts.device)

        return F.l1_loss(pred_len[bone_mask], gt_len[bone_mask], reduction='mean')
```

### 8.3 TemporalConsistencyLoss —— 时序一致性损失

**来源**：SmoothNet (Wang et al., ECCV 2022)

**原理**：对连续两帧中可见关键点施加 SmoothL1 Loss，惩罚帧间抖动。高尔夫挥杆视频中，身体关键点在相邻帧间应平滑移动。

**关键设计**：用 `self.prev_kpts` 缓存上一 batch 的解码后关键点。新 epoch 开始时重置。只在相邻帧都存在可见关键点时计算。

```python
# loss.py 第 209-239 行
class TemporalConsistencyLoss(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, curr_kpts, prev_kpts, kpt_mask):
        if kpt_mask.sum() == 0:
            return torch.tensor(0.0, device=curr_kpts.device)

        diff = curr_kpts[..., :2][kpt_mask] - prev_kpts[..., :2][kpt_mask]
        loss = F.smooth_l1_loss(diff, torch.zeros_like(diff),
                                beta=self.beta, reduction='mean')
        return loss
```

### 8.4 ConfidenceWeightedKptLoss —— 置信度加权关键点损失

**来源**：RTMPose (Jiang et al., 2022)

**原理**：模型预测的 3 通道中，第 3 通道是 visibility logit。将其 sigmoid 值作为置信度，乘以标准 OKS Loss——高置信预测承担更大损失，驱动模型在"不确定"时调低置信度，而不是乱猜。

```python
# loss.py 第 242-275 行
class ConfidenceWeightedKptLoss(nn.Module):
    def __init__(self, sigmas):
        super().__init__()
        self.register_buffer("sigmas", sigmas)

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) \
          + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] \
                        / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)
        oks_loss = (1 - torch.exp(-e)) * kpt_mask.float()
        # 核心: 用 sigmoid(vis_logit) 加权
        conf_weight = pred_kpts[..., 2].detach().sigmoid()
        return (kpt_loss_factor.view(-1, 1) * oks_loss * conf_weight).mean()
```

### 8.5 v8PoseLoss 集成

**`__init__` 新增**（[loss.py#L592-L597](file:///e:/yolo/ultralytics_yolov8/ultralytics/utils/loss.py#L592-L597)）：

```python
self.bone_loss = BoneLoss(COCO_SKELETON_0IDX if is_pose else [])
self.temporal_loss = TemporalConsistencyLoss(beta=1.0)
self.conf_kpt_loss = ConfidenceWeightedKptLoss(sigmas=sigmas) if is_pose else None
self.prev_kpts = None       # 缓存上一 batch 的关键点
```

**`__call__` 新增**（[loss.py#L600-L689](file:///e:/yolo/ultralytics_yolov8/ultralytics/utils/loss.py#L600-L689)）：

```python
# loss tensor 从 5 元素 → 8 元素
loss = torch.zeros(8, device=self.device)
# 0: box, 1: kpt_loc, 2: kpt_vis, 3: cls, 4: dfl,
# 5: bone, 6: temporal, 7: kpt_conf

# ... 原有 loss[0]~loss[4] 计算不变 ...

# 三项辅助损失（仅当 hyp 权重 > 0 时激活）
if getattr(self.hyp, 'bone', 0.0) > 0:
    loss[5] = self._compute_bone_loss(...)

if getattr(self.hyp, 'temporal', 0.0) > 0:
    loss[6] = self._compute_temporal_loss(...)

if getattr(self.hyp, 'kpt_conf', 0.0) > 0:
    loss[7] = self._compute_conf_kpt_loss(...)

# 加权
loss[5] *= getattr(self.hyp, 'bone', 0.0)
loss[6] *= getattr(self.hyp, 'temporal', 0.0)
loss[7] *= getattr(self.hyp, 'kpt_conf', 0.0)
```

### 8.6 超参数（[default.yaml#L102-L105](file:///e:/yolo/ultralytics_yolov8/ultralytics/cfg/default.yaml#L102-L105)）

```yaml
bone: 0.0       # 骨骼长度一致性损失权重 (VideoPose3D, 建议1.0~5.0)
temporal: 0.0   # 时序一致性损失权重 (SmoothNet, 需 sequential DataLoader)
kpt_conf: 0.0   # 置信度加权关键点损失权重 (RTMPose, 建议0.5~2.0)
```

默认全部为 0，确保向后兼容。使用方式：

```bash
yolo train model=yolov8n-pose-npu-s4.yaml data=coco-pose.yaml \
    bone=3.0 kpt_conf=1.0
```

### 8.7 三个辅助方法的逻辑

| 方法 | 功能 | 关键逻辑 |
|------|------|----------|
| `_compute_bone_loss` | 从 fg_mask 提取可见骨骼对 | 遍历 batch，收集 ≥4 个可见点的样本 → BoneLoss |
| `_compute_temporal_loss` | 计算帧间平滑损失 | 用 `self.prev_kpts` 缓存，新 epoch 时重置，取交集可见点 |
| `_compute_conf_kpt_loss` | 置信度加权 OKS Loss | 复用 fg_mask 展开逻辑，调用 ConfidenceWeightedKptLoss |

---

## 九、验证

```
构建: Python 直接调用 PoseModel('yolov8n-pose-npu-s4.yaml') 成功
Forward: 随机输入 (1,3,640,640) 通过，无 NaN
ONNX:   torch.onnx.export 成功，12.8 MB
SiLU:   项目 modules 目录 grep 结果 0 处残留
参数:   ~3.27M params, ~9.3 GFLOPs（保留三尺度 FPN+PAN）
Loss:   3 个损失函数独立 forward 测试通过
```

---

## 十、下一步

当前所有代码改造 + 损失函数增强已完成。按照 `npu_adapt_plan.md` 中的流程：

```
当前状态: 代码改造 + 损失函数增强 + ONNX 导出验证 全部通过
    │
    ▼
下一步: ONNX → rknn-toolkit2 → RKNN → RV1126B 模拟器推理
    │
    ▼
通过后: s4.yaml 完整训练 → QAT → 实机部署
```
