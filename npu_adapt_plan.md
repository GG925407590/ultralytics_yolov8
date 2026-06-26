# YOLOv8-pose RV1126B NPU 适配改造计划

> 基于 `fix.md` 的 6 项改造，结合源码可行性分析后制定。
> 核心约束：推理时模型结构必须完全兼容瑞芯微 NPU（仅使用标准卷积、池化、LeakyReLU、逐元素运算）。

---

## 全局约定

### YAML 版本控制

每次修改 YAML 都另存为新文件，命名规则：

```
yolov8n-pose-npu-s1.yaml   # 阶段1产物（LeakyReLU，保留三尺度）
yolov8n-pose-npu-s2.yaml   # 阶段2产物（+ CGBlock/C2f_Context）
yolov8n-pose-npu-s3.yaml   # 阶段3产物（+ SEAM）
yolov8n-pose-npu-s4.yaml   # 阶段4产物（+ Pose_DWC）
yolov8n-pose-npu-s5.yaml   # 阶段5产物（+ TSM，可选）
```

每个 YAML 文件头部加注释说明改动点，便于回溯。

### QAT 策略

不在每个阶段做 QAT。阶段 4 完成后，生成一个 FP32 的"全特性"模型，**统一做一次 QAT**，一次性量化部署到 NPU。

### 性能验收标准

在 RV1126B 实机上测试，测量"NPU 推理 + 后处理"端到端耗时：
- 100 帧总耗时不超过 5 秒（单帧平均 ≤ 50ms）
- 注意 NPU 频率、DDR 带宽对实测的影响，不只看 TOPS 纸面数据

---

## 核心策略：先验证算子，再投入训练

**全部代码改造完成后，不训练，直接用随机权重导出 ONNX → 转 RKNN → 模拟器推理。模拟器跑通后才开始训练。** 这样可以避免投入大量训练资源后发现 NPU 不支持某个算子而返工。

```
代码改造完成 → ONNX 导出 → rknn-toolkit2 转换 → RV1126B 模拟器推理 → ✅ 通过 → 开始训练
                                                    └─ ❌ 不通过 → 定位不支持算子 → 修改 → 重新验证
```

> 随机权重的模型在 RKNN 转换/模拟器阶段与训练后模型行为一致（算子层面），足以暴露所有不兼容算子。

---

## 阶段 0：基线准备

改造前必须完成的基础工作。

| # | 任务 | 操作 | 验证标准 |
|---|------|------|----------|
| 0.1 | 训练基线模型 | 用 `yolov8n-pose.yaml` 完整训练一轮 | 记录 mAP、FLOPs、参数量 |
| 0.2 | ONNX 导出验证 | `format=rknn` 导出 ONNX | 无报错 |
| 0.3 | 整理测试集 | 准备高尔夫挥杆近景遮挡样本 | 可视化确认标注正确 |

---

## 阶段 1：基础兼容层（已废弃）

**rknn-toolkit2 v2.3.2 已支持 RV1126B + SiLU 融合，此阶段不需要执行。激活函数保持 SiLU。**

<details>
<summary>原始计划（已废弃，仅供参考）</summary>

**目标：SiLU → LeakyReLU。保留 P3+P4+P5 三尺度检测（关键点定位需高分辨率特征图）。**

### 1.1 激活函数替换

| # | 文件 | 行 | 改动 |
|---|------|----|------|
| 1.1 | `ultralytics/nn/modules/conv.py` | L38 | `Conv.default_act = nn.LeakyReLU(0.1)` |
| 1.2 | `ultralytics/nn/modules/conv.py` | L113 | `ConvTranspose.default_act = nn.LeakyReLU(0.1)` |
| 1.3 | `ultralytics/nn/modules/conv.py` | L173 | `RepConv.default_act = nn.LeakyReLU(0.1)` |
| 1.4 | `ultralytics/nn/modules/block.py` | L287 | `BottleneckCSP.act = nn.LeakyReLU(0.1)` |
| 1.5 | `ultralytics/nn/modules/block.py` | L510 | `RepVGGDW.act = nn.LeakyReLU(0.1)` |

</details>

### 1.2 新建 s1 YAML

新建 `ultralytics/cfg/models/v8/yolov8n-pose-npu-s1.yaml`，Head 结构与原始 `yolov8-pose.yaml` 一致（三尺度 FPN+PAN），仅伴随激活函数全局替换生效。

```yaml
# Head 保持原始三尺度结构（P3+P4+P5）
# 不作任何层裁剪，保证关键点定位精度

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]     # P5 → 上采样
  - [[-1, 6], 1, Concat, [1]]                       # 拼接 backbone P4
  - [-1, 3, C2f, [512]]                              # P4 融合

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]       # P4 → 上采样
  - [[-1, 4], 1, Concat, [1]]                        # 拼接 backbone P3
  - [-1, 3, C2f, [256]]                              # P3 输出

  - [-1, 1, Conv, [256, 3, 2]]                      # 下采样
  - [[-1, 12], 1, Concat, [1]]                       # 拼接 P4 路径
  - [-1, 3, C2f, [512]]                              # P4 输出

  - [-1, 1, Conv, [512, 3, 2]]                      # 下采样
  - [[-1, 9], 1, Concat, [1]]                        # 拼接 backbone P5
  - [-1, 3, C2f, [1024]]                             # P5 输出

  - [[15, 18, 21], 1, Pose, [nc, kpt_shape]]         # 三尺度 Pose
```

> 不砍 P3 的原因：关键点是像素级小目标，P3 (80×80) 的精细网格对定位精度至关重要。

| 验证项 | 标准 |
|--------|------|
| ONNX 导出 | 随机权重 + `do_constant_folding=True` 无报错 |

> 此阶段不训练。所有阶段代码改造完成后，统一用 s4.yaml（随机权重）做 NPU 算子验证。

---

## 阶段 2：上下文增强模块（低风险）

**目标：CGBlock + C2f_Context 提升遮挡推理能力。**

### 2.1 新建 CGBlock

**文件**：`ultralytics/nn/modules/block.py`

```
CGBlock(c1, c2, shortcut=True, e=0.5):
  └─ 分支1: DWConv(c_, c_, 3, 1)              # 局部细节
  └─ 分支2: Conv(c_, c_, 3, d=3)              # 空洞上下文
  └─ 分支3: AdaptiveAvgPool2d(1) → Conv → 上采样  # 全局上下文
  └─ 融合: 三分支相加 + 残差连接
```

### 2.2 新建 C2f_Context

**文件**：`ultralytics/nn/modules/block.py`

复刻 `C2f` 结构，`self.m` 中用 `CGBlock` 替代 `Bottleneck`。

### 2.3 注册模块

| 文件 | 改动 |
|------|------|
| `ultralytics/nn/modules/__init__.py` | `from .block import ..., CGBlock, C2f_Context` |
| `ultralytics/nn/tasks.py` (import) | 增加 `CGBlock, C2f_Context` |
| `ultralytics/nn/tasks.py` (parse_model) | 集合加入 `C2f_Context` 和 `CGBlock` |

### 2.4 YAML 替换

在 `yolov8n-pose-npu.yaml` 中将 Neck 末端 1~2 个 `C2f` 替换为 `C2f_Context`。

| 验证项 | 标准 |
|--------|------|
| ONNX 导出 | 随机权重导出无报错 |
| NPU 算子审计 | 确认无 SiLU、无 Softmax、无 LayerNorm |

---

## 阶段 3：空间注意力抗遮挡（低风险）

**目标：Pose Head 前插入 SEAM 模块。**

### 3.1 新建 SEAM

**文件**：`ultralytics/nn/modules/block.py`

```
SEAM(c1):
  └─ 分支1: DWConv(c1, c1, 3, d=1)
  └─ 分支2: DWConv(c1, c1, 3, d=2)
  └─ 分支3: DWConv(c1, c1, 3, d=3)
  └─ Concat → Conv(3*c1, 1, 1) → Sigmoid
  └─ 输出 = 输入 * attention + 输入 (残差)
```

### 3.2 YAML 接入

对每个尺度独立插入 SEAM：

```yaml
  - [[P4索引], 1, SEAM, []]
  - [[P5索引], 1, SEAM, []]
  - [[SEAM4索引, SEAM5索引], 1, Pose, [nc, kpt_shape]]
```

| 验证项 | 标准 |
|--------|------|
| ONNX 导出 | 随机权重导出无报错 |
| NPU 兼容 | Sigmoid + DWConv + Concat 均 NPU 原生支持 |

---

## 阶段 4：姿态头增强（中等风险）

**目标：用 7×7 深度卷积替代 Self-Attention，避免 Softmax NPU 风险。**

### 4.1 新建 Pose_DWC

**文件**：`ultralytics/nn/modules/head.py`

```python
class Pose_DWC(Pose):          # 继承 Pose
    def __init__(self, nc, kpt_shape, ch):
        super().__init__(nc, kpt_shape, ch)
        c_ = max(ch[0] // 4, self.nk)
        self.sa = nn.ModuleList(
            nn.Sequential(
                Conv(x, c_, 1, 1),     # 降维
                DWConv(c_, c_, 7, 1),   # 大核空间交互
                Conv(c_, x, 1, 1),      # 升维
            ) for x in ch
        )
```

### 4.2 YAML 替换

将 `Pose` 替换为 `Pose_DWC`。

| 验证项 | 标准 |
|--------|------|
| ONNX 导出 | 随机权重导出无报错 |
| NPU 兼容 | 仅含 Conv + DWConv，零风险 |

### 4.3 统一 QAT（训练后执行）

阶段 1~4 改造的代码全部通过 NPU 模拟器验证后，进行一次完整训练，然后对 FP32 模型**统一做一次 QAT**，一步到位量化部署。不必每个阶段单独做 QAT。

### 4.4 损失函数增强（仅训练端，不影响 NPU 推理）

在 `ultralytics/utils/loss.py` 的 `v8PoseLoss` 中新增三项可选损失：

| 损失项 | 来源 | 功能 | 超参数 |
|--------|------|------|--------|
| **BoneLoss** | VideoPose3D (CVPR 2019) | 惩罚预测骨骼长度与 GT 的 L1 偏差，约束肢体比例 | `bone` (默认 0) |
| **TemporalConsistencyLoss** | SmoothNet (ECCV 2022) | 惩罚相邻帧关键点位移，减少帧间抖动 | `temporal` (默认 0) |
| **ConfidenceWeightedKptLoss** | RTMPose (2022) | 用 sigmoid(vis_logit) 加权 OKS Loss，模型学会"不确定时调低置信" | `kpt_conf` (默认 0) |

**使用方式**：`yolo train ... bone=3.0 kpt_conf=1.0`（默认全关，向后兼容）

| 验证项 | 标准 |
|--------|------|
| Loss forward | 三个 loss 独立输入测试通过（已验证） |
| 训练兼容 | 权重为 0 时 loss[5:8] 不参与梯度，与原版行为一致 |

> 此项改造已实现完毕，代码在 `loss.py` 中。

---

## 阶段 5：TSM 时序建模（可选，高风险高收益）

**目标：训练时注入时序信息，推理时折叠为零开销。**

### 5.1 新建 TSMConv

**文件**：`ultralytics/nn/modules/conv.py`

训练时通道移位，推理时等价于普通 Conv2d。

### 5.2 DataLoader 改造

- 支持返回 3 帧堆叠 (B, 9, H, W)
- 中间帧关键点为标签，前后帧仅视觉上下文
- 3 帧同步数据增强

### 5.3 推理重参数化

将移位权重吸收进卷积核，转换为标准 `nn.Conv2d`。

| 验证项 | 标准 |
|--------|------|
| 推理结构 | 与普通 Conv2d 一致，NPU 无感知 |
| 抖动指标 | 连续帧间关键点位置方差降低 |

---

## 实施路线图

```
阶段0：基线准备 ──────────────────────────── 原始 yolov8n-pose 训练 + ONNX 导出
    │
    ├─ 代码改造（不训练）─────────────────── 阶段1~4 全部代码改动 + YAML 文件
    │   │
    │   ├─ 阶段1：LeakyReLU（保留三尺度） ──── 5处改动 + s1.yaml
    │   ├─ 阶段2：CGBlock + C2f_Context  ─── 3处改动 + s2.yaml
    │   ├─ 阶段3：SEAM                     ─ 2处改动 + s3.yaml
    │   └─ 阶段4：Pose_DWC                 ─ 2处改动 + s4.yaml
    │
    ├─ 🔑 NPU 算子验证（硬卡点）────────────── 不训练！随机权重 s4.yaml → ONNX → RKNN → 模拟器
    │   │
    │   ├─ ❌ 不通过 → 定位不支持算子 → 修改代码 → 重新验证
    │   └─ ✅ 通过 → 进入训练阶段
    │
    ├─ 训练 ──────────────────────────────── 用 s4.yaml（全特性） + 损失函数增强训练
    │   │
    │   ├─ 模型: LeakyReLU + CGBlock + C2f_Context + SEAM + Pose_DWC
    │   ├─ Loss: BoneLoss(bone=3.0) + ConfKptLoss(kpt_conf=1.0) + 原始 OKS Loss
    │   ├─ TemporalLoss 按需开启（需要时序 DataLoader）
    │   └─ 验证：mAP 不低于基线
    │
    ├─ 统一 QAT ──────────────────────────── 全特性 FP32 → 一次性量化
    │
    ├─ 实机性能测试 ───────────────────────── RV1126B 实机
    │   ├─ "仅 NPU 推理" 耗时
    │   ├─ "NPU + 后处理" 端到端耗时
    │   └─ 验收：100帧 ≤ 5秒
    │
    └─ 阶段5：TSM 时序建模 [可选]  ────────── 需 DataLoader 改造 + 时序数据
```

**核心原则**：代码全部改造 → 模拟器先跑通 → 再训练。不训练就导出 ONNX 转 RKNN 跑模拟器，是验证 NPU 算子兼容性最高效的方法。
