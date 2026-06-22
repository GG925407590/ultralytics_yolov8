对 YOLOv8-pose 进行针对性改造。所有改造都遵循一个硬约束：**推理时模型结构必须完全兼容瑞芯微 NPU**（仅使用标准卷积、池化、LeakyReLU、逐元素运算等）。

***

## 改造总览

```
原始 YOLOv8-pose
    │
    ├─ 1. 激活函数替换 (SiLU → LeakyReLU)          ← 必做，为 NPU 铺路
    │
    ├─ 2. Backbone/Neck 增强 (C2f → C2f_Context)   ← 引入 CGBlock 全局上下文
    │
    ├─ 3. 空间注意力插入 (SEAM)                    ← 抗遮挡
    │
    ├─ 4. 检测头增强 (Pose Head → Pose_SA)         ← 自注意力肢体协调
    │
    ├─ 5. 时序隐式建模 (TSM 折叠式)                ← 训练时注入时序信息
    │
    └─ 6. 检测分支裁剪 (砍 P3，只留 P4+P5)         ← 适配近景大目标
```

***

## 改造 1：激活函数替换（必做，零风险）

### 为什么

YOLOv8 默认用 `SiLU`（Swish），瑞芯微 NPU 不支持。`LeakyReLU` 是 NPU 原生支持的最佳替代品。

### 操作

修改 `ultralytics/nn/modules.py` 中 `Conv` 类的默认激活函数：

```python
# 找到 Conv 类
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        # ...
        # 将 self.act = nn.SiLU() 改为：
        self.act = nn.LeakyReLU(0.1) if act else nn.Identity()
```

**影响**：训练和推理都使用 `LeakyReLU`，NPU 完美支持。

***

## 改造 2：C2f\_Context（CGBlock 替换 Bottleneck）

### 为什么

原始 `C2f` 里的 `Bottleneck` 只有局部 3×3 感受野。高尔夫挥杆中手腕被遮挡时，需要全局上下文（肩膀、髋部位置）来推理遮挡点位置。

### CGBlock 结构

```
输入
  │
  ├─→ 深度可分离卷积 (3×3) ──→ 局部细节
  ├─→ 空洞卷积 (3×3, d=3)  ──→ 周围上下文
  └─→ 全局池化 → 1×1 Conv → ReLU → 上采样 ──→ 全局上下文
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
        相加融合           残差连接(输入)
          │                   │
          └─────────┬─────────┘
                    ▼
                  输出
```

### 操作

在 `modules.py` 中新建 `CGBlock` 类，然后新建 `C2f_Context` 类（复制 `C2f` 代码，把 `Bottleneck` 换成 `CGBlock`）。

在模型 YAML 中，将 Neck 末端的 1\~2 个 `C2f` 替换为 `C2f_Context`。

***

## 改造 3：SEAM 空间增强注意力 

### 为什么

球杆遮挡手腕时，SEAM 能用可见的肘、肩特征去“补偿”被遮挡区域的特征响应。

### 结构

```
输入
  │
  ├─→ 深度可分离卷积 (3×3, d=1) ─┐
  ├─→ 深度可分离卷积 (3×3, d=2) ─┤
  └─→ 深度可分离卷积 (3×3, d=3) ─┤
                    │
                    ▼
                 拼接 → 空间注意力(Sigmoid) → 与原输入逐元素乘 → 残差加
```

### 操作

在 `modules.py` 中新建 `SEAM` 类。在 YAML 中，在 Pose Head 的输入前插入 `SEAM`：

```yaml
- [[15, 18, 21], 1, SEAM, [256]]  # 新增
- [[15, 18, 21], 1, Pose, [nc, kpt_shape]]  # 原来的 Pose Head 输入改为 SEAM 输出
```

**NPU 兼容**：所有算子（深度可分离卷积、膨胀卷积、Sigmoid、逐元素乘加）均为标准算子。

***

## 改造 4：Pose\_SA 轻量自注意力头

### 为什么

原始 Pose Head 用纯卷积，各关键点独立预测。高尔夫挥杆中左手腕和右肩膀是协调运动的，自注意力能捕获这种长距离依赖。

### 结构

```
输入特征
  │
  ├─→ 标准卷积分支 (CBS × 2) ──────────────────┐
  │                                              │
  └─→ 1×1 Conv (降维至64) → MHSA → 1×1 Conv (升维) ─┤
                                                    ▼
                                                 相加 → 输出
```

### 操作

新建 `PoseSAHead` 类，继承自原始 `Pose` 头。关键代码：

```python
class PoseSAHead(Pose):
    def __init__(self, ...):
        super().__init__(...)
        self.reduce = nn.Conv2d(256, 64, 1)  # 降维
        self.sa = nn.MultiheadAttention(64, 4, batch_first=True)
        self.expand = nn.Conv2d(64, 256, 1)  # 升维
    
    def forward(self, x):
        base = self.conv_branch(x)  # 原始卷积分支
        b, c, h, w = x.shape
        sa_in = self.reduce(x).flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        sa_out, _ = self.sa(sa_in, sa_in, sa_in)
        sa_out = sa_out.permute(0, 2, 1).view(b, 64, h, w)
        sa_out = self.expand(sa_out)
        return base + sa_out  # 残差融合
```

**NPU 兼容**：降维后 `MultiheadAttention` 等效于矩阵乘法，NPU 支持。若担心 Softmax 转换风险，可替换为 **7×7 大核深度卷积** 来近似空间交互。

***

## 改造 5：TSM 时序移位模块（训练时使用）

### 为什么

单帧推理时模型不知道前后帧信息。TSM 让训练时卷积能“看到”相邻帧特征，推理时折叠为标准卷积，零额外开销。

### 原理

```
训练时（输入 3 帧堆叠，9 通道）:
  通道 [0,1,2,3,4,5,6,7,8]
  移位 [←,←,←,→,→,→,0,0,0]
       前一帧        后一帧    当前帧
```

### 操作

在 `modules.py` 中新建 `TSMConv` 类，继承 `nn.Conv2d`。在 Backbone 浅层或 Neck 替换 1\~2 个普通卷积为 `TSMConv`。

训练时输入为 3 帧堆叠（通过 DataLoader 特殊处理），推理时重参数化为普通卷积。

***

## 改造 6：检测分支裁剪（砍 P3）

### 为什么

你的场景是近距离拍摄，人体在画面中很大。P3（80×80）检测头负责小目标，可以砍掉，减少约 25% 计算量。

### 操作

修改 YAML，删除 P3 相关连接，最终检测头只输入 P4 和 P5：

```yaml
# 删除所有 P3 的 Upsample 和 Concat
# 最后的 Pose 输入改为:
- [[12, 15], 1, Pose, [nc, kpt_shape]]  # 只保留 P4 和 P5
```

***

## 改造实施顺序（推荐）

| 顺序 | 改造                     | 风险 | 收益     |
| -- | ---------------------- | -- | ------ |
| 1  | 激活函数 → LeakyReLU       | 零  | NPU 兼容 |
| 2  | 砍掉 P3 检测分支             | 零  | 加速 25% |
| 3  | C2f\_Context (CGBlock) | 低  | 遮挡推理   |
| 4  | SEAM                   | 低  | 抗遮挡    |
| 5  | Pose\_SA 头             | 中  | 肢体协调   |
| 6  | TSM                    | 中  | 时序稳定   |

每完成一步，训练验证一轮，确保 mAP 不降再继续下一步。

***

**一句话总结**：这套改造从 NPU 兼容性出发，逐步引入全局上下文（CGBlock）、空间注意力（SEAM）、自注意力（Pose\_SA）和时序隐式建模（TSM），最终得到一个专为高尔夫挥杆优化、抗遮挡、低抖动、可完全部署在 RV1126B 上的 YOLOv8-pose 改进版。
