# LipidForge 第一版模型设计

## 1. 输入字段

第一版模型只读取 `polarity`、`precursor_mz`、MS/MS 峰的 `fragment_mz` 和 `intensity`。仪器、碰撞能、加合物、保留时间、样本来源等字段只保留在元数据中用于质控，不进入网络。

当前可用实验谱来自 `glycerophospholipid_pilot_v1/experimental_ms2_pilot.jsonl`，共有 6 条，覆盖 PA、PC、PE、PG、PI、PS 各 1 条，且全部为负离子模式。LIPID MAPS seed 文件只有结构标签，没有可用峰表。

## 2. 峰预处理

每条谱按以下顺序处理峰：

1. 删除无法解析的峰；
2. 删除非有限数值；
3. 删除负强度；
4. 强度除以该谱最大强度；
5. 对归一化强度做平方根变换；
6. 峰数超过 `max_peaks=200` 时，保留强度最高的 200 个峰；
7. 保留后按 m/z 从小到大排序；
8. batch 中不足 200 个峰的位置补零；
9. 生成 `peak_padding_mask`，其中 `True` 表示 padding 位置。

每个有效峰的 3 维特征为：

```text
[
  fragment_mz / mz_scale,
  sqrt(normalized_intensity),
  (precursor_mz - fragment_mz) / mz_scale
]
```

默认 `mz_scale=1000.0`。

## 3. Batch 张量形状

设 batch size 为 `B`，最大峰数为 `P=200`：

```text
peak_features: [B, P, 3]
peak_padding_mask: [B, P]
precursor_mz: [B]
polarity: [B]
headgroup_label: [B]
chain_count_label: [B]
chain_present: [B, 2]
chain_carbon_labels: [B, 2]
chain_double_bond_labels: [B, 2]
chain_linkage_labels: [B, 2]
chain_mask: [B, 2]
```

`polarity` 编码为 `negative=0`、`positive=1`。lyso 脂质的第二条链 `chain_mask=False`，其碳数、双键数和连接类型不参与损失。

## 4. Transformer 网络结构

PeakEncoder：

```text
Linear(3, 128)
GELU
Linear(128, 128)
LayerNorm(128)
```

CLS token：

```text
learnable_cls + precursor_embedding + polarity_embedding
```

其中 `precursor_embedding` 为：

```text
precursor_mz / 1000
Linear(1, 128)
GELU
Linear(128, 128)
```

Transformer Encoder 固定为 4 层：

```text
d_model=128
nhead=4
dim_feedforward=256
dropout=0.10
activation=gelu
batch_first=True
norm_first=True
enable_nested_tensor=False
```

序列形状为 `[B, 201, 128]`，第一个 token 是 CLS，后 200 个 token 是峰。`src_key_padding_mask` 形状为 `[B, 201]`，CLS 的 mask 恒为 `False`。

## 5. 输出头

最终使用 encoder 输出的 CLS 向量表示整张谱图：

```text
headgroup_logits: [B, 6]
chain_count_logits: [B, 2]
chain_present_logits: [B, 2]
chain_carbon_logits: [B, 2, 39]        # carbon 2-40
chain_double_bond_logits: [B, 2, 13]   # double bonds 0-12
chain_linkage_logits: [B, 2, 3]        # ester/ether/vinyl_ether
```

头基类别为 PA、PC、PE、PG、PI、PS。链数量类别为 1 条链和 2 条链。

## 6. 损失函数

```text
headgroup: CrossEntropyLoss
chain_count: CrossEntropyLoss
chain_present: BCEWithLogitsLoss
chain_carbon: masked CrossEntropyLoss
chain_double_bond: masked CrossEntropyLoss
chain_linkage: masked CrossEntropyLoss
```

总损失为以上各项直接求和。链属性损失只在 `chain_mask=True` 的槽位计算。

## 7. 链标签排序

第一版不预测 sn-1/sn-2 位置。链标签统一排序：

```text
1. 碳数升序
2. 双键数升序
3. 连接类型升序：ester, ether, vinyl_ether
```

例如 `20:4 / 18:1` 会统一成 `18:1 / 20:4`。

## 8. 当前数据限制

当前只有 6 条真实实验 MS/MS 谱，不能报告模型性能或科学结论。该数据只用于验证 JSONL 读取、峰预处理、batch padding、forward、loss、backward 和小样本过拟合。结构 seed 文件可用于理解标签空间，但不能当作谱图训练数据。

当前 6 条实验谱全部是负离子模式，正离子模式路径只能通过代码兼容性测试，不能代表真实分布。

低置信度 `Unknown-P-headgroup(...)` 只是推理显示规则，不是严格的未知类别检测，也不能证明谱图一定含磷酸。真正开放集识别留到后续版本。

## 9. 电脑分工

办公电脑用于写代码、清洗数据、运行单元测试、CPU/GPU forward、少量 batch smoke test、loss backward 验证和小样本过拟合。不要假定具体 CUDA 版本，检测到 GPU 时统一使用 `torch.device("cuda")`。

RX 9070 XT 训练电脑使用 WSL + ROCm。PyTorch ROCm 后端仍通过 `torch.cuda` 和 `torch.device("cuda")` 使用，不写 `rocm` 或 `hip` device。正式训练第一轮使用 FP32，先确认 forward、backward、optimizer step、validation、checkpoint、无 NaN、无 OOM，再测试 FP16 AMP。
