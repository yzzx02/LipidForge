# LipidForge 第一版模型设计

## 1. 输入字段

第一版模型只读取 `polarity`、`precursor_mz`、MS/MS 峰的 `fragment_mz` 和 `intensity`。仪器、碰撞能、加合物、保留时间、样本来源等字段只保留在元数据中用于质控，不进入网络。

当前仓库提交的小型 pilot 文件为 `data/pilot/experimental_ms2_pilot.jsonl`，共有 6 条真实实验谱，覆盖 PA、PC、PE、PG、PI、PS 各 1 条，且全部为负离子模式。LIPID MAPS structure seed 文件不随源码提交，且本阶段不能当作谱图训练数据。

## 2. 峰预处理

`peaks_raw` 和顶层 `peaks` 被视为原始强度，按以下顺序处理：

1. 删除无法解析的峰；
2. 删除非有限数值；
3. 删除负强度；
4. 强度除以该谱最大强度；
5. 对归一化强度做平方根变换；
6. 峰数超过 `max_peaks=200` 时，保留强度最高的 200 个峰；
7. 保留后按 m/z 从小到大排序；
8. batch 中不足 200 个峰的位置补零；
9. 生成 `peak_padding_mask`，其中 `True` 表示 padding 位置。

`recommended_model_input.peaks` 被视为已经完成 sqrt-normalized 的 `(fragment_mz, sqrt_intensity)`，不会再次归一化或开根号。

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
chain_carbon_labels: [B, 2]
chain_double_bond_labels: [B, 2]
chain_linkage_labels: [B, 2]
chain_mask: [B, 2]
chain_linkage_mask: [B, 2]
```

`polarity` 编码为 `negative=0`、`positive=1`。`chain_mask=True` 表示该链槽位存在，控制碳数和双键数 loss。`chain_linkage_mask=True` 表示该链槽位的 linkage 可以监督，单独控制 linkage loss。

linkage 标签规则：

- 逐链 `chain["linkage"]` 明确时，计算该槽位 linkage loss；
- 单链脂质缺少逐链 linkage，但记录级 `chain_linkage_summary` 明确时，可以推断给唯一链槽位并计算 linkage loss；
- 双链脂质只有记录级 `chain_linkage_summary` 时，不能映射到具体槽位，`chain_linkage_mask=False`；
- 未知 linkage 不默认填充为 `ester`，只保留占位 label 且不计算 linkage loss。

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
chain_carbon_logits: [B, 2, 39]        # carbon 2-40
chain_double_bond_logits: [B, 2, 13]   # double bonds 0-12
chain_linkage_logits: [B, 2, 3]        # ester/ether/vinyl_ether
```

头基类别为 PA、PC、PE、PG、PI、PS。链数量类别为 1 条链和 2 条链。第一版不再单独预测 `chain_present`，推理时根据 `chain_count_logits` 决定有效链槽位数量。

## 6. 损失函数

```text
headgroup: CrossEntropyLoss
chain_count: CrossEntropyLoss
chain_carbon: masked CrossEntropyLoss, mask=chain_mask
chain_double_bond: masked CrossEntropyLoss, mask=chain_mask
chain_linkage: masked CrossEntropyLoss, mask=chain_linkage_mask
```

总损失为以上各项直接求和。未知 linkage 或无法映射到具体链槽位的记录不会对 `chain_linkage_logits` 产生有效梯度。

## 7. 链标签排序

第一版不预测 sn-1/sn-2 位置。链标签统一排序：

```text
1. 碳数升序
2. 双键数升序
3. 连接类型升序：ester, ether, vinyl_ether
```

如果 linkage 未知，则只用于占位排序，不作为真实类别监督。例如 `20:4 / 18:1` 会统一成 `18:1 / 20:4`。

## 8. Checkpoint 配置

正式训练保存 checkpoint 时包含：

```text
model_state_dict
model_config
preprocessing_config
label_schema
```

`predict.py` 优先使用 checkpoint 内的模型和预处理配置。如果外部 config 与 checkpoint 的架构或预处理字段冲突，推理会报错，避免用错误网络形状或错误峰预处理加载权重。

## 9. 当前数据限制

当前只有 6 条真实实验 MS/MS 谱，不能报告模型性能或科学结论。该数据只用于验证 JSONL 读取、峰预处理、batch padding、forward、loss、backward 和小样本过拟合。

当前 6 条实验谱全部是负离子模式，正离子模式路径只能通过代码兼容性测试，不能代表真实分布。

低置信度 `Unknown-P-headgroup(...)` 只是推理显示规则，不是严格的未知类别检测，也不能证明谱图一定含磷酸。真正开放集识别留到后续版本。

## 10. 电脑分工

办公电脑用于写代码、清洗数据、运行单元测试、CPU/GPU forward、少量 batch smoke test、loss backward 验证和小样本过拟合。不要假定具体 CUDA 版本，检测到 GPU 时统一使用 `torch.device("cuda")`。

RX 9070 XT 训练电脑使用 WSL + ROCm。PyTorch ROCm 后端仍通过 `torch.cuda` 和 `torch.device("cuda")` 使用，不写 `rocm` 或 `hip` device。正式训练第一轮使用 FP32，先确认 forward、backward、optimizer step、validation、checkpoint、无 NaN、无 OOM，再测试 FP16 AMP。
