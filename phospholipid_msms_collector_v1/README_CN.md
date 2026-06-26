# LipidForge 磷脂 MS/MS 批量采集包 v1

这个包的目标不是只收集 PC/PE/PG/PI/PS/PA 和 lyso 类，而是先建立**广覆盖磷脂候选池**，再逐步提高标签精度。

## 已纳入的类别范围

### 甘油磷脂
PA/LPA、PC/LPC、PE/LPE、PG/LPG、PI/LPI、PS/LPS、CL/MLCL/DLCL、BMP/LBMP、PIP/PIP2/PIP3、PGP、CDP-DAG、NAPE/LNAPE、MMPE/DMPE、PEt/PMe/PPr/PBut、PT、PHS、磷脂酰糖类等。

### 醚脂与质醚脂
不把它们当独立头基，而通过 `linkage_modifications` 标记 `ether`、`plasmalogen`。

### 鞘磷脂与含磷鞘脂
SM/LysoSM、CerP、S1P/dhS1P、CerPE、IPC/MIPC/MIP2C。

### 微生物与古菌特殊磷脂
PIM、archaetidyl/caldarchaetidyl 类、phosphonolipid 候选。

### 未解析候选
结构中含 P、具备明显脂质元素组成，但没有可靠类别名称的记录放入：

`phosphorus_lipid_candidates_unresolved.jsonl`

不要为了扩大类别数强行把它们塞进 PC/PE 等类别。

## 数据来源

1. **MassBank-data 2026.03**：官方公开 MassBank 文本记录，全库下载后筛选真实 MS2。
2. **MassSpecGym 1.5**：大规模带 SMILES、formula、m/z/intensity 的实验谱集合。只做含磷脂质结构候选筛选；因没有名称字段，不强行赋头基标签。
3. **GNPS / MoNA / 其他 MGF**：通过 `--local-mgf` 接入。建议从 GNPS 公共库导出：
   - PNNL Lipids positive / negative
   - HCE Cell Lysate Lipids
   - IOBA-NHC Lipids
   - All GNPS Library Spectra
4. **LipidBlast**：属于预测谱，必须单独保存，不得混进 experimental=true 的训练集。

## Windows 运行

```powershell
cd phospholipid_msms_collector_v1
.\run_collect.ps1
```

自定义项目、输出和下载目录：

```powershell
.\run_collect.ps1 `
  -ProjectRoot "D:\projects\LipidForge" `
  -OutputDir "D:\lipid_data\expanded_phospholipids" `
  -WorkDir "D:\lipid_data\downloads"
```

加入本地 GNPS MGF：

```powershell
python scripts/collect_phospholipid_msms.py `
  --out "<PROJECT_ROOT>\data\expanded_phospholipids" `
  --work "<DOWNLOAD_DIR>" `
  --local-mgf "<DOWNLOAD_DIR>\PNNL-LIPIDS-POSITIVE.mgf" `
  --local-mgf "<DOWNLOAD_DIR>\PNNL-LIPIDS-NEGATIVE.mgf"
```

## 输出

- `phospholipid_msms_all.jsonl`：全部高置信类别 + 未解析含磷脂质候选
- `phospholipid_msms_strict.jsonl`：名称能可靠归类的记录
- `phosphorus_lipid_candidates_unresolved.jsonl`：含磷脂质结构候选，等待图结构分类
- `summary.json`
- `class_counts.csv`

## v2 结构标识输出

使用 `--schema-version v2` 时，采集器会保留全部 acquisition 记录，并额外输出结构标准化字段、`peak_identity_hash`、`acquisition_metadata_hash`、`acquisition_record_hash` 与 `duplicate_relation`。

- `removable_exact_duplicate` 只表示同一 `acquisition_record_hash` 的重复记录，可由可选的 `phospholipid_msms_acquisition_dedup_v2.jsonl` 视图去除。
- `same_source_acquisition_duplicate_candidate` 表示同一来源、同一峰身份和同一采集元数据，但不同 `source_record_id` 的候选重复；它不是安全删除标签。
- 结构标签阶段只把 LIPID MAPS 匹配结果写入 `data/structure_labeling/` 派生目录，不回写第三方原始 acquisition 记录。

## 典型甘油磷脂结构解析器 v0.1

结构解析器 v0.1 只根据分子图/SMILES 解析典型甘油磷脂，不读取 `lipid_class`、名称、abbreviation 或来源标签来决定类别。当前支持 PA/PC/PE/PG/PI/PS 及对应 lyso 类，输出头基、glycerol backbone、链、free hydroxyl、linkage multiset、分区和重构检查。

`ester`、`alkyl_ether` 和 `vinyl_ether` 是链连接属性，不是独立头基；lyso 类由链数量派生。范围外结构会明确返回 unsupported 状态，例如 sphingoid backbone、SM/S1P/CerP、NAPE、BMP/CL、多磷拓扑、C-methyl/额外骨架取代以及复杂 PIP/IPC/古菌脂质/膦脂。

Phase 3B v0.1 验收口径固定为：strict structures 814，Phase 3B eligible 785，图解析并 reconstruction exact 为 776，其中 734 条与 strict 标签一致并作为 gold v0.1，41 条是 parser-derived high-confidence label-disagreement candidates，需要后续人工审核，不能自动回写 strict 数据。派生报告、图片、JSONL/CSV 结果和本地数据不进入 Git。

## 重要原则

- v2 默认保留全部 acquisition 记录。
- 峰表相同的记录会进入 duplicate group，但不会仅因峰表相同而删除。
- 不同碰撞能、仪器、来源或 `source_record_id` 的记录必须保留其 provenance。
- `same_source_acquisition_duplicate_candidate` 只是候选重复，不是安全删除标签。
- 只有同一 `acquisition_record_hash` 的 `removable_exact_duplicate` 才能在可选 acquisition-dedup 视图中去除。
- v1 的 peak-only dedup 输出仅保留作历史对比，不应作为 v2 数据生成策略。
- 训练/验证/测试必须按分子结构分组，不能随机按谱图行切分。
- `MassSpecGym` 的未解析候选不能直接作为 PC/PE/PG 等监督标签。
- 每条记录保留来源和许可证；未经许可核对，不把大规模第三方谱图提交到公开 GitHub。
- 当前脚本是广召回第一版。正式训练前还需做结构合法性、链解析、头基图规则和跨源去重。
