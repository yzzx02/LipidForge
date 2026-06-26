# 数据源清单与使用策略

## MassBank-data

- 官方仓库：MassBank/MassBank-data
- 固定版本：2026.03
- 用途：优先获取有名称、有结构、有真实峰表的高置信实验 MS/MS。
- 注意：许可证可能随贡献者记录变化，必须逐条保留 `LICENSE`/`COPYRIGHT`。

## GNPS

优先库：
- PNNL Lipids positive / negative（GNPS 页面说明约 1790 lipids，多碰撞能）
- HCE Cell Lysate Lipids
- IOBA-NHC Lipids（约 200 MS/MS）
- All GNPS Library Spectra

GNPS 下载链接可能调整，因此采集器接受本地 MGF，不把不稳定 URL 硬编码为唯一入口。

## MassSpecGym 1.5

- Hugging Face 数据集提供约 23.1 万条谱图行。
- 字段包括峰、SMILES、formula、precursor m/z、adduct、instrument type 和 collision energy。
- 用途：高召回含磷脂质结构候选池。
- 限制：无化合物名称/脂质类别字段，因此默认标为 `P-lipid-unresolved`。

## MoNA

建议后续作为增量来源接入本地 MSP/MGF 导出。需要保留原始数据库标识，防止与 MassBank/GNPS 重复。

## LipidBlast

仅作为 `predicted_spectra` 独立数据层。不可与实验谱混合计算模型性能，也不用于实验谱测试集。

## LIPID MAPS LMSD

- 官方下载页：https://www.lipidmaps.org/databases/lmsd/download
- 用途：Phase 1/2 结构标准化后的参考结构匹配，不作为原始 MS/MS acquisition 来源。
- 许可证：`lipidmaps_ids_cc0.tsv` 按 CC0 参考使用；`LMSD_extended.sdf.zip` 按 LIPID MAPS 页面标注的 CC BY 4.0 来源保留。
- 注意：匹配结果写入 `data/structure_labeling/` 派生输出，不回写或覆盖 MassBank、MassSpecGym、GNPS、MoNA 等第三方谱图来源字段。
