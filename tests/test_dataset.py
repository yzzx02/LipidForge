from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from lipidforge.dataset import (
    LipidSpectrumDataset,
    collate_spectra,
    featurize_record,
    preprocess_peaks,
)
from lipidforge.labels import CARBON_TO_INDEX, LINKAGE_TO_INDEX, sort_chains


def _record(**overrides):
    record = {
        "spectrum_id": "demo",
        "prototype_headgroup": "PC",
        "polarity": "negative",
        "precursor_mz": 1000.0,
        "chains": [
            {"carbon": 20, "double_bonds": 4},
            {"carbon": 18, "double_bonds": 1},
        ],
        "chain_linkage_summary": "ester",
        "peaks_raw": [[300.0, 9.0], [100.0, 1.0], [200.0, 4.0]],
        "usable_for_pilot_training": True,
    }
    record.update(overrides)
    return record


def test_preprocess_peaks_filters_normalizes_truncates_and_sorts():
    record = _record(
        peaks_raw=[
            [300.0, 9.0],
            [100.0, 1.0],
            [200.0, 4.0],
            [400.0, -1.0],
            ["bad", 2.0],
            [500.0, float("nan")],
        ]
    )
    features = preprocess_peaks(record, max_peaks=2, mz_scale=1000.0)

    assert features.shape == (2, 3)
    assert torch.allclose(features[:, 0], torch.tensor([0.2, 0.3]))
    assert math.isclose(float(features[0, 1]), math.sqrt(4.0 / 9.0), rel_tol=1e-6)
    assert math.isclose(float(features[1, 1]), 1.0, rel_tol=1e-6)
    assert torch.allclose(features[:, 2], torch.tensor([0.8, 0.7]))


def test_empty_or_zero_peak_spectrum_raises():
    with pytest.raises(ValueError, match="No valid"):
        preprocess_peaks(_record(peaks_raw=[]))
    with pytest.raises(ValueError, match="positive"):
        preprocess_peaks(_record(peaks_raw=[[100.0, 0.0]]))


def test_collate_pads_and_masks_correctly():
    one = featurize_record(_record(peaks_raw=[[100.0, 1.0], [200.0, 2.0]]))
    two = featurize_record(_record(peaks_raw=[[150.0, 3.0]]))
    batch = collate_spectra([one, two], max_peaks=4)

    assert batch["peak_features"].shape == (2, 4, 3)
    assert batch["peak_counts"].tolist() == [2, 1]
    assert batch["peak_padding_mask"].tolist() == [
        [False, False, True, True],
        [False, True, True, True],
    ]
    assert torch.count_nonzero(batch["peak_features"][1, 1:]) == 0


def test_polarity_chain_sort_and_lyso_mask():
    sample = featurize_record(
        _record(
            polarity="positive",
            chains=[{"carbon": 16, "double_bonds": 0}],
            chain_linkage_summary="vinyl_ether",
        )
    )

    assert int(sample["polarity"]) == 1
    assert int(sample["chain_count_label"]) == 0
    assert sample["chain_present"].tolist() == [1.0, 0.0]
    assert sample["chain_mask"].tolist() == [True, False]
    assert int(sample["chain_carbon_labels"][0]) == CARBON_TO_INDEX[16]
    assert int(sample["chain_linkage_labels"][0]) == LINKAGE_TO_INDEX["vinyl_ether"]

    sorted_chains = sort_chains(
        [
            {"carbon": 20, "double_bonds": 4},
            {"carbon": 18, "double_bonds": 1},
        ]
    )
    assert [(chain.carbon, chain.double_bonds) for chain in sorted_chains] == [
        (18, 1),
        (20, 4),
    ]


def test_jsonl_dataset_reads_records(tmp_path: Path):
    path = tmp_path / "data.jsonl"
    rows = [_record(spectrum_id="a"), _record(spectrum_id="b")]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    dataset = LipidSpectrumDataset(path)
    assert len(dataset) == 2
    assert dataset[0]["peak_features"].shape[1] == 3
