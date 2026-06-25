from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import Dataset

from .labels import encode_polarity, encode_record_labels


def _extract_peak_pairs(record: dict[str, Any]) -> list[Any]:
    if record.get("peaks_raw") is not None:
        return list(record["peaks_raw"])
    if record.get("peaks") is not None:
        return list(record["peaks"])
    model_input = record.get("recommended_model_input") or {}
    if model_input.get("peaks") is not None:
        return list(model_input["peaks"])
    return []


def preprocess_peaks(
    record: dict[str, Any],
    max_peaks: int = 200,
    mz_scale: float = 1000.0,
) -> torch.Tensor:
    precursor_mz = float(
        (record.get("recommended_model_input") or {}).get(
            "precursor_mz",
            record.get("precursor_mz"),
        )
    )
    if not math.isfinite(precursor_mz):
        raise ValueError("precursor_mz must be finite")

    parsed: list[tuple[float, float]] = []
    for peak in _extract_peak_pairs(record):
        try:
            fragment_mz = float(peak[0])
            intensity = float(peak[1])
        except (TypeError, ValueError, IndexError):
            continue

        if not math.isfinite(fragment_mz) or not math.isfinite(intensity):
            continue
        if intensity < 0:
            continue
        parsed.append((fragment_mz, intensity))

    if not parsed:
        raise ValueError("No valid MS/MS peaks remain after filtering")

    max_intensity = max(intensity for _, intensity in parsed)
    if max_intensity <= 0:
        raise ValueError("At least one peak intensity must be positive")

    normalized = [
        (fragment_mz, intensity / max_intensity)
        for fragment_mz, intensity in parsed
    ]
    if len(normalized) > max_peaks:
        normalized = sorted(
            normalized,
            key=lambda item: item[1],
            reverse=True,
        )[:max_peaks]

    normalized = sorted(normalized, key=lambda item: item[0])
    features = [
        [
            fragment_mz / mz_scale,
            math.sqrt(relative_intensity),
            (precursor_mz - fragment_mz) / mz_scale,
        ]
        for fragment_mz, relative_intensity in normalized
    ]
    return torch.tensor(features, dtype=torch.float32)


def featurize_record(
    record: dict[str, Any],
    max_peaks: int = 200,
    mz_scale: float = 1000.0,
    require_labels: bool = True,
) -> dict[str, Any]:
    model_input = record.get("recommended_model_input") or {}
    polarity = model_input.get("polarity", record.get("polarity"))
    precursor_mz = model_input.get("precursor_mz", record.get("precursor_mz"))

    sample: dict[str, Any] = {
        "peak_features": preprocess_peaks(record, max_peaks=max_peaks, mz_scale=mz_scale),
        "precursor_mz": torch.tensor(float(precursor_mz), dtype=torch.float32),
        "polarity": torch.tensor(encode_polarity(polarity), dtype=torch.long),
        "metadata": {
            "spectrum_id": record.get("spectrum_id") or record.get("structure_id"),
            "display_name": record.get("display_name"),
        },
    }
    if require_labels:
        sample.update(encode_record_labels(record))
    return sample


class LipidSpectrumDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        max_peaks: int = 200,
        mz_scale: float = 1000.0,
        require_labels: bool = True,
        filter_unusable: bool = True,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.max_peaks = max_peaks
        self.mz_scale = mz_scale
        self.require_labels = require_labels
        self.records: list[dict[str, Any]] = []

        with self.jsonl_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if filter_unusable and record.get("usable_for_pilot_training") is False:
                    continue
                try:
                    featurize_record(
                        record,
                        max_peaks=max_peaks,
                        mz_scale=mz_scale,
                        require_labels=require_labels,
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"{self.jsonl_path}:{line_number}: {exc}"
                    ) from exc
                self.records.append(record)

        if not self.records:
            raise ValueError(f"No usable records found in {self.jsonl_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return featurize_record(
            self.records[index],
            max_peaks=self.max_peaks,
            mz_scale=self.mz_scale,
            require_labels=self.require_labels,
        )


def collate_spectra(
    samples: list[dict[str, Any]],
    max_peaks: int = 200,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty batch")

    batch_size = len(samples)
    peak_features = torch.zeros(batch_size, max_peaks, 3, dtype=torch.float32)
    peak_padding_mask = torch.ones(batch_size, max_peaks, dtype=torch.bool)
    peak_counts = torch.zeros(batch_size, dtype=torch.long)

    for row, sample in enumerate(samples):
        features = sample["peak_features"][:max_peaks]
        count = features.shape[0]
        peak_features[row, :count] = features
        peak_padding_mask[row, :count] = False
        peak_counts[row] = count

    batch: dict[str, Any] = {
        "peak_features": peak_features,
        "peak_padding_mask": peak_padding_mask,
        "peak_counts": peak_counts,
        "precursor_mz": torch.stack([sample["precursor_mz"] for sample in samples]),
        "polarity": torch.stack([sample["polarity"] for sample in samples]),
        "metadata": [sample.get("metadata", {}) for sample in samples],
    }

    label_keys = [
        "headgroup_label",
        "chain_count_label",
        "chain_present",
        "chain_carbon_labels",
        "chain_double_bond_labels",
        "chain_linkage_labels",
        "chain_mask",
    ]
    for key in label_keys:
        if key in samples[0]:
            batch[key] = torch.stack([sample[key] for sample in samples])

    return batch


def make_collate_fn(max_peaks: int = 200) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    def _collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
        return collate_spectra(samples, max_peaks=max_peaks)

    return _collate
