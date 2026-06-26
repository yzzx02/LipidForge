#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from build_training_split_v0_1 import (
    DEFAULT_OUT_DIR,
    GOLD_SPECTRA_PATH,
    GOLD_STRUCTURES_PATH,
    SPLIT_ORDER,
    build_split,
    read_jsonl,
)


EXPECTED_COUNTS = {
    "structures": 734,
    "connectivity": 731,
    "spectra": 1758,
}

REQUIRED_FILES = [
    "split_assignments.jsonl",
    "train_structures.jsonl",
    "validation_structures.jsonl",
    "test_structures.jsonl",
    "train_spectra.jsonl",
    "validation_spectra.jsonl",
    "test_spectra.jsonl",
    "split_summary.json",
    "split_counts_by_class.csv",
    "split_counts_by_linkage.csv",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise AssertionError(f"{name}: expected {expected!r}, got {actual!r}")


def load_split_rows(out_dir: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    structures = {split: read_jsonl(out_dir / f"{split}_structures.jsonl") for split in SPLIT_ORDER}
    spectra = {split: read_jsonl(out_dir / f"{split}_spectra.jsonl") for split in SPLIT_ORDER}
    return structures, spectra


def check_partition(out_dir: Path) -> dict[str, Any]:
    for name in REQUIRED_FILES:
        path = out_dir / name
        if not path.exists():
            raise AssertionError(f"Required split output missing: {path}")
    gold_structures = read_jsonl(GOLD_STRUCTURES_PATH)
    gold_spectra = read_jsonl(GOLD_SPECTRA_PATH)
    split_structures, split_spectra = load_split_rows(out_dir)
    gold_structure_ids = {row["structure_record_id"] for row in gold_structures}
    gold_connectivity_ids = {row["connectivity_id"] for row in gold_structures}
    gold_spectrum_ids = {row["spectrum_id"] for row in gold_spectra}
    assert_equal("gold structure count", len(gold_structure_ids), EXPECTED_COUNTS["structures"])
    assert_equal("gold connectivity count", len(gold_connectivity_ids), EXPECTED_COUNTS["connectivity"])
    assert_equal("gold spectrum count", len(gold_spectrum_ids), EXPECTED_COUNTS["spectra"])
    split_structure_ids: dict[str, set[str]] = {}
    split_connectivity_ids: dict[str, set[str]] = {}
    split_spectrum_ids: dict[str, set[str]] = {}
    counts = {}
    for split in SPLIT_ORDER:
        split_structure_ids[split] = {row["structure_record_id"] for row in split_structures[split]}
        split_connectivity_ids[split] = {row["connectivity_id"] for row in split_structures[split]}
        split_spectrum_ids[split] = {row["spectrum_id"] for row in split_spectra[split]}
        counts[split] = {
            "structure_count": len(split_structure_ids[split]),
            "connectivity_count": len(split_connectivity_ids[split]),
            "spectrum_count": len(split_spectrum_ids[split]),
        }
        for row in split_spectra[split]:
            if row["connectivity_id"] not in split_connectivity_ids[split]:
                raise AssertionError(f"Spectrum connectivity not present in same split: {row['spectrum_id']}")
            if row["structure_record_id"] not in split_structure_ids[split]:
                raise AssertionError(f"Spectrum structure not present in same split: {row['spectrum_id']}")
    all_structure_ids = set().union(*split_structure_ids.values())
    all_connectivity_ids = set().union(*split_connectivity_ids.values())
    all_spectrum_ids = set().union(*split_spectrum_ids.values())
    assert_equal("partitioned structures", all_structure_ids, gold_structure_ids)
    assert_equal("partitioned connectivity", all_connectivity_ids, gold_connectivity_ids)
    assert_equal("partitioned spectra", all_spectrum_ids, gold_spectrum_ids)
    overlaps = {}
    for left_index, left in enumerate(SPLIT_ORDER):
        for right in SPLIT_ORDER[left_index + 1 :]:
            overlaps[f"{left}_vs_{right}"] = {
                "structure": sorted(split_structure_ids[left] & split_structure_ids[right]),
                "connectivity": sorted(split_connectivity_ids[left] & split_connectivity_ids[right]),
                "spectrum": sorted(split_spectrum_ids[left] & split_spectrum_ids[right]),
            }
    if any(value for pair in overlaps.values() for value in pair.values()):
        raise AssertionError(f"Split leakage detected: {overlaps}")
    summary = read_json(out_dir / "split_summary.json")
    assert_equal("summary total structures", summary["total_structure_count"], EXPECTED_COUNTS["structures"])
    assert_equal("summary total connectivity", summary["total_connectivity_count"], EXPECTED_COUNTS["connectivity"])
    assert_equal("summary total spectra", summary["total_spectrum_count"], EXPECTED_COUNTS["spectra"])
    if not summary["leakage"]["no_connectivity_leakage"]:
        raise AssertionError("Summary reports connectivity leakage")
    return {
        "counts": counts,
        "overlap_counts": {
            name: {kind: len(items) for kind, items in values.items()} for name, values in overlaps.items()
        },
    }


def check_reproducibility() -> dict[str, str]:
    with tempfile.TemporaryDirectory(prefix="lipidforge_split_check_") as tmp:
        root = Path(tmp)
        out_a = root / "run_a"
        out_b = root / "run_b"
        build_split(out_dir=out_a)
        build_split(out_dir=out_b)
        hashes = {}
        for name in REQUIRED_FILES:
            hash_a = sha256_file(out_a / name)
            hash_b = sha256_file(out_b / name)
            if hash_a != hash_b:
                raise AssertionError(f"Reproducibility mismatch for {name}: {hash_a} != {hash_b}")
            hashes[name] = hash_a
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Gold v0.1 training split outputs.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    result = check_partition(args.out_dir)
    hashes = check_reproducibility()
    result["reproducibility_hashes"] = hashes
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
