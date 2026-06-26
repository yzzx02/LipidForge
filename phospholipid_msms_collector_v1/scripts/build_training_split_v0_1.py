#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLD_DIR = PROJECT_ROOT / "data" / "structure_labeling" / "gold_v0_1"
GOLD_STRUCTURES_PATH = GOLD_DIR / "gold_structures_v0_1.jsonl"
GOLD_SPECTRA_PATH = GOLD_DIR / "gold_spectra_v0_1.jsonl"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "structure_labeling" / "training_split_v0_1"

SEED = 20260626
SPLIT_RATIOS = {
    "train": 0.8,
    "validation": 0.1,
    "test": 0.1,
}
SPLIT_ORDER = ["train", "validation", "test"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stable_key(value: str) -> str:
    return hashlib.sha256(f"{SEED}:{value}".encode("utf-8")).hexdigest()


def normalize_linkage(pattern: Any) -> str:
    if isinstance(pattern, list):
        values = [str(item) for item in pattern]
        if len(values) == 1:
            return f"single_chain_{values[0]}"
        order = {"alkyl_ether": 0, "vinyl_ether": 0, "ester": 1}
        return "/".join(sorted(values, key=lambda item: (order.get(item, 2), item)))
    return str(pattern or "")


def require_inputs(structures_path: Path, spectra_path: Path) -> None:
    for path in [structures_path, spectra_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required input file not found: {path}")


def target_group_counts(total: int) -> dict[str, int]:
    train = round(total * SPLIT_RATIOS["train"])
    validation = round(total * SPLIT_RATIOS["validation"])
    test = total - train - validation
    return {"train": train, "validation": validation, "test": test}


def group_inputs(structures: list[dict[str, Any]], spectra: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for row in structures:
        conn = row.get("split_group_id") or row.get("connectivity_id")
        if not conn:
            raise ValueError(f"Structure lacks split_group_id/connectivity_id: {row.get('structure_record_id')}")
        groups.setdefault(conn, {"connectivity_id": conn, "structures": [], "spectra": []})["structures"].append(row)
    known_structures = {row["structure_record_id"] for row in structures}
    for row in spectra:
        conn = row.get("split_group_id") or row.get("connectivity_id")
        sid = row.get("structure_record_id")
        if conn not in groups:
            raise ValueError(f"Spectrum references unknown connectivity_id: {conn}")
        if sid not in known_structures:
            raise ValueError(f"Spectrum references unknown structure_record_id: {sid}")
        groups[conn]["spectra"].append(row)
    for group in groups.values():
        group["structures"].sort(key=lambda item: item["structure_record_id"])
        group["spectra"].sort(key=lambda item: (str(item.get("spectrum_id") or ""), str(item.get("source_record_id") or "")))
        classes = sorted({str(item.get("gold_lipid_class") or "") for item in group["structures"]})
        linkages = sorted({normalize_linkage(item.get("linkage_pattern")) for item in group["structures"]})
        polarities = sorted({str(item.get("polarity") or "") for item in group["spectra"]})
        group["gold_lipid_class"] = "+".join(classes)
        group["linkage_pattern"] = "+".join(linkages)
        group["polarity"] = "+".join(polarities)
        group["structure_record_ids"] = [item["structure_record_id"] for item in group["structures"]]
        group["spectrum_ids"] = [item["spectrum_id"] for item in group["spectra"]]
        group["features"] = [
            f"class={group['gold_lipid_class']}",
            f"linkage={group['linkage_pattern']}",
            f"polarity={group['polarity']}",
        ]
    return groups


def split_remainder(total: int, split: str) -> float:
    return (total * SPLIT_RATIOS[split]) % 1


def allocate_stratum_targets(strata: dict[tuple[str, str, str], list[dict[str, Any]]]) -> dict[tuple[str, str, str], dict[str, int]]:
    targets: dict[tuple[str, str, str], dict[str, int]] = {}
    for key, groups in strata.items():
        total = len(groups)
        validation = int(total * SPLIT_RATIOS["validation"]) if total >= 10 else 0
        test = int(total * SPLIT_RATIOS["test"]) if total >= 10 else 0
        targets[key] = {"train": total - validation - test, "validation": validation, "test": test}
    desired = target_group_counts(sum(len(groups) for groups in strata.values()))
    for split in ["validation", "test"]:
        current = sum(value[split] for value in targets.values())
        if current < desired[split]:
            candidates = sorted(
                strata,
                key=lambda key: (-split_remainder(len(strata[key]), split), -len(strata[key]), str(key)),
            )
            while current < desired[split]:
                changed = False
                for key in candidates:
                    if targets[key]["train"] <= 0:
                        continue
                    targets[key][split] += 1
                    targets[key]["train"] -= 1
                    current += 1
                    changed = True
                    if current == desired[split]:
                        break
                if not changed:
                    raise ValueError(f"Unable to allocate requested {split} groups")
        elif current > desired[split]:
            candidates = sorted(
                strata,
                key=lambda key: (split_remainder(len(strata[key]), split), len(strata[key]), str(key)),
            )
            while current > desired[split]:
                changed = False
                for key in candidates:
                    if targets[key][split] <= 0:
                        continue
                    targets[key][split] -= 1
                    targets[key]["train"] += 1
                    current -= 1
                    changed = True
                    if current == desired[split]:
                        break
                if not changed:
                    raise ValueError(f"Unable to reduce {split} groups")
    return targets


def assign_groups(groups: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    strata: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for group in groups.values():
        strata[(group["gold_lipid_class"], group["linkage_pattern"], group["polarity"])].append(group)
    targets = allocate_stratum_targets(strata)
    assignments = []
    for key in sorted(strata):
        ordered_groups = sorted(strata[key], key=lambda group: (stable_key(group["connectivity_id"]), group["connectivity_id"]))
        split_sequence = (
            ["validation"] * targets[key]["validation"]
            + ["test"] * targets[key]["test"]
            + ["train"] * targets[key]["train"]
        )
        for group, split in zip(ordered_groups, split_sequence, strict=True):
            assignments.append(
                {
                    "split_version": "0.1",
                    "seed": SEED,
                    "split": split,
                    "split_group_id": group["connectivity_id"],
                    "connectivity_id": group["connectivity_id"],
                    "gold_lipid_class": group["gold_lipid_class"],
                    "linkage_pattern": group["linkage_pattern"],
                    "polarity": group["polarity"],
                    "structure_count": len(group["structures"]),
                    "spectrum_count": len(group["spectra"]),
                    "structure_record_ids": group["structure_record_ids"],
                    "spectrum_ids": group["spectrum_ids"],
                }
            )
    return sorted(assignments, key=lambda row: (row["split"], row["connectivity_id"]))


def rows_by_split(rows: list[dict[str, Any]], assignment_by_conn: dict[str, str]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLIT_ORDER}
    for row in rows:
        conn = row.get("split_group_id") or row.get("connectivity_id")
        split = assignment_by_conn[conn]
        enriched = dict(row)
        enriched["split"] = split
        enriched["split_version"] = "0.1"
        enriched["split_seed"] = SEED
        out[split].append(enriched)
    for split in SPLIT_ORDER:
        out[split].sort(key=lambda item: (str(item.get("connectivity_id") or ""), str(item.get("structure_record_id") or ""), str(item.get("spectrum_id") or "")))
    return out


def count_table(
    split_structures: dict[str, list[dict[str, Any]]],
    split_spectra: dict[str, list[dict[str, Any]]],
    field: str,
) -> list[dict[str, Any]]:
    rows = []
    values = sorted({str(item.get(field) if field != "linkage_bucket" else normalize_linkage(item.get("linkage_pattern"))) for split in SPLIT_ORDER for item in split_structures[split]})
    for split in SPLIT_ORDER:
        structures = split_structures[split]
        spectra = split_spectra[split]
        for value in values:
            if field == "linkage_bucket":
                struct_subset = [row for row in structures if normalize_linkage(row.get("linkage_pattern")) == value]
                spec_subset = [row for row in spectra if normalize_linkage(row.get("linkage_pattern")) == value]
                key = "linkage_bucket"
            else:
                struct_subset = [row for row in structures if str(row.get(field)) == value]
                spec_subset = [row for row in spectra if str(row.get(field)) == value]
                key = field
            if not struct_subset and not spec_subset:
                continue
            rows.append(
                {
                    "split": split,
                    key: value,
                    "structure_count": len({row["structure_record_id"] for row in struct_subset}),
                    "connectivity_count": len({row["connectivity_id"] for row in struct_subset}),
                    "spectrum_count": len(spec_subset),
                }
            )
    return rows


def leakage_summary(assignments: list[dict[str, Any]]) -> dict[str, Any]:
    by_split = {split: {row["connectivity_id"] for row in assignments if row["split"] == split} for split in SPLIT_ORDER}
    overlaps = {}
    for left_index, left in enumerate(SPLIT_ORDER):
        for right in SPLIT_ORDER[left_index + 1 :]:
            overlaps[f"{left}_vs_{right}"] = sorted(by_split[left] & by_split[right])
    return {
        "connectivity_overlap_counts": {key: len(value) for key, value in overlaps.items()},
        "connectivity_overlaps": overlaps,
        "no_connectivity_leakage": all(not value for value in overlaps.values()),
    }


def build_split(
    structures_path: Path = GOLD_STRUCTURES_PATH,
    spectra_path: Path = GOLD_SPECTRA_PATH,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    require_inputs(structures_path, spectra_path)
    structures = read_jsonl(structures_path)
    spectra = read_jsonl(spectra_path)
    groups = group_inputs(structures, spectra)
    assignments = assign_groups(groups)
    assignment_by_conn = {row["connectivity_id"]: row["split"] for row in assignments}
    split_structures = rows_by_split(structures, assignment_by_conn)
    split_spectra = rows_by_split(spectra, assignment_by_conn)
    staging = out_dir.parent / f".{out_dir.name}.tmp"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=False)
    write_jsonl(staging / "split_assignments.jsonl", assignments)
    for split in SPLIT_ORDER:
        write_jsonl(staging / f"{split}_structures.jsonl", split_structures[split])
        write_jsonl(staging / f"{split}_spectra.jsonl", split_spectra[split])
    class_rows = count_table(split_structures, split_spectra, "gold_lipid_class")
    linkage_rows = count_table(split_structures, split_spectra, "linkage_bucket")
    write_csv(
        staging / "split_counts_by_class.csv",
        class_rows,
        ["split", "gold_lipid_class", "structure_count", "connectivity_count", "spectrum_count"],
    )
    write_csv(
        staging / "split_counts_by_linkage.csv",
        linkage_rows,
        ["split", "linkage_bucket", "structure_count", "connectivity_count", "spectrum_count"],
    )
    counts = {
        split: {
            "structure_count": len(split_structures[split]),
            "connectivity_count": len({row["connectivity_id"] for row in split_structures[split]}),
            "spectrum_count": len(split_spectra[split]),
        }
        for split in SPLIT_ORDER
    }
    summary = {
        "split_version": "0.1",
        "seed": SEED,
        "ratios": SPLIT_RATIOS,
        "split_unit": "connectivity_id",
        "input_files": {
            "gold_structures": str(structures_path),
            "gold_spectra": str(spectra_path),
        },
        "total_structure_count": len(structures),
        "total_connectivity_count": len(groups),
        "total_spectrum_count": len(spectra),
        "split_counts": counts,
        "leakage": leakage_summary(assignments),
        "stratification_fields": ["gold_lipid_class", "linkage_pattern", "polarity"],
        "note": "Connectivity non-leakage has priority over exact stratification.",
    }
    write_json(staging / "split_summary.json", summary)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    staging.rename(out_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Gold v0.1 connectivity-level train/validation/test splits.")
    parser.add_argument("--structures", type=Path, default=GOLD_STRUCTURES_PATH)
    parser.add_argument("--spectra", type=Path, default=GOLD_SPECTRA_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = build_split(args.structures, args.spectra, args.out_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
