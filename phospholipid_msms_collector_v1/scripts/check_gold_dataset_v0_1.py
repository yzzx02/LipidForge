#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from build_gold_dataset_v0_1 import DEFAULT_OUT_DIR, STRICT_PATH, WORKSPACE_ROOT, build, read_jsonl as read_source_jsonl

EXPECTED = {
    "strict_structures": 814,
    "gold_structure_count": 734,
    "label_correction_count": 41,
    "parser_failure": 0,
}

REPRO_FILES = [
    "all_strict_curation_index_v0_1.jsonl",
    "gold_structures_v0_1.jsonl",
    "gold_spectra_v0_1.jsonl",
    "label_corrections_proposed_v0_1.jsonl",
]

PROVENANCE_FIELDS = [
    "source",
    "source_record_id",
    "spectrum_id",
    "structure_record_id",
    "connectivity_id",
    "gold_lipid_class",
    "peak_identity_hash",
    "acquisition_metadata_hash",
]

PRECOMMIT_DIR = WORKSPACE_ROOT / ".tmp_gold_v0_1_precommit"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_gold_dir(gold_dir: Path) -> dict[str, Any]:
    summary = json.loads((gold_dir / "summary.json").read_text(encoding="utf-8"))
    for key, expected in EXPECTED.items():
        require(summary.get(key) == expected, f"{key}: expected {expected}, observed {summary.get(key)}")
    require(summary.get("layer_count_sum") == 814, f"curation layer sum changed: {summary.get('layer_count_sum')}")
    require(summary.get("gold_connectivity_count") is not None, "gold connectivity count missing")

    index_rows = read_jsonl(gold_dir / "all_strict_curation_index_v0_1.jsonl")
    require(len(index_rows) == 814, "curation index must contain one row per strict structure")
    require(len({row["structure_record_id"] for row in index_rows}) == 814, "curation index contains duplicate structure_record_id")
    require(sum(Counter(row["curation_layer"] for row in index_rows).values()) == 814, "curation layer counts do not sum to 814")
    layer_sets: dict[str, set[str]] = {}
    for row in index_rows:
        layer_sets.setdefault(row["curation_layer"], set()).add(row["structure_record_id"])
        require(row.get("split_group_id") == row.get("connectivity_id"), f"{row['structure_record_id']} split_group_id differs from connectivity_id")
    seen_ids = set()
    for layer, ids in layer_sets.items():
        require(not (seen_ids & ids), f"{layer} overlaps another curation layer")
        seen_ids.update(ids)

    gold_rows = read_jsonl(gold_dir / "gold_structures_v0_1.jsonl")
    require(len(gold_rows) == 734, "gold structure count changed")
    for row in gold_rows:
        require(row["review_status"] == "gold_v0_1", f"{row['structure_record_id']} has wrong review_status")
        require(row["annotation_tier"] == "parser_validated_strict_agreement", f"{row['structure_record_id']} has wrong annotation tier")
        require(row["reconstruction_connectivity_exact"] is True, f"{row['structure_record_id']} reconstruction is not exact")
        require(row["heavy_atom_coverage"] == 1.0, f"{row['structure_record_id']} heavy atom coverage is not 1.0")
        require(row["unassigned_heavy_atom_count"] == 0, f"{row['structure_record_id']} has unassigned atoms")
        require(row["overlap_heavy_atom_count"] == 0, f"{row['structure_record_id']} has overlapping atoms")

    gold_ids = {row["structure_record_id"] for row in gold_rows}
    spectrum_rows = read_jsonl(gold_dir / "gold_spectra_v0_1.jsonl")
    require(all(row["structure_record_id"] in gold_ids for row in spectrum_rows), "gold spectra include non-gold structures")
    require(len(spectrum_rows) == 1758, "gold spectrum count changed")
    for row in spectrum_rows:
        missing = [field for field in PROVENANCE_FIELDS if row.get(field) in {None, ""}]
        require(not missing, f"{row.get('spectrum_id')} has incomplete acquisition provenance: {missing}")
        require(row.get("split_group_id") == row.get("connectivity_id"), f"{row.get('spectrum_id')} split_group_id differs from connectivity_id")

    corrections = read_jsonl(gold_dir / "label_corrections_proposed_v0_1.jsonl")
    require(len(corrections) == 41, "label correction count changed")
    correction_counts = Counter(f"{row['current_label']}->{row['proposed_label']}" for row in corrections)
    require(correction_counts == Counter({"PC->LPC": 22, "PE->LPE": 15, "PI->LPI": 4}), f"unexpected correction counts: {dict(correction_counts)}")
    require(all(row["writeback_allowed"] is False for row in corrections), "a relabel candidate allows writeback")
    require(all(row["decision"] == "pending_user_review" for row in corrections), "a relabel candidate has a non-pending decision")
    correction_ids = {row["structure_record_id"] for row in corrections}
    require(not (gold_ids & correction_ids), "gold and relabel candidate sets overlap")

    with (gold_dir / "label_review_decisions_template.csv").open("r", encoding="utf-8", newline="") as handle:
        review_rows = list(csv.DictReader(handle))
    require(len(review_rows) == 41, "review template row count changed")
    require(all(row["decision"] == "" and row["user_label"] == "" and row["user_note"] == "" for row in review_rows), "review template should leave user decision fields blank")

    conflicts = read_jsonl(gold_dir / "connectivity_layer_conflicts.jsonl")
    require(len(conflicts) == 0, "connectivity layer conflicts were found")
    return {
        "summary": summary,
        "correction_counts": dict(sorted(correction_counts.items())),
        "gold_spectrum_count": len(spectrum_rows),
        "connectivity_conflict_count": len(conflicts),
    }


def check_reproducibility(tmp_root: Path) -> dict[str, Any]:
    run_a = tmp_root / "run_a"
    run_b = tmp_root / "run_b"
    build(run_a, run_a, make_figures=False, quiet=True)
    build(run_b, run_b, make_figures=False, quiet=True)
    hashes = {}
    for name in REPRO_FILES:
        a_hash = sha256(run_a / name)
        b_hash = sha256(run_b / name)
        require(a_hash == b_hash, f"reproducibility mismatch for {name}")
        hashes[name] = a_hash
    reversed_dir = tmp_root / "run_reversed"
    reversed_rows = list(reversed(read_source_jsonl(STRICT_PATH)))
    build(reversed_dir, reversed_dir, make_figures=False, quiet=True, strict_rows_override=reversed_rows)
    reversed_hashes = {name: sha256(reversed_dir / name) for name in REPRO_FILES}
    require(reversed_hashes == hashes, "reversed input order changed stable outputs")
    return {"stable_files": hashes, "run_a": str(run_a), "run_b": str(run_b), "run_reversed": str(reversed_dir)}


def supports_proposed_label(row: dict[str, Any]) -> bool:
    text = str(row.get("provided_name") or "").lower()
    proposed = str(row.get("proposed_label") or "").lower()
    return "lyso" in text or proposed in text


def supports_current_label(row: dict[str, Any]) -> bool:
    text = str(row.get("provided_name") or "").lower()
    current = str(row.get("current_label") or "").lower()
    if current == "pc":
        return "phosphatidylcholine" in text or " pc" in f" {text}"
    if current == "pe":
        return "phosphatidylethanolamine" in text or " pe" in f" {text}"
    if current == "pi":
        return "phosphatidylinositol" in text or " pi" in f" {text}"
    return current in text


def review_sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
    group_order = {"PC->LPC": 0, "PE->LPE": 1, "PI->LPI": 2}
    group = f"{row['current_label']}->{row['proposed_label']}"
    proposed = supports_proposed_label(row)
    current = supports_current_label(row)
    if proposed:
        within = 0
    elif current:
        within = 2
    else:
        within = 1
    return (group_order[group], within, row["structure_record_id"])


def build_relabel_review_package(gold_dir: Path, precommit_dir: Path) -> dict[str, Any]:
    package_dir = precommit_dir / "relabel_review"
    if package_dir.exists():
        shutil.rmtree(package_dir)
    individual_dir = package_dir / "individual_figures"
    contact_dir = package_dir / "contact_sheets"
    individual_dir.mkdir(parents=True, exist_ok=True)
    contact_dir.mkdir(parents=True, exist_ok=True)

    corrections = sorted(read_jsonl(gold_dir / "label_corrections_proposed_v0_1.jsonl"), key=review_sort_key)
    source_figures = WORKSPACE_ROOT / ".tmp_gold_v0_1" / "figures"
    compact_rows = []
    for order, row in enumerate(corrections, start=1):
        source_figure = source_figures / f"{row['structure_record_id']}.png"
        target_figure = individual_dir / source_figure.name
        require(source_figure.exists(), f"missing relabel review figure: {source_figure}")
        shutil.copy2(source_figure, target_figure)
        proposed_support = supports_proposed_label(row)
        current_support = supports_current_label(row)
        graph_ok = (
            row.get("chain_count") == 1
            and row.get("free_hydroxyl_count") == 1
            and row.get("reconstruction_exact") is True
            and row.get("heavy_atom_coverage") == 1.0
            and row.get("unassigned_heavy_atom_count") == 0
            and row.get("overlap_heavy_atom_count") == 0
        )
        if graph_ok and proposed_support:
            batch = "A"
            suggested = "approve_candidate"
        elif graph_ok and not current_support:
            batch = "B"
            suggested = "inspect"
        else:
            batch = "C"
            suggested = "inspect"
        compact_rows.append(
            {
                "review_order": order,
                "review_batch": batch,
                "structure_record_id": row["structure_record_id"],
                "connectivity_id": row["connectivity_id"],
                "current_label": row["current_label"],
                "proposed_label": row["proposed_label"],
                "provided_name": row.get("provided_name"),
                "source_count": len(row.get("sources") or []),
                "spectrum_count": row.get("spectrum_count"),
                "canonical_smiles": row.get("canonical_smiles"),
                "formula": row.get("formula"),
                "formal_charge": row.get("formal_charge"),
                "chain_count": row.get("chain_count"),
                "linkage_pattern": "|".join(row.get("linkage_pattern") or []),
                "free_hydroxyl_count": row.get("free_hydroxyl_count"),
                "reconstruction_exact": row.get("reconstruction_exact"),
                "parser_confidence": row.get("confidence"),
                "name_supports_current_label": current_support,
                "name_supports_proposed_label": proposed_support,
                "suggested_decision": suggested,
                "decision": "",
                "user_label": "",
                "user_note": "",
                "figure_path": str(Path("individual_figures") / target_figure.name),
            }
        )

    compact_fields = [
        "review_order",
        "review_batch",
        "structure_record_id",
        "connectivity_id",
        "current_label",
        "proposed_label",
        "provided_name",
        "source_count",
        "spectrum_count",
        "canonical_smiles",
        "formula",
        "formal_charge",
        "chain_count",
        "linkage_pattern",
        "free_hydroxyl_count",
        "reconstruction_exact",
        "parser_confidence",
        "name_supports_current_label",
        "name_supports_proposed_label",
        "suggested_decision",
        "decision",
        "user_label",
        "user_note",
        "figure_path",
    ]
    with (package_dir / "relabel_review_compact.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=compact_fields)
        writer.writeheader()
        writer.writerows(compact_rows)
    with (package_dir / "relabel_review_detailed.jsonl").open("w", encoding="utf-8") as handle:
        for row in compact_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    contact_sheets = make_contact_sheets(compact_rows, package_dir, contact_dir)
    batch_counts = dict(sorted(Counter(row["review_batch"] for row in compact_rows).items()))
    require(len(list(individual_dir.glob("*.png"))) == 41, "individual relabel figure count changed")
    summary_lines = [
        "# Relabel Review Package",
        "",
        f"- candidates: {len(compact_rows)}",
        f"- batch counts: {batch_counts}",
        "- `suggested_decision` is advisory only; user decision fields are blank.",
        "- No source data is modified by this package.",
        "",
        "Files:",
        "",
        "- `relabel_review_compact.csv`",
        "- `relabel_review_detailed.jsonl`",
        "- `contact_sheets/`",
        "- `individual_figures/`",
    ]
    (package_dir / "relabel_review_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return {
        "candidate_count": len(compact_rows),
        "batch_counts": batch_counts,
        "contact_sheets": contact_sheets,
        "individual_figure_count": len(list(individual_dir.glob("*.png"))),
        "compact_csv": str(package_dir / "relabel_review_compact.csv"),
    }


def make_contact_sheets(compact_rows: list[dict[str, Any]], package_dir: Path, contact_dir: Path) -> list[str]:
    per_page = 6
    sheet_paths = []
    font = ImageFont.load_default()
    for page_index in range(0, len(compact_rows), per_page):
        page_rows = compact_rows[page_index : page_index + per_page]
        cell_w, cell_h = 760, 560
        cols = 2
        rows = 3
        sheet = Image.new("RGB", (cell_w * cols, cell_h * rows), "white")
        draw = ImageDraw.Draw(sheet)
        for offset, row in enumerate(page_rows):
            col = offset % cols
            row_idx = offset // cols
            x = col * cell_w
            y = row_idx * cell_h
            fig = Image.open(package_dir / row["figure_path"]).convert("RGB")
            fig.thumbnail((cell_w - 40, cell_h - 140))
            sheet.paste(fig, (x + 20, y + 120))
            title = [
                f"#{row['review_order']} {row['structure_record_id']}",
                f"{row['current_label']} -> {row['proposed_label']} | batch {row['review_batch']}",
                f"chain={row['chain_count']} linkage={row['linkage_pattern']} spectra={row['spectrum_count']}",
                str(row.get("provided_name") or "")[:95],
            ]
            for line_idx, text in enumerate(title):
                draw.text((x + 18, y + 18 + line_idx * 22), text, fill="black", font=font)
            draw.rectangle((x, y, x + cell_w - 1, y + cell_h - 1), outline=(180, 180, 180))
        path = contact_dir / f"relabel_review_contact_sheet_{len(sheet_paths) + 1:02d}.png"
        sheet.save(path)
        sheet_paths.append(str(path))
    return sheet_paths


def check_missing_input_failure(tmp_root: Path) -> bool:
    missing = tmp_root / "missing_gold_dir"
    try:
        check_gold_dir(missing)
    except FileNotFoundError:
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-check Gold Curation v0.1 outputs.")
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tmp-root", type=Path, default=WORKSPACE_ROOT / ".tmp_gold_v0_1_repro")
    parser.add_argument("--precommit-dir", type=Path, default=PRECOMMIT_DIR)
    parser.add_argument("--skip-review-package", action="store_true")
    args = parser.parse_args()
    result = check_gold_dir(args.gold_dir)
    result["reproducibility"] = check_reproducibility(args.tmp_root)
    result["missing_input_failure_checked"] = check_missing_input_failure(args.tmp_root)
    if not args.skip_review_package:
        result["relabel_review_package"] = build_relabel_review_package(args.gold_dir, args.precommit_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
