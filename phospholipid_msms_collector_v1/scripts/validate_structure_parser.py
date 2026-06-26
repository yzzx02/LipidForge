#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from rdkit import Chem, RDLogger
from rdkit.Chem import Draw

from phospholipid_structure_parser import CONFIG_PATH, load_headgroup_config, parse_glycerophospholipid_smiles

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STRICT_PATH = PROJECT_ROOT / "data" / "expanded_phospholipids_v2" / "phospholipid_msms_strict_v2.jsonl"
MOLECULE_INDEX_PATH = PROJECT_ROOT / "data" / "structure_labeling" / "phase1_v2" / "molecule_index.jsonl"
WORKSPACE_ROOT = PROJECT_ROOT.parent
PHASE3A2_DIR = WORKSPACE_ROOT / ".tmp_phase3a2"
OUT_DIR = WORKSPACE_ROOT / ".tmp_phase3b"

SUPPORTED_CLASSES = ["PA", "PC", "PE", "PG", "PI", "PS", "LPA", "LPC", "LPE", "LPG", "LPI", "LPS"]
NEGATIVE_CONTROL_CLASSES = ["SM", "S1P", "NAPE", "BMP", "CL"]
EXPECTED_HEADGROUP = {
    "PA": "phosphate",
    "LPA": "phosphate",
    "PC": "phosphocholine",
    "LPC": "phosphocholine",
    "PE": "phosphoethanolamine",
    "LPE": "phosphoethanolamine",
    "PG": "phosphoglycerol",
    "LPG": "phosphoglycerol",
    "PI": "phosphoinositol",
    "LPI": "phosphoinositol",
    "PS": "phosphoserine",
    "LPS": "phosphoserine",
}
EXPECTED_CHAIN_COUNT = {
    "PA": 2,
    "PC": 2,
    "PE": 2,
    "PG": 2,
    "PI": 2,
    "PS": 2,
    "LPA": 1,
    "LPC": 1,
    "LPE": 1,
    "LPG": 1,
    "LPI": 1,
    "LPS": 1,
}
FORBIDDEN_NEGATIVE_DERIVATIONS = {
    "SM": {"PC", "LPC"},
    "S1P": {"PA", "LPA"},
    "NAPE": {"PE", "LPE"},
    "BMP": {"PG", "LPG"},
    "CL": {"PG", "LPG", "PA", "LPA"},
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def choose_smiles(row: dict[str, Any]) -> tuple[str | None, str]:
    for key in ("canonical_isomeric_smiles", "canonical_nonisomeric_smiles", "major_component_smiles", "smiles"):
        value = row.get(key)
        if value and str(value).strip().upper() not in {"N/A", "NA", "NONE", "NULL"}:
            return value, key
    raw_values = row.get("raw_smiles_values") or []
    if raw_values:
        return raw_values[0], "raw_smiles_values[0]"
    return None, "missing"


def unique_strict_records(strict_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in strict_rows:
        grouped[row["structure_record_id"]].append(row)
    records = []
    for structure_record_id, rows in sorted(grouped.items()):
        first = rows[0]
        class_counts = Counter(row.get("lipid_class") for row in rows)
        expected_class = class_counts.most_common(1)[0][0]
        smiles, smiles_source = choose_smiles(first)
        names = []
        for row in rows:
            names.extend(row.get("all_names") or [])
            if row.get("name"):
                names.append(row["name"])
            if row.get("record_title"):
                names.append(row["record_title"])
        records.append(
            {
                "structure_record_id": structure_record_id,
                "expected_class": expected_class,
                "lipid_class_values": sorted(k for k in class_counts if k),
                "spectrum_count": len(rows),
                "smiles": smiles,
                "smiles_source": smiles_source,
                "canonical_isomeric_smiles": first.get("canonical_isomeric_smiles"),
                "canonical_nonisomeric_smiles": first.get("canonical_nonisomeric_smiles"),
                "formula": first.get("formula"),
                "linkage_modifications": sorted({mod for row in rows for mod in (row.get("linkage_modifications") or [])}),
                "names": sorted(set(names))[:20],
            }
        )
    return records


def molecule_index_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["structure_record_id"]: row for row in rows}


def expected_linkage_pattern(record: dict[str, Any], chain_count: int | None) -> tuple[list[str] | None, bool]:
    if not chain_count:
        return None, False
    text = " ".join(
        [record.get("expected_class") or ""]
        + [str(x) for x in record.get("linkage_modifications") or []]
        + [str(x) for x in record.get("names") or []]
    ).lower()
    vinyl_tokens = set(re.findall(r"\b(?:p-|plasmenyl|vinyl ether|vinyl-ether|alkenyl|plasmalogen)\b", text))
    alkyl_tokens = set(re.findall(r"\b(?:o-|plasmanyl|alkyl ether|alkyl-ether|alkyl)\b", text))
    shorthand_vinyl = set()
    shorthand_alkyl = set()
    for carbons, double_bonds, suffix in re.findall(r"\b(\d{1,2}):(\d{1,2})([ep])\b", text):
        token = f"{carbons}:{double_bonds}{suffix}"
        if suffix == "p" or (suffix == "e" and int(double_bonds) > 0):
            shorthand_vinyl.add(token)
        elif suffix == "e":
            shorthand_alkyl.add(token)
    vinyl_hits = len(shorthand_vinyl) + (1 if vinyl_tokens else 0)
    alkyl_hits = len(shorthand_alkyl) + (1 if alkyl_tokens else 0)
    explicit = bool(vinyl_hits or alkyl_hits or re.search(r"\b(?:diacyl|acyl|ester)\b", text))
    if vinyl_hits:
        return sorted(["vinyl_ether"] * min(vinyl_hits, chain_count) + ["ester"] * max(0, chain_count - min(vinyl_hits, chain_count))), True
    if alkyl_hits:
        return sorted(["alkyl_ether"] * min(alkyl_hits, chain_count) + ["ester"] * max(0, chain_count - min(alkyl_hits, chain_count))), True
    if explicit:
        return sorted(["ester"] * chain_count), True
    return None, False


def gold_exclusion_reason(record: dict[str, Any], result: dict[str, Any]) -> str | None:
    if not record.get("smiles"):
        return "canonical structure missing; do not use as a gold standard"
    if record.get("expected_class") == "PS" and result.get("headgroup_id") not in {None, "phosphoserine"}:
        return "source PS structure lacks the ordinary serine headgroup; do not use as PS gold standard"
    return None


def evaluate_record(record: dict[str, Any]) -> dict[str, Any]:
    result = parse_glycerophospholipid_smiles(record.get("smiles"), record["structure_record_id"])
    expected_class = record["expected_class"]
    exclusion_reason = gold_exclusion_reason(record, result)
    gold_eligible = expected_class in SUPPORTED_CLASSES and exclusion_reason is None
    expected_chain_count = EXPECTED_CHAIN_COUNT.get(expected_class)
    expected_linkage, linkage_evaluable = expected_linkage_pattern(record, expected_chain_count)
    detected_linkage = sorted(result.get("linkage_pattern") or [])
    headgroup_correct = result.get("headgroup_id") == EXPECTED_HEADGROUP.get(expected_class)
    backbone_correct = result.get("backbone_family") == "glycerol" and result.get("backbone_confidence") == "high"
    chain_count_correct = expected_chain_count is not None and result.get("chain_count") == expected_chain_count
    linkage_correct = expected_linkage is not None and detected_linkage == expected_linkage
    row = {
        **record,
        "gold_standard_eligible": gold_eligible,
        "gold_exclusion_reason": exclusion_reason,
        "parse_status": result.get("parse_status"),
        "failure_reasons": result.get("failure_reasons") or [],
        "headgroup_id": result.get("headgroup_id"),
        "headgroup_match_tier": result.get("headgroup_match_tier"),
        "charge_normalization_used": result.get("charge_normalization_used"),
        "backbone_family": result.get("backbone_family"),
        "chain_count": result.get("chain_count"),
        "linkage_pattern": detected_linkage,
        "expected_linkage_pattern": expected_linkage,
        "linkage_evaluable": linkage_evaluable,
        "derived_lipid_class_candidate": result.get("derived_lipid_class_candidate"),
        "headgroup_correct": bool(headgroup_correct) if gold_eligible else None,
        "backbone_correct": bool(backbone_correct) if gold_eligible else None,
        "chain_count_correct": bool(chain_count_correct) if gold_eligible else None,
        "linkage_correct": bool(linkage_correct) if gold_eligible and linkage_evaluable else None,
        "reconstruction_exact": bool(result.get("reconstruction_connectivity_exact")) if gold_eligible else None,
        "parser_result": result,
    }
    row["success_for_gold"] = bool(
        gold_eligible
        and result.get("parse_status") == "success"
        and row["derived_lipid_class_candidate"] == expected_class
        and row["headgroup_correct"]
        and row["backbone_correct"]
        and row["chain_count_correct"]
        and row["reconstruction_exact"]
    )
    return row


def class_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_class[row["expected_class"]].append(row)
    out = []
    for cls in sorted(by_class, key=lambda c: (c not in SUPPORTED_CLASSES, c)):
        items = by_class[cls]
        eligible = [row for row in items if row["gold_standard_eligible"]]
        success = [row for row in eligible if row["success_for_gold"]]
        out.append(
            {
                "class": cls,
                "total": len(items),
                "spectrum_count": sum(row["spectrum_count"] for row in items),
                "gold_eligible": len(eligible),
                "success": len(success),
                "success_rate": round(len(success) / len(eligible), 6) if eligible else None,
                "unsupported": sum(1 for row in items if str(row["parse_status"]).startswith("unsupported")),
                "ambiguous": sum(1 for row in items if str(row["parse_status"]).startswith("ambiguous")),
                "failed": sum(1 for row in items if row["parse_status"] == "failed"),
                "headgroup_correct": sum(1 for row in eligible if row["headgroup_correct"]),
                "backbone_correct": sum(1 for row in eligible if row["backbone_correct"]),
                "chain_count_correct": sum(1 for row in eligible if row["chain_count_correct"]),
                "linkage_evaluable": sum(1 for row in eligible if row["linkage_evaluable"]),
                "linkage_correct": sum(1 for row in eligible if row["linkage_correct"]),
                "reconstruction_exact": sum(1 for row in eligible if row["reconstruction_exact"]),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def compact_result(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    parser_result = result.pop("parser_result")
    result.update(
        {
            "backbone_atom_indices": parser_result.get("backbone_atom_indices"),
            "headgroup_atom_indices": parser_result.get("headgroup_atom_indices"),
            "chains": parser_result.get("chains"),
            "cut_bonds": parser_result.get("cut_bonds"),
            "unassigned_atom_indices": parser_result.get("unassigned_atom_indices"),
            "overlapping_atom_indices": parser_result.get("overlapping_atom_indices"),
            "heavy_atom_coverage": parser_result.get("heavy_atom_coverage"),
            "unassigned_heavy_atom_count": parser_result.get("unassigned_heavy_atom_count"),
            "overlap_heavy_atom_count": parser_result.get("overlap_heavy_atom_count"),
            "warnings": parser_result.get("warnings"),
        }
    )
    return result


def linkage_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        detected = row["linkage_pattern"] or []
        expected = row["expected_linkage_pattern"] or []
        maybe_ether = bool(set(detected) & {"alkyl_ether", "vinyl_ether"} or set(expected) & {"alkyl_ether", "vinyl_ether"})
        if not maybe_ether and not row["linkage_evaluable"]:
            continue
        out.append(
            {
                "structure_record_id": row["structure_record_id"],
                "expected_class": row["expected_class"],
                "gold_standard_eligible": row["gold_standard_eligible"],
                "linkage_evaluable": row["linkage_evaluable"],
                "true_alkyl_ether": expected.count("alkyl_ether"),
                "detected_alkyl_ether": detected.count("alkyl_ether"),
                "true_vinyl_ether": expected.count("vinyl_ether"),
                "detected_vinyl_ether": detected.count("vinyl_ether"),
                "expected_linkage_pattern": "|".join(expected),
                "detected_linkage_pattern": "|".join(detected),
                "linkage_correct": row["linkage_correct"],
                "name_or_modification_basis": " ".join((row.get("linkage_modifications") or []) + (row.get("names") or [])[:3]),
            }
        )
    return out


def negative_control_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for cls in NEGATIVE_CONTROL_CLASSES:
        items = [row for row in rows if row["expected_class"] == cls]
        if not items:
            out.append(
                {
                    "negative_control_class": cls,
                    "structure_record_id": None,
                    "sample_status": "no_strict_sample",
                    "parse_status": None,
                    "derived_lipid_class_candidate": None,
                    "violation": False,
                    "failure_reasons": [],
                }
            )
            continue
        forbidden = FORBIDDEN_NEGATIVE_DERIVATIONS[cls]
        for row in items:
            out.append(
                {
                    "negative_control_class": cls,
                    "structure_record_id": row["structure_record_id"],
                    "sample_status": "tested",
                    "parse_status": row["parse_status"],
                    "derived_lipid_class_candidate": row["derived_lipid_class_candidate"],
                    "violation": row["derived_lipid_class_candidate"] in forbidden,
                    "failure_reasons": row["failure_reasons"],
                }
            )
    return out


def highlight_colors(result: dict[str, Any]) -> tuple[list[int], dict[int, tuple[float, float, float]], list[int], dict[int, tuple[float, float, float]]]:
    atom_colors: dict[int, tuple[float, float, float]] = {}
    for idx in result.get("headgroup_atom_indices") or []:
        atom_colors[idx] = (0.25, 0.45, 0.95)
    for idx in result.get("backbone_atom_indices") or []:
        atom_colors[idx] = (1.0, 0.55, 0.15)
    chain_palette = [(0.2, 0.65, 0.35), (0.55, 0.35, 0.9), (0.65, 0.65, 0.25)]
    for chain_idx, chain in enumerate(result.get("chains") or []):
        color = chain_palette[chain_idx % len(chain_palette)]
        for idx in chain.get("chain_partition_atom_indices") or chain.get("chain_atom_indices") or []:
            atom_colors[idx] = color
    bond_colors: dict[int, tuple[float, float, float]] = {}
    mol = Chem.MolFromSmiles(result.get("input_smiles") or "")
    if mol is not None:
        for cut in result.get("cut_bonds") or []:
            bond = mol.GetBondBetweenAtoms(int(cut["atom_1"]), int(cut["atom_2"]))
            if bond is not None:
                bond_colors[bond.GetIdx()] = (0.95, 0.1, 0.1)
    return sorted(atom_colors), atom_colors, sorted(bond_colors), bond_colors


def draw_figure(row: dict[str, Any], path: Path) -> None:
    result = row["parser_result"]
    smiles = result.get("input_smiles")
    if not smiles:
        return
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return
    drawer = Draw.MolDraw2DCairo(1000, 720)
    options = drawer.drawOptions()
    options.addAtomIndices = True
    atoms, atom_colors, bonds, bond_colors = highlight_colors(result)
    drawer.DrawMolecule(mol, highlightAtoms=atoms, highlightAtomColors=atom_colors, highlightBonds=bonds, highlightBondColors=bond_colors)
    drawer.FinishDrawing()
    path.write_bytes(drawer.GetDrawingText())


def select_figure_rows(rows: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    selected: list[tuple[str, dict[str, Any]]] = []
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["success_for_gold"]:
            by_class[row["expected_class"]].append(row)
    for cls in SUPPORTED_CLASSES:
        for row in by_class.get(cls, [])[:3]:
            selected.append((f"success_{cls}_{row['structure_record_id']}.png", row))
    for row in rows:
        if row["parse_status"] == "failed":
            selected.append((f"failed_{row['expected_class']}_{row['structure_record_id']}.png", row))
        if str(row["parse_status"]).startswith("ambiguous"):
            selected.append((f"ambiguous_{row['expected_class']}_{row['structure_record_id']}.png", row))
        if set(row["linkage_pattern"] or []) & {"alkyl_ether", "vinyl_ether"}:
            selected.append((f"ether_{row['expected_class']}_{row['structure_record_id']}.png", row))
        if row["expected_class"] == "PS" and row.get("headgroup_match_tier") == 3:
            selected.append((f"ps_charge_normalized_{row['structure_record_id']}.png", row))
        if row["expected_class"] in NEGATIVE_CONTROL_CLASSES and row["derived_lipid_class_candidate"] in FORBIDDEN_NEGATIVE_DERIVATIONS[row["expected_class"]]:
            selected.append((f"negative_violation_{row['expected_class']}_{row['structure_record_id']}.png", row))
    seen = set()
    deduped = []
    for name, row in selected:
        if name in seen:
            continue
        seen.add(name)
        deduped.append((name, row))
    return deduped


def user_review_markdown(rows: list[dict[str, Any]], negative_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Phase 3B User Review Items",
        "",
        "- AC3PIM2 and AC4PIM2 remain deferred invalid SMILES and were not added to executable parser rules.",
        "- S1P/CerP keep a shared phosphate-core relationship, but v0.1 does not parse sphingoid or ceramide backbones.",
        "- The parser did not use lipid_class, name, abbreviation, or source label as structural input.",
        "",
        "## Excluded Gold Standards",
    ]
    for row in rows:
        if row.get("gold_exclusion_reason"):
            lines.append(f"- `{row['structure_record_id']}` ({row['expected_class']}): {row['gold_exclusion_reason']}")
    lines.extend(["", "## Failures And Ambiguities"])
    for row in rows:
        if row["parse_status"] in {"failed", "ambiguous_backbone"} and not row.get("gold_exclusion_reason"):
            lines.append(f"- `{row['structure_record_id']}` ({row['expected_class']}): {', '.join(row['failure_reasons'])}")
    lines.extend(["", "## Strict Label And Topology Conflicts"])
    conflicts = [row for row in rows if row["gold_standard_eligible"] and not row["success_for_gold"]]
    for row in conflicts:
        names = "; ".join((row.get("names") or [])[:2])
        reason = ", ".join(row.get("failure_reasons") or []) or f"derived `{row.get('derived_lipid_class_candidate')}`"
        lines.append(
            f"- `{row['structure_record_id']}` ({row['expected_class']}): {row['parse_status']}; {reason}; names: {names}"
        )
    lines.extend(["", "## Negative Controls"])
    for row in negative_rows:
        if row["sample_status"] == "no_strict_sample":
            lines.append(f"- `{row['negative_control_class']}`: no strict sample available in the current dataset.")
        elif row["violation"]:
            lines.append(
                f"- `{row['structure_record_id']}` ({row['negative_control_class']}): derived as `{row['derived_lipid_class_candidate']}` and needs parser review."
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate glycerophospholipid parser v0.1 on strict structures.")
    parser.add_argument("--strict", type=Path, default=STRICT_PATH)
    parser.add_argument("--molecule-index", type=Path, default=MOLECULE_INDEX_PATH)
    parser.add_argument("--phase3a2-dir", type=Path, default=PHASE3A2_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    load_headgroup_config(CONFIG_PATH)
    strict_rows = read_jsonl(args.strict)
    molecule_rows = read_jsonl(args.molecule_index)
    _molecule_index = molecule_index_by_id(molecule_rows)
    records = unique_strict_records(strict_rows)
    results = [evaluate_record(record) for record in records]
    metrics = class_metrics(results)
    linkages = linkage_rows(results)
    negative = negative_control_rows(results)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "strict_parser_results.jsonl", (compact_result(row) for row in results))
    write_csv(
        args.out_dir / "strict_parser_by_class.csv",
        metrics,
        [
            "class",
            "total",
            "spectrum_count",
            "gold_eligible",
            "success",
            "success_rate",
            "unsupported",
            "ambiguous",
            "failed",
            "headgroup_correct",
            "backbone_correct",
            "chain_count_correct",
            "linkage_evaluable",
            "linkage_correct",
            "reconstruction_exact",
        ],
    )
    write_csv(args.out_dir / "linkage_validation.csv", linkages)
    write_jsonl(args.out_dir / "negative_control_results.jsonl", negative)
    failures = [compact_result(row) for row in results if row["parse_status"] == "failed"]
    ambiguous = [compact_result(row) for row in results if str(row["parse_status"]).startswith("ambiguous")]
    write_jsonl(args.out_dir / "failure_examples.jsonl", failures)
    write_jsonl(args.out_dir / "ambiguous_examples.jsonl", ambiguous)
    write_csv(
        args.out_dir / "reconstruction_validation.csv",
        [
            {
                "structure_record_id": row["structure_record_id"],
                "expected_class": row["expected_class"],
                "gold_standard_eligible": row["gold_standard_eligible"],
                "parse_status": row["parse_status"],
                "reconstruction_connectivity_exact": row["parser_result"].get("reconstruction_connectivity_exact"),
                "reconstruction_nonisomeric_smiles_exact": row["parser_result"].get("reconstruction_nonisomeric_smiles_exact"),
                "reconstruction_isomeric_smiles_exact": row["parser_result"].get("reconstruction_isomeric_smiles_exact"),
            }
            for row in results
        ],
    )
    write_csv(
        args.out_dir / "headgroup_validation.csv",
        [
            {
                "structure_record_id": row["structure_record_id"],
                "expected_class": row["expected_class"],
                "gold_standard_eligible": row["gold_standard_eligible"],
                "headgroup_id": row["headgroup_id"],
                "headgroup_match_tier": row["headgroup_match_tier"],
                "charge_normalization_used": row["charge_normalization_used"],
                "headgroup_correct": row["headgroup_correct"],
            }
            for row in results
        ],
    )
    for name, row in select_figure_rows(results):
        draw_figure(row, figure_dir / name)

    status_counts = Counter(row["parse_status"] for row in results)
    supported_eligible = [row for row in results if row["gold_standard_eligible"]]
    supported_success = [row for row in supported_eligible if row["success_for_gold"]]
    linkage_evaluable_rows = [row for row in linkages if row["linkage_evaluable"]]
    linkage_summary = {
        "metadata_evaluable_rows": len(linkage_evaluable_rows),
        "graph_detected_rows": sum(1 for row in linkages if row["detected_alkyl_ether"] or row["detected_vinyl_ether"]),
        "true_alkyl_ether": sum(row["true_alkyl_ether"] for row in linkages),
        "detected_alkyl_ether": sum(row["detected_alkyl_ether"] for row in linkages),
        "true_vinyl_ether": sum(row["true_vinyl_ether"] for row in linkages),
        "detected_vinyl_ether": sum(row["detected_vinyl_ether"] for row in linkages),
        "false_positive_alkyl_ether": sum(1 for row in linkage_evaluable_rows if row["detected_alkyl_ether"] > row["true_alkyl_ether"]),
        "false_negative_alkyl_ether": sum(1 for row in linkage_evaluable_rows if row["detected_alkyl_ether"] < row["true_alkyl_ether"]),
        "false_positive_vinyl_ether": sum(1 for row in linkage_evaluable_rows if row["detected_vinyl_ether"] > row["true_vinyl_ether"]),
        "false_negative_vinyl_ether": sum(1 for row in linkage_evaluable_rows if row["detected_vinyl_ether"] < row["true_vinyl_ether"]),
    }
    negative_summary = {
        cls: {
            "samples": sum(1 for row in negative if row["negative_control_class"] == cls and row["sample_status"] == "tested"),
            "violations": sum(1 for row in negative if row["negative_control_class"] == cls and row["violation"]),
        }
        for cls in NEGATIVE_CONTROL_CLASSES
    }
    summary = {
        "parser_version": "gpl_v0.1",
        "strict_unique_structures": len(results),
        "strict_spectra": sum(row["spectrum_count"] for row in results),
        "supported_gold_eligible": len(supported_eligible),
        "supported_gold_success": len(supported_success),
        "supported_gold_success_rate": round(len(supported_success) / len(supported_eligible), 6) if supported_eligible else None,
        "parse_status_counts": dict(status_counts),
        "by_class": metrics,
        "headgroup_correct": sum(1 for row in supported_eligible if row["headgroup_correct"]),
        "backbone_correct": sum(1 for row in supported_eligible if row["backbone_correct"]),
        "chain_count_correct": sum(1 for row in supported_eligible if row["chain_count_correct"]),
        "linkage_evaluable": sum(1 for row in supported_eligible if row["linkage_evaluable"]),
        "linkage_correct": sum(1 for row in supported_eligible if row["linkage_correct"]),
        "reconstruction_exact": sum(1 for row in supported_eligible if row["reconstruction_exact"]),
        "linkage_summary": linkage_summary,
        "negative_control_summary": negative_summary,
        "figure_count": len(list(figure_dir.glob("*.png"))),
        "outputs": {
            "out_dir": str(args.out_dir),
            "figures": str(figure_dir),
            "phase3a2_dir": str(args.phase3a2_dir),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.out_dir / "user_review_items.md").write_text(user_review_markdown(results, negative), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
