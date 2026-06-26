#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from rdkit import Chem, RDLogger
from rdkit.Chem import Draw

from validate_structure_parser import (
    EXPECTED_CHAIN_COUNT,
    SUPPORTED_CLASSES,
    evaluate_record,
    read_jsonl,
    unique_strict_records,
)

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
STRICT_PATH = PROJECT_ROOT / "data" / "expanded_phospholipids_v2" / "phospholipid_msms_strict_v2.jsonl"
MOLECULE_INDEX_PATH = PROJECT_ROOT / "data" / "structure_labeling" / "phase1_v2" / "molecule_index.jsonl"
SPECTRUM_TO_STRUCTURE_PATH = PROJECT_ROOT / "data" / "structure_labeling" / "phase1_v2" / "spectrum_to_structure.jsonl"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "structure_labeling" / "gold_v0_1"
DEFAULT_AUDIT_DIR = WORKSPACE_ROOT / ".tmp_gold_v0_1"

EXPECTED_BASELINE = {
    "strict_structures": 814,
    "graph_parse_success_reconstruction_exact": 776,
    "gold_v0_1": 734,
    "label_disagreement_candidates": 41,
    "parser_failure": 0,
}


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
            writer.writerow({key: row.get(key) for key in fieldnames})


def require_input(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")


def require_inputs() -> None:
    for path in [STRICT_PATH, MOLECULE_INDEX_PATH, SPECTRUM_TO_STRUCTURE_PATH]:
        require_input(path)


def safe_staging_dir(target: Path) -> Path:
    resolved = target.resolve()
    forbidden = {Path("/").resolve(), PROJECT_ROOT.resolve(), WORKSPACE_ROOT.resolve()}
    if resolved in forbidden or resolved == resolved.parent:
        raise ValueError(f"Refusing to replace unsafe output directory: {target}")
    staging = resolved.parent / f".{resolved.name}.tmp_gold_build"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=False)
    return staging


def replace_output_dir(staging: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    staging.rename(target)


def first_name(row: dict[str, Any]) -> str | None:
    names = row.get("names") or []
    return names[0] if names else None


def group_strict_rows(strict_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in strict_rows:
        grouped[row["structure_record_id"]].append(row)
    return {key: sorted(rows, key=lambda r: (str(r.get("source") or ""), str(r.get("source_record_id") or ""), str(r.get("spectrum_id") or ""))) for key, rows in grouped.items()}


def molecule_index_by_structure(path: Path) -> dict[str, dict[str, Any]]:
    return {row["structure_record_id"]: row for row in read_jsonl(path)}


def enrich_result_rows(rows: list[dict[str, Any]], strict_by_id: dict[str, list[dict[str, Any]]], molecule_by_id: dict[str, dict[str, Any]]) -> None:
    for row in rows:
        sid = row["structure_record_id"]
        strict_rows = strict_by_id[sid]
        first = strict_rows[0]
        molecule = molecule_by_id.get(sid, {})
        row["connectivity_id"] = first.get("connectivity_id") or molecule.get("connectivity_id")
        row["sources"] = sorted({str(item.get("source")) for item in strict_rows if item.get("source")})
        row["source_record_ids"] = sorted({str(item.get("source_record_id")) for item in strict_rows if item.get("source_record_id")})
        row["spectrum_ids"] = sorted({str(item.get("spectrum_id")) for item in strict_rows if item.get("spectrum_id")})
        row["provided_name"] = first_name(row) or first.get("name") or first.get("record_title")
        row["canonical_isomeric_smiles"] = row.get("canonical_isomeric_smiles") or molecule.get("canonical_isomeric_smiles")
        row["canonical_nonisomeric_smiles"] = row.get("canonical_nonisomeric_smiles") or molecule.get("canonical_nonisomeric_smiles")


def names_text(row: dict[str, Any]) -> str:
    return " | ".join(str(name) for name in (row.get("names") or []))


def classify_curation_layer(row: dict[str, Any]) -> tuple[str, str, bool]:
    status = row.get("parse_status")
    expected = row.get("expected_class")
    derived = row.get("derived_lipid_class_candidate")
    if row.get("success_for_gold"):
        return "gold_v0_1", "parser_success_label_agree", False
    if expected not in SUPPORTED_CLASSES:
        return "class_out_of_scope", "strict_class_outside_glycerophospholipid_v0_1_scope", False
    if row.get("gold_standard_eligible") and status == "success" and row.get("reconstruction_exact") and derived != expected:
        if row.get("chain_count") == 1 and EXPECTED_CHAIN_COUNT.get(expected) == 2:
            return "relabel_review", "single_chain_graph_with_free_glycerol_hydroxyl", True
        return "relabel_review", "parser_derived_class_disagrees_with_source_strict_label", True
    if row.get("gold_exclusion_reason") or status == "invalid_input" or not row.get("smiles"):
        if not row.get("smiles"):
            return "invalid_or_incomplete_structure", "canonical_structure_missing", True
        return "invalid_or_incomplete_structure", "source_label_structure_conflict", True
    if expected == "PA" and status == "unsupported_backbone":
        return "invalid_or_incomplete_structure", "zero_chain_glycerophosphate_labelled_as_PA", True
    if expected == "PC" and status == "unsupported_backbone":
        return "invalid_or_incomplete_structure", "headgroup_or_glycerophosphocholine_record_lacks_hydrophobic_chain", True
    if status == "unsupported_extra_substitution":
        return "parser_scope_exclusion", "extra_backbone_substitution_outside_v0_1", True
    if status == "unsupported_topology":
        reasons = set(row.get("failure_reasons") or [])
        if "multiple_phosphorus_atoms" in reasons:
            return "parser_scope_exclusion", "multi_phosphorus_topology_outside_v0_1", True
        return "parser_scope_exclusion", "unsupported_topology_outside_v0_1", True
    if status == "unsupported_backbone":
        if "sphingosyl" in names_text(row).lower() or row.get("backbone_family") == "sphingoid_or_unsupported":
            return "parser_scope_exclusion", "sphingoid_backbone_outside_v0_1", True
        return "parser_scope_exclusion", "unsupported_backbone_outside_v0_1", True
    return "needs_manual_review", "unclassified_non_gold_case", True


def apply_curation_layers(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index_rows = []
    for row in sorted(rows, key=lambda item: item["structure_record_id"]):
        layer, reason, needs_review = classify_curation_layer(row)
        index_rows.append(
            {
                "structure_record_id": row["structure_record_id"],
                "connectivity_id": row.get("connectivity_id"),
                "source_strict_label": row.get("expected_class"),
                "derived_class": row.get("derived_lipid_class_candidate"),
                "parser_status": row.get("parse_status"),
                "curation_layer": layer,
                "curation_reason": reason,
                "spectrum_count": row.get("spectrum_count"),
                "source_record_ids": row.get("source_record_ids") or [],
                "needs_user_review": needs_review,
                "split_group_id": row.get("connectivity_id"),
            }
        )
    conflicts = connectivity_conflicts(index_rows)
    if conflicts:
        conflict_ids = {item["connectivity_id"] for item in conflicts}
        for row in index_rows:
            if row.get("connectivity_id") in conflict_ids:
                row["curation_layer"] = "needs_manual_review"
                row["curation_reason"] = "connectivity_id_crosses_curation_layers"
                row["needs_user_review"] = True
    return index_rows


def connectivity_conflicts(index_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_conn: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in index_rows:
        conn = row.get("connectivity_id")
        if conn:
            by_conn[conn].append(row)
    out = []
    for conn, rows in sorted(by_conn.items()):
        layers = sorted({row["curation_layer"] for row in rows})
        if len(layers) <= 1:
            continue
        out.append(
            {
                "connectivity_id": conn,
                "layers": layers,
                "structure_record_ids": [row["structure_record_id"] for row in rows],
                "recommended_layer": "needs_manual_review",
            }
        )
    return out


def gold_structure_row(row: dict[str, Any]) -> dict[str, Any]:
    result = row["parser_result"]
    return {
        "gold_version": "0.1",
        "structure_record_id": row["structure_record_id"],
        "connectivity_id": row.get("connectivity_id"),
        "split_group_id": row.get("connectivity_id"),
        "canonical_isomeric_smiles": row.get("canonical_isomeric_smiles"),
        "canonical_nonisomeric_smiles": row.get("canonical_nonisomeric_smiles"),
        "gold_lipid_class": row.get("expected_class"),
        "source_strict_label": row.get("expected_class"),
        "headgroup_id": result.get("headgroup_id"),
        "backbone_family": result.get("backbone_family"),
        "headgroup_atom_indices": result.get("headgroup_atom_indices") or [],
        "backbone_atom_indices": result.get("backbone_atom_indices") or [],
        "chains": result.get("chains") or [],
        "chain_count": result.get("chain_count"),
        "linkage_pattern": result.get("linkage_pattern") or [],
        "free_hydroxyl_sites": result.get("free_hydroxyl_sites") or [],
        "sn_assignment_status": "unknown",
        "parser_sn_assignment_status": result.get("sn_assignment_status"),
        "cut_bonds": result.get("cut_bonds") or [],
        "heavy_atom_coverage": result.get("heavy_atom_coverage"),
        "unassigned_heavy_atom_count": result.get("unassigned_heavy_atom_count"),
        "overlap_heavy_atom_count": result.get("overlap_heavy_atom_count"),
        "reconstruction_connectivity_exact": result.get("reconstruction_connectivity_exact"),
        "annotation_tier": "parser_validated_strict_agreement",
        "annotation_evidence": ["source_strict_label", "graph_parser_v0.1", "exact_reconstruction"],
        "review_status": "gold_v0_1",
        "reference_only": False,
        "sample_id": None,
        "evidence_type": None,
        "manual_reviewer": None,
        "review_date": None,
        "supporting_spectra": [],
        "annotation_notes": None,
    }


def spectrum_gold_rows(strict_rows: list[dict[str, Any]], gold_by_structure: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in strict_rows:
        sid = row.get("structure_record_id")
        gold = gold_by_structure.get(sid)
        if not gold:
            continue
        rows.append(
            {
                "source": row.get("source"),
                "source_record_id": row.get("source_record_id"),
                "spectrum_id": row.get("spectrum_id"),
                "structure_record_id": sid,
                "connectivity_id": gold.get("connectivity_id"),
                "split_group_id": gold.get("split_group_id"),
                "gold_lipid_class": gold.get("gold_lipid_class"),
                "headgroup_id": gold.get("headgroup_id"),
                "backbone_family": gold.get("backbone_family"),
                "chain_count": gold.get("chain_count"),
                "linkage_pattern": gold.get("linkage_pattern"),
                "peak_identity_hash": row.get("peak_identity_hash"),
                "acquisition_metadata_hash": row.get("acquisition_metadata_hash"),
                "acquisition_record_hash": row.get("acquisition_record_hash"),
                "duplicate_relation": row.get("duplicate_relation"),
                "adduct": row.get("adduct"),
                "polarity": row.get("polarity"),
                "collision_energy_normalized": row.get("collision_energy_normalized"),
                "instrument": row.get("instrument"),
                "license": row.get("license"),
            }
        )
    return sorted(rows, key=lambda item: (item["structure_record_id"], str(item.get("source") or ""), str(item.get("source_record_id") or ""), str(item.get("spectrum_id") or "")))


def relabel_rows(rows: list[dict[str, Any]], index_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in sorted(rows, key=lambda item: item["structure_record_id"]):
        if index_by_id[row["structure_record_id"]]["curation_layer"] != "relabel_review":
            continue
        result = row["parser_result"]
        free_hydroxyl_count = len(result.get("free_hydroxyl_sites") or [])
        mol = Chem.MolFromSmiles(result.get("input_smiles") or "")
        formal_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms()) if mol is not None else None
        out.append(
            {
                "structure_record_id": row["structure_record_id"],
                "connectivity_id": row.get("connectivity_id"),
                "current_label": row.get("expected_class"),
                "proposed_label": row.get("derived_lipid_class_candidate"),
                "headgroup_id": result.get("headgroup_id"),
                "chain_count": result.get("chain_count"),
                "free_hydroxyl_count": free_hydroxyl_count,
                "linkage_pattern": result.get("linkage_pattern") or [],
                "reconstruction_exact": result.get("reconstruction_connectivity_exact"),
                "heavy_atom_coverage": result.get("heavy_atom_coverage"),
                "unassigned_heavy_atom_count": result.get("unassigned_heavy_atom_count"),
                "overlap_heavy_atom_count": result.get("overlap_heavy_atom_count"),
                "proposal_reason": "molecular graph contains one hydrophobic chain",
                "confidence": "high",
                "decision": "pending_user_review",
                "writeback_allowed": False,
                "provided_name": row.get("provided_name"),
                "canonical_smiles": row.get("canonical_isomeric_smiles") or row.get("canonical_nonisomeric_smiles") or row.get("smiles"),
                "formula": row.get("formula"),
                "formal_charge": formal_charge,
                "spectrum_count": row.get("spectrum_count"),
                "source_record_ids": row.get("source_record_ids") or [],
                "sources": row.get("sources") or [],
            }
        )
    return out


def review_template_rows(corrections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in corrections:
        rows.append(
            {
                "structure_record_id": row["structure_record_id"],
                "connectivity_id": row["connectivity_id"],
                "current_label": row["current_label"],
                "proposed_label": row["proposed_label"],
                "provided_name": row.get("provided_name"),
                "canonical_smiles": row.get("canonical_smiles"),
                "chain_count": row.get("chain_count"),
                "linkage_pattern": "|".join(row.get("linkage_pattern") or []),
                "spectrum_count": row.get("spectrum_count"),
                "decision": "",
                "user_label": "",
                "user_note": "",
            }
        )
    return rows


def compact_non_gold_row(row: dict[str, Any], index_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result = row["parser_result"]
    index = index_by_id[row["structure_record_id"]]
    return {
        **index,
        "provided_name": row.get("provided_name"),
        "canonical_smiles": row.get("canonical_isomeric_smiles") or row.get("canonical_nonisomeric_smiles") or row.get("smiles"),
        "failure_reasons": row.get("failure_reasons") or [],
        "headgroup_id": result.get("headgroup_id"),
        "backbone_family": result.get("backbone_family"),
        "chain_count": result.get("chain_count"),
        "linkage_pattern": result.get("linkage_pattern") or [],
        "reconstruction_connectivity_exact": result.get("reconstruction_connectivity_exact"),
        "unassigned_heavy_atom_count": result.get("unassigned_heavy_atom_count"),
        "overlap_heavy_atom_count": result.get("overlap_heavy_atom_count"),
    }


def linkage_bucket(pattern: list[str]) -> str:
    values = sorted(pattern or [])
    if values == ["ester", "ester"]:
        return "ester/ester"
    if values == ["alkyl_ether", "ester"]:
        return "alkyl_ether/ester"
    if values == ["ester", "vinyl_ether"]:
        return "vinyl_ether/ester"
    if values == ["ester"]:
        return "single_chain_ester"
    if values == ["alkyl_ether"]:
        return "single_chain_alkyl_ether"
    if values == ["vinyl_ether"]:
        return "single_chain_vinyl_ether"
    return "other"


def count_by_structure_key(
    gold_structures: list[dict[str, Any]],
    gold_spectra: list[dict[str, Any]],
    structure_key: str,
    spectrum_key: str,
    output_key: str,
) -> list[dict[str, Any]]:
    structures_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in gold_structures:
        structures_by_key[str(row.get(structure_key))].append(row)
    spectra_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in gold_spectra:
        spectra_by_key[str(row.get(spectrum_key))].append(row)
    all_keys = sorted(set(structures_by_key) | set(spectra_by_key))
    out = []
    for value in all_keys:
        srows = structures_by_key.get(value, [])
        arows = spectra_by_key.get(value, [])
        out.append(
            {
                output_key: value,
                "structure_count": len({row["structure_record_id"] for row in srows}),
                "connectivity_count": len({row["connectivity_id"] for row in srows if row.get("connectivity_id")}),
                "spectrum_count": len(arows),
            }
        )
    return out


def count_by_spectrum_key(gold_spectra: list[dict[str, Any]], key: str, output_key: str) -> list[dict[str, Any]]:
    spectra_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in gold_spectra:
        value = "" if row.get(key) is None else str(row.get(key))
        spectra_by_key[value].append(row)
    out = []
    for value, rows in sorted(spectra_by_key.items()):
        out.append(
            {
                output_key: value,
                "structure_count": len({row["structure_record_id"] for row in rows}),
                "connectivity_count": len({row["connectivity_id"] for row in rows if row.get("connectivity_id")}),
                "spectrum_count": len(rows),
            }
        )
    return out


def gold_counts(gold_structures: list[dict[str, Any]], gold_spectra: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    structure_by_id = {row["structure_record_id"]: row for row in gold_structures}
    spectra_with_linkage = []
    for row in gold_spectra:
        structure = structure_by_id[row["structure_record_id"]]
        spectra_with_linkage.append({**row, "linkage_bucket": linkage_bucket(structure.get("linkage_pattern") or [])})
    structures_with_linkage = [{**row, "linkage_bucket": linkage_bucket(row.get("linkage_pattern") or [])} for row in gold_structures]
    return {
        "gold_counts_by_class.csv": count_by_structure_key(gold_structures, gold_spectra, "gold_lipid_class", "gold_lipid_class", "gold_lipid_class"),
        "gold_counts_by_linkage.csv": count_by_structure_key(structures_with_linkage, spectra_with_linkage, "linkage_bucket", "linkage_bucket", "linkage_bucket"),
        "gold_counts_by_source.csv": count_by_spectrum_key(gold_spectra, "source", "source"),
        "gold_counts_by_adduct.csv": count_by_spectrum_key(gold_spectra, "adduct", "adduct"),
        "gold_counts_by_polarity.csv": count_by_spectrum_key(gold_spectra, "polarity", "polarity"),
    }


def draw_relabel_figures(corrections: list[dict[str, Any]], rows_by_id: dict[str, dict[str, Any]], figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    for correction in corrections:
        row = rows_by_id[correction["structure_record_id"]]
        result = row["parser_result"]
        smiles = result.get("input_smiles")
        mol = Chem.MolFromSmiles(smiles or "")
        if mol is None:
            continue
        atom_colors: dict[int, tuple[float, float, float]] = {}
        for idx in result.get("headgroup_atom_indices") or []:
            atom_colors[idx] = (0.25, 0.45, 0.95)
        for idx in result.get("backbone_atom_indices") or []:
            atom_colors[idx] = (1.0, 0.55, 0.15)
        for chain in result.get("chains") or []:
            for idx in chain.get("chain_partition_atom_indices") or []:
                atom_colors[idx] = (0.20, 0.65, 0.35)
        for site in result.get("free_hydroxyl_atom_indices") or []:
            atom_colors[site] = (0.95, 0.20, 0.25)
        drawer = Draw.MolDraw2DCairo(1100, 780)
        drawer.drawOptions().addAtomIndices = True
        drawer.DrawMolecule(mol, highlightAtoms=sorted(atom_colors), highlightAtomColors=atom_colors)
        drawer.FinishDrawing()
        (figure_dir / f"{correction['structure_record_id']}.png").write_bytes(drawer.GetDrawingText())


def extension_schema_markdown() -> str:
    return """# Gold Extension Schema

Annotation tiers:

- `parser_validated_strict_agreement`: source strict label agrees with parser v0.1 graph partition and exact reconstruction.
- `user_confirmed_relabel`: a user reviewed a proposed label correction and approved it.
- `reference_standard_confirmed`: a reference standard establishes the structure/class identity.
- `sample_high_confidence`: a real sample is supported by high-confidence evidence but is not a reference standard.
- `reference_only`: useful as a structural reference but not an experimental training label.

Future real-sample additions must preserve these fields when the information exists:

- `sample_id`
- `evidence_type`
- `manual_reviewer`
- `review_date`
- `supporting_spectra`
- `annotation_notes`

Do not fabricate missing provenance. Rows produced in Gold Curation v0.1 use
`parser_validated_strict_agreement` and keep future sample-provenance fields null.
"""


def review_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Gold Curation v0.1 User Review Summary",
        "",
        f"- Strict structures indexed: {summary['strict_structures']}",
        f"- Gold structures: {summary['gold_structure_count']}",
        f"- Gold spectra/acquisitions: {summary['gold_spectrum_count']}",
        f"- Label correction candidates: {summary['label_correction_count']}",
        f"- Connectivity conflicts: {summary['connectivity_conflict_count']}",
        "",
        "User-editable review file:",
        "",
        "- `label_review_decisions_template.csv`",
        "",
        "Review figures:",
        "",
        "- `figures/<structure_record_id>.png`",
        "",
        "The proposed corrections are review candidates only. They are not written back to strict or v2 data.",
    ]
    return "\n".join(lines) + "\n"


def validate_baseline(summary: dict[str, Any]) -> None:
    problems = []
    for key, expected in EXPECTED_BASELINE.items():
        if summary.get(key) != expected:
            problems.append(f"{key}: expected {expected}, observed {summary.get(key)}")
    if problems:
        raise SystemExit("Baseline changed; refusing to write gold outputs:\n" + "\n".join(problems))


def write_gold_outputs(
    directory: Path,
    summary: dict[str, Any],
    index_rows: list[dict[str, Any]],
    gold_structures: list[dict[str, Any]],
    gold_spectra: list[dict[str, Any]],
    corrections: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
    parser_scope_rows: list[dict[str, Any]],
    invalid_rows: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
    counts: dict[str, list[dict[str, Any]]],
    make_figures: bool,
    rows_by_id: dict[str, dict[str, Any]],
) -> None:
    write_jsonl(directory / "all_strict_curation_index_v0_1.jsonl", index_rows)
    write_jsonl(directory / "gold_structures_v0_1.jsonl", gold_structures)
    write_jsonl(directory / "gold_spectra_v0_1.jsonl", gold_spectra)
    write_jsonl(directory / "label_corrections_proposed_v0_1.jsonl", corrections)
    write_csv(
        directory / "label_review_decisions_template.csv",
        review_rows,
        [
            "structure_record_id",
            "connectivity_id",
            "current_label",
            "proposed_label",
            "provided_name",
            "canonical_smiles",
            "chain_count",
            "linkage_pattern",
            "spectrum_count",
            "decision",
            "user_label",
            "user_note",
        ],
    )
    write_jsonl(directory / "parser_scope_exclusions_v0_1.jsonl", parser_scope_rows)
    write_jsonl(directory / "invalid_or_incomplete_v0_1.jsonl", invalid_rows)
    write_jsonl(directory / "connectivity_layer_conflicts.jsonl", conflicts)
    write_csv(directory / "gold_counts_by_class.csv", counts["gold_counts_by_class.csv"], ["gold_lipid_class", "structure_count", "connectivity_count", "spectrum_count"])
    write_csv(directory / "gold_counts_by_linkage.csv", counts["gold_counts_by_linkage.csv"], ["linkage_bucket", "structure_count", "connectivity_count", "spectrum_count"])
    write_csv(directory / "gold_counts_by_source.csv", counts["gold_counts_by_source.csv"], ["source", "structure_count", "connectivity_count", "spectrum_count"])
    write_csv(directory / "gold_counts_by_adduct.csv", counts["gold_counts_by_adduct.csv"], ["adduct", "structure_count", "connectivity_count", "spectrum_count"])
    write_csv(directory / "gold_counts_by_polarity.csv", counts["gold_counts_by_polarity.csv"], ["polarity", "structure_count", "connectivity_count", "spectrum_count"])
    (directory / "gold_extension_schema.md").write_text(extension_schema_markdown(), encoding="utf-8")
    (directory / "user_review_summary.md").write_text(review_summary_markdown(summary), encoding="utf-8")
    (directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if make_figures:
        draw_relabel_figures(corrections, rows_by_id, directory / "figures")


def build(
    out_dir: Path,
    audit_dir: Path,
    make_figures: bool = True,
    quiet: bool = False,
    strict_rows_override: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    require_inputs()
    strict_rows = strict_rows_override if strict_rows_override is not None else read_jsonl(STRICT_PATH)
    molecule_by_id = molecule_index_by_structure(MOLECULE_INDEX_PATH)
    strict_by_id = group_strict_rows(strict_rows)
    records = unique_strict_records(strict_rows)
    results = [evaluate_record(record) for record in records]
    enrich_result_rows(results, strict_by_id, molecule_by_id)
    index_rows = apply_curation_layers(results)
    index_by_id = {row["structure_record_id"]: row for row in index_rows}
    rows_by_id = {row["structure_record_id"]: row for row in results}
    conflicts = connectivity_conflicts(index_rows)

    gold_results = [rows_by_id[row["structure_record_id"]] for row in index_rows if row["curation_layer"] == "gold_v0_1"]
    gold_structures = [gold_structure_row(row) for row in gold_results]
    gold_by_structure = {row["structure_record_id"]: row for row in gold_structures}
    gold_spectra = spectrum_gold_rows(strict_rows, gold_by_structure)
    corrections = relabel_rows(results, index_by_id)
    review_rows = review_template_rows(corrections)
    parser_scope_rows = [
        compact_non_gold_row(rows_by_id[row["structure_record_id"]], index_by_id)
        for row in index_rows
        if row["curation_layer"] in {"parser_scope_exclusion", "class_out_of_scope"}
    ]
    invalid_rows = [
        compact_non_gold_row(rows_by_id[row["structure_record_id"]], index_by_id)
        for row in index_rows
        if row["curation_layer"] == "invalid_or_incomplete_structure"
    ]
    layer_counts = Counter(row["curation_layer"] for row in index_rows)
    correction_counts = Counter(f"{row['current_label']}->{row['proposed_label']}" for row in corrections)
    baseline_summary = {
        "strict_structures": len(results),
        "graph_parse_success_reconstruction_exact": sum(
            1
            for row in results
            if row.get("parse_status") == "success" and row["parser_result"].get("reconstruction_connectivity_exact")
        ),
        "gold_v0_1": layer_counts.get("gold_v0_1", 0),
        "label_disagreement_candidates": layer_counts.get("relabel_review", 0),
        "parser_failure": sum(1 for row in index_rows if row["curation_layer"] == "parser_failure"),
    }
    validate_baseline(baseline_summary)
    summary = {
        "gold_version": "0.1",
        **baseline_summary,
        "curation_layer_counts": dict(sorted(layer_counts.items())),
        "layer_count_sum": sum(layer_counts.values()),
        "gold_structure_count": len(gold_structures),
        "gold_connectivity_count": len({row["connectivity_id"] for row in gold_structures if row.get("connectivity_id")}),
        "gold_spectrum_count": len(gold_spectra),
        "label_correction_count": len(corrections),
        "label_correction_counts": dict(sorted(correction_counts.items())),
        "parser_scope_exclusion_count": layer_counts.get("parser_scope_exclusion", 0),
        "class_out_of_scope_count": layer_counts.get("class_out_of_scope", 0),
        "invalid_or_incomplete_count": layer_counts.get("invalid_or_incomplete_structure", 0),
        "connectivity_conflict_count": len(conflicts),
        "persistent_output_dir": str(out_dir),
        "audit_output_dir": str(audit_dir),
    }

    counts = gold_counts(gold_structures, gold_spectra)
    targets: list[Path] = []
    for target in [out_dir, audit_dir]:
        resolved_target = target.resolve()
        if resolved_target not in [item.resolve() for item in targets]:
            targets.append(resolved_target)
    staged: list[tuple[Path, Path]] = []
    try:
        for target in targets:
            staging = safe_staging_dir(target)
            staged.append((staging, target))
            write_gold_outputs(
                staging,
                summary,
                index_rows,
                gold_structures,
                gold_spectra,
                corrections,
                review_rows,
                parser_scope_rows,
                invalid_rows,
                conflicts,
                counts,
                make_figures and target.resolve() == audit_dir.resolve(),
                rows_by_id,
            )
        for staging, target in staged:
            replace_output_dir(staging, target)
    finally:
        for staging, _target in staged:
            if staging.exists():
                shutil.rmtree(staging)
    if not quiet:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Gold Curation v0.1 outputs for parser-validated glycerophospholipids.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    build(args.out_dir, args.audit_dir, make_figures=not args.no_figures, quiet=args.quiet)


if __name__ == "__main__":
    main()
