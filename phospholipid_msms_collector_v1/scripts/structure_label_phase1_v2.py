#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from rdkit import Chem, RDLogger, rdBase

from collect_phospholipid_msms import parse_formula, stable_hash, standardize_structure_identity

RDLogger.DisableLog("rdApp.*")

SOURCE_PAGE = "https://www.lipidmaps.org/databases/lmsd/download"
CLASS_TERMS = {
    "PC": ("glycerophosphocholine", "phosphocholine", " pc ", "lpc"),
    "LPC": ("monoacylglycerophosphocholine", "lysophosphatidylcholine", " lpc ", " lyso-pc"),
    "PE": ("glycerophosphoethanolamine", "phosphoethanolamine", " pe ", "lpe"),
    "LPE": ("monoacylglycerophosphoethanolamine", "lysophosphatidylethanolamine", " lpe "),
    "PI": ("glycerophosphoinositol", "phosphatidylinositol", " pi "),
    "PG": ("glycerophosphoglycerol", "phosphatidylglycerol", " pg "),
    "LPG": ("lysophosphatidylglycerol", " lpg "),
    "PS": ("glycerophosphoserine", "phosphatidylserine", " ps "),
    "PA": ("glycerophosphate", "phosphatidic acid", " pa "),
    "LPA": ("lysophosphatidic acid", " lpa "),
    "SM": ("sphingomyelin", "ceramide phosphocholine", " sm("),
    "LysoSM": ("lysosphingomyelin", "lyso-sm"),
    "S1P": ("sphingosine-1-phosphate", "sphingosine 1-phosphate", " s1p"),
}
SUSPICIOUS = re.compile(r"\b(?:CoA|coenzyme|ATP|ADP|AMP|GTP|GDP|GMP|nucleotide|oligonucleotide|DNA|RNA|diazinon|malaoxon)\b", re.I)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def formula_key(value: str | None) -> str | None:
    counts = parse_formula(value)
    if not counts:
        return None
    return "".join(f"{element}{counts[element] if counts[element] != 1 else ''}" for element in sorted(counts))


def first_text(*values: Any) -> str | None:
    for value in values:
        if value:
            return str(value)
    return None


def lmsd_text(ref: dict[str, Any]) -> str:
    keys = ["LM_ID", "NAME", "ABBREVIATION", "SYNONYMS", "CATEGORY", "MAIN_CLASS", "SUB_CLASS", "CLASS_LEVEL4"]
    return " ".join(str(ref.get(k) or "") for k in keys).lower()


def class_consistency(classes: Iterable[str], refs: list[dict[str, Any]]) -> dict[str, bool]:
    text = (" " + " ".join(lmsd_text(ref) for ref in refs) + " ").lower()
    out = {}
    for cls in sorted({c for c in classes if c and c != "P-lipid-unresolved"}):
        terms = CLASS_TERMS.get(cls, (cls.lower(),))
        out[cls] = any(term.lower() in text for term in terms)
    return out


def ref_from_mol(
    mol: Any,
    props: dict[str, Any],
    tsv_row: dict[str, str] | None,
    source_file: str,
    source_license: str,
) -> dict[str, Any] | None:
    lm_id = first_text(props.get("LM_ID"), props.get("LMID"), (tsv_row or {}).get("lm_id"))
    smiles = first_text(props.get("SMILES"), (tsv_row or {}).get("smiles"))
    if mol is None and smiles:
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    inchikey = first_text(props.get("INCHI_KEY"), props.get("INCHIKEY"), (tsv_row or {}).get("inchi_key"))
    if not inchikey:
        try:
            inchikey = Chem.MolToInchiKey(mol)
        except Exception:
            inchikey = None
    try:
        canonical_iso = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        canonical_noniso = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        formal_charge = Chem.GetFormalCharge(mol)
    except Exception:
        canonical_iso = canonical_noniso = None
        formal_charge = None
    ref = {
        "LM_ID": lm_id,
        "INCHI_KEY": inchikey,
        "SMILES": smiles,
        "NAME": first_text(props.get("COMMON_NAME"), props.get("NAME"), props.get("SYSTEMATIC_NAME"), props.get("ABBREVIATION")),
        "ABBREVIATION": first_text(props.get("ABBREVIATION")),
        "SYNONYMS": first_text(props.get("SYNONYMS")),
        "CATEGORY": first_text(props.get("CATEGORY")),
        "MAIN_CLASS": first_text(props.get("MAIN_CLASS")),
        "SUB_CLASS": first_text(props.get("SUB_CLASS")),
        "CLASS_LEVEL4": first_text(props.get("CLASS_LEVEL4")),
        "FORMULA": first_text(props.get("FORMULA")),
        "INCHI": first_text(props.get("INCHI")),
        "lmsd_canonical_isomeric_smiles": canonical_iso,
        "lmsd_canonical_nonisomeric_smiles": canonical_noniso,
        "lmsd_connectivity_key": inchikey.split("-", 1)[0] if inchikey and "-" in inchikey else None,
        "lmsd_formula_key": formula_key(first_text(props.get("FORMULA"))),
        "lmsd_formal_charge": formal_charge,
        "source_database": "LIPID MAPS LMSD",
        "source_file": source_file,
        "source_license": source_license,
        "source_page": SOURCE_PAGE,
        "from_sdf": source_file.endswith(".sdf.zip"),
        "from_tsv": bool(tsv_row),
    }
    return ref if ref["LM_ID"] else None


def load_lmsd(reference_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = reference_dir / "raw"
    tsv_path = raw / "lipidmaps_ids_cc0.tsv"
    sdf_zip = raw / "LMSD_extended.sdf.zip"
    tsv_rows: dict[str, dict[str, str]] = {}
    with tsv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("lm_id"):
                tsv_rows[row["lm_id"]] = row

    refs_by_id: dict[str, dict[str, Any]] = {}
    records_seen = 0
    records_none = 0
    sdf_member = None
    with zipfile.ZipFile(sdf_zip) as zf:
        sdf_member = next(name for name in zf.namelist() if name.lower().endswith(".sdf"))
        data = zf.read(sdf_member)
    supplier = Chem.ForwardSDMolSupplier(io.BytesIO(data), sanitize=True, removeHs=False)
    for mol in supplier:
        records_seen += 1
        if mol is None:
            records_none += 1
            continue
        props = {name: mol.GetProp(name) for name in mol.GetPropNames()}
        lm_id = first_text(props.get("LM_ID"), props.get("LMID"))
        ref = ref_from_mol(mol, props, tsv_rows.get(lm_id or ""), "LMSD_extended.sdf.zip", "CC BY 4.0")
        if ref:
            refs_by_id[ref["LM_ID"]] = ref
    for lm_id, row in tsv_rows.items():
        if lm_id not in refs_by_id:
            ref = ref_from_mol(None, {"LM_ID": lm_id}, row, "lipidmaps_ids_cc0.tsv", "CC0")
            if ref:
                refs_by_id[lm_id] = ref

    manifest_path = reference_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    parse_summary = {
        "manifest": manifest,
        "sdf": {"records_seen": records_seen, "records_none": records_none, "sdf_member": sdf_member},
        "tsv": {"rows_seen": len(tsv_rows)},
        "total_lmsd_records": len(refs_by_id),
    }
    return list(refs_by_id.values()), parse_summary


def build_lmsd_indexes(refs: list[dict[str, Any]]) -> dict[str, dict[Any, list[dict[str, Any]]]]:
    indexes: dict[str, dict[Any, list[dict[str, Any]]]] = {
        "full": defaultdict(list),
        "iso": defaultdict(list),
        "conn_formula": defaultdict(list),
        "conn": defaultdict(list),
    }
    for ref in refs:
        if ref.get("INCHI_KEY"):
            indexes["full"][ref["INCHI_KEY"]].append(ref)
        if ref.get("lmsd_canonical_isomeric_smiles"):
            indexes["iso"][ref["lmsd_canonical_isomeric_smiles"]].append(ref)
        if ref.get("lmsd_connectivity_key"):
            indexes["conn"][ref["lmsd_connectivity_key"]].append(ref)
            if ref.get("lmsd_formula_key"):
                indexes["conn_formula"][(ref["lmsd_connectivity_key"], ref["lmsd_formula_key"])].append(ref)
    return indexes


def build_structures(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        grouped[rec["structure_record_id"]].append(rec)
    structures = []
    spectrum_map = []
    for structure_record_id, items in sorted(grouped.items()):
        first = items[0]
        strict_count = sum(1 for r in items if r["lipid_class"] != "P-lipid-unresolved")
        unresolved_count = len(items) - strict_count
        row = {
            "molecule_id": structure_record_id,
            "structure_record_id": structure_record_id,
            "connectivity_id": first.get("connectivity_id"),
            "standardization_version": first.get("standardization_version"),
            "rdkit_version": first.get("rdkit_version"),
            "identity_fallback": first.get("identity_fallback"),
            "identity_warnings": sorted({w for r in items for w in r.get("identity_warnings", [])}),
            "raw_smiles_values": sorted({r.get("smiles") or "" for r in items if r.get("smiles")}),
            "canonical_isomeric_smiles": first.get("canonical_isomeric_smiles"),
            "canonical_nonisomeric_smiles": first.get("canonical_nonisomeric_smiles"),
            "major_component_smiles": first.get("major_component_smiles"),
            "fragment_count": first.get("fragment_count"),
            "all_fragment_smiles": first.get("all_fragment_smiles"),
            "minor_fragment_smiles": first.get("minor_fragment_smiles"),
            "fragment_signature": first.get("fragment_signature"),
            "has_multiple_fragments": first.get("has_multiple_fragments"),
            "likely_counterions": first.get("likely_counterions"),
            "rdkit_full_inchikey": first.get("rdkit_full_inchikey"),
            "rdkit_connectivity_key": first.get("rdkit_connectivity_key"),
            "provided_full_inchikey_values": sorted({r.get("provided_full_inchikey") for r in items if r.get("provided_full_inchikey")}),
            "provided_connectivity_key_values": sorted({r.get("provided_connectivity_key") for r in items if r.get("provided_connectivity_key")}),
            "provided_identifier_raw_values": sorted({r.get("provided_identifier_raw") for r in items if r.get("provided_identifier_raw")}),
            "provided_formula_values": sorted({r.get("formula") for r in items if r.get("formula")}),
            "formula_keys": sorted({formula_key(r.get("formula")) for r in items if formula_key(r.get("formula"))}),
            "sources": sorted({r.get("source") for r in items if r.get("source")}),
            "licenses": sorted({r.get("license") for r in items if r.get("license")}),
            "lipid_class_values": sorted({r.get("lipid_class") for r in items if r.get("lipid_class")}),
            "adducts": sorted({r.get("adduct") or "unknown" for r in items}),
            "polarities": sorted({r.get("polarity") or "unknown" for r in items}),
            "collision_energies": sorted({str(r.get("collision_energy_raw")) for r in items if r.get("collision_energy_raw")}),
            "instruments": sorted({str(r.get("instrument") or r.get("instrument_type")) for r in items if r.get("instrument") or r.get("instrument_type")}),
            "names": sorted({name for r in items for name in ([r.get("name")] + (r.get("all_names") or [])) if name}),
            "record_titles": sorted({r.get("record_title") for r in items if r.get("record_title")}),
            "spectrum_count": len(items),
            "strict_spectrum_count": strict_count,
            "unresolved_spectrum_count": unresolved_count,
            "spectrum_ids_sample": [r.get("spectrum_id") for r in items[:20]],
        }
        structures.append(row)
        for rec in items:
            spectrum_map.append({
                "spectrum_id": rec.get("spectrum_id"),
                "source": rec.get("source"),
                "source_record_id": rec.get("source_record_id"),
                "source_file": rec.get("source_file"),
                "source_release": rec.get("source_release"),
                "license": rec.get("license"),
                "structure_record_id": structure_record_id,
                "connectivity_id": rec.get("connectivity_id"),
                "lipid_class": rec.get("lipid_class"),
                "peak_identity_hash": rec.get("peak_identity_hash"),
                "acquisition_record_hash": rec.get("acquisition_record_hash"),
                "duplicate_group_id": rec.get("duplicate_group_id"),
                "duplicate_relation": rec.get("duplicate_relation"),
            })
    return structures, spectrum_map


def match_structure(struct: dict[str, Any], indexes: dict[str, dict[Any, list[dict[str, Any]]]]) -> dict[str, Any] | None:
    full_keys = list(dict.fromkeys(struct.get("provided_full_inchikey_values", []) + [struct.get("rdkit_full_inchikey")]))
    full_keys = [key for key in full_keys if key]
    for key in full_keys:
        refs = indexes["full"].get(key)
        if refs:
            return make_match(struct, refs, "A", "full_inchikey_exact")
    iso = struct.get("canonical_isomeric_smiles")
    if iso and indexes["iso"].get(iso):
        return make_match(struct, indexes["iso"][iso], "B", "canonical_isomeric_smiles_exact")
    conn_keys = list(dict.fromkeys(struct.get("provided_connectivity_key_values", []) + [struct.get("rdkit_connectivity_key")]))
    conn_keys = [key for key in conn_keys if key]
    for conn in conn_keys:
        for fkey in struct.get("formula_keys", []):
            refs = indexes["conn_formula"].get((conn, fkey))
            if refs:
                return make_match(struct, refs, "C", "connectivity_key_plus_formula")
    for conn in conn_keys:
        refs = indexes["conn"].get(conn)
        if refs:
            return make_match(struct, refs, "D", "connectivity_key_only")
    return None


def make_match(struct: dict[str, Any], refs: list[dict[str, Any]], tier: str, method: str) -> dict[str, Any]:
    formulas = {ref.get("lmsd_formula_key") for ref in refs if ref.get("lmsd_formula_key")}
    query_formulas = set(struct.get("formula_keys") or [])
    exact = tier in {"A", "B"}
    return {
        "molecule_id": struct["molecule_id"],
        "structure_record_id": struct["structure_record_id"],
        "connectivity_id": struct.get("connectivity_id"),
        "match_tier": tier,
        "match_method": method,
        "candidate_count": len(refs),
        "lm_ids": [ref.get("LM_ID") for ref in refs],
        "lmsd_formulas": sorted(formulas),
        "query_formulas": sorted(query_formulas),
        "formula_consistent": bool(formulas & query_formulas) if formulas and query_formulas else None,
        "exact_reference_match": exact,
        "connectivity_only_match": tier == "D",
        "stereochemistry_warning": tier in {"C", "D"},
        "charge_warning": tier == "D",
        "match_warnings": [] if tier != "D" else ["connectivity-only match may hide formula, charge, salt, or stereochemistry differences"],
        "lmsd_records": refs,
    }


def audit_false_positive(struct: dict[str, Any], match: dict[str, Any] | None) -> dict[str, Any] | None:
    flags = []
    name_text = " ".join(struct.get("names", []) + struct.get("record_titles", []))
    if SUSPICIOUS.search(name_text):
        flags.append("source_name_suspicious_non_phospholipid")
    if match and match.get("match_tier") == "D":
        flags.append("connectivity_only_lmsd_match")
    if match and struct["strict_spectrum_count"]:
        consistency = class_consistency(struct.get("lipid_class_values", []), match["lmsd_records"])
        if consistency and not any(consistency.values()):
            flags.append("strict_class_not_seen_in_lmsd_text")
    if not flags:
        return None
    return {
        "molecule_id": struct["molecule_id"],
        "structure_record_id": struct["structure_record_id"],
        "flags": flags,
        "names": struct.get("names", [])[:10],
        "lipid_class_values": struct.get("lipid_class_values", []),
        "match_tier": match.get("match_tier") if match else None,
        "lm_ids": match.get("lm_ids") if match else [],
        "spectrum_count": struct.get("spectrum_count"),
    }


def sample_audits(structures: list[dict[str, Any]]) -> dict[str, Any]:
    by_conn: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for struct in structures:
        by_conn[struct.get("connectivity_id")].append(struct)
    same_conn_diff_structure = []
    multi_spectrum_connectivity = []
    for conn, items in sorted(by_conn.items(), key=lambda kv: (-len(kv[1]), str(kv[0]))):
        if len(items) > 1 and len(same_conn_diff_structure) < 20:
            same_conn_diff_structure.append({
                "connectivity_id": conn,
                "structure_record_ids": [x["structure_record_id"] for x in items[:10]],
                "structure_count": len(items),
            })
        total_spectra = sum(x.get("spectrum_count", 0) for x in items)
        if total_spectra > 1 and len(multi_spectrum_connectivity) < 20:
            multi_spectrum_connectivity.append({
                "connectivity_id": conn,
                "structure_count": len(items),
                "spectrum_count": total_spectra,
                "structure_record_ids": [x["structure_record_id"] for x in items[:10]],
            })
    multi_fragment = [x for x in structures if x.get("has_multiple_fragments")]
    invalid = [x for x in structures if x.get("identity_fallback")]
    return {
        "same_connectivity_different_structure_sample": same_conn_diff_structure,
        "multi_spectrum_connectivity_sample": multi_spectrum_connectivity,
        "all_multi_fragment_structures": [
            {
                "structure_record_id": x["structure_record_id"],
                "connectivity_id": x.get("connectivity_id"),
                "all_fragment_smiles": x.get("all_fragment_smiles"),
                "likely_counterions": x.get("likely_counterions"),
                "spectrum_count": x.get("spectrum_count"),
            }
            for x in multi_fragment
        ],
        "all_rdkit_invalid_structures": [
            {
                "structure_record_id": x["structure_record_id"],
                "connectivity_id": x.get("connectivity_id"),
                "raw_smiles_values": x.get("raw_smiles_values"),
                "identity_warnings": x.get("identity_warnings"),
                "spectrum_count": x.get("spectrum_count"),
            }
            for x in invalid
        ],
    }


def stability_check(records: list[dict[str, Any]], limit: int = 500) -> dict[str, Any]:
    checked = 0
    failures = []
    for rec in records[:limit]:
        first = standardize_structure_identity(rec.get("smiles"))
        second = standardize_structure_identity(rec.get("smiles"))
        checked += 1
        if (
            first.get("structure_record_id") != second.get("structure_record_id")
            or first.get("connectivity_id") != second.get("connectivity_id")
        ):
            failures.append({"spectrum_id": rec.get("spectrum_id"), "smiles": rec.get("smiles")})
            if len(failures) >= 10:
                break
    return {"checked_records": checked, "stable": not failures, "failures": failures}


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 1 v2 structure standardization and LIPID MAPS matching")
    ap.add_argument("--input", default="data/expanded_phospholipids_v2/phospholipid_msms_all_v2.jsonl")
    ap.add_argument("--out", default="data/structure_labeling/phase1_v2")
    ap.add_argument("--reference", default="data/reference/lipidmaps")
    args = ap.parse_args()

    input_path = Path(args.input)
    out = Path(args.out)
    ref_dir = Path(args.reference)
    out.mkdir(parents=True, exist_ok=True)

    before_hash = sha256_file(input_path)
    records = read_jsonl(input_path)
    refs, lmsd_parse = load_lmsd(ref_dir)
    indexes = build_lmsd_indexes(refs)
    structures, spectrum_map = build_structures(records)

    matches = []
    unmatched = []
    strict_rows = []
    false_positive_rows = []
    raw_condition_counts = Counter()
    for struct in structures:
        full_hit = any(indexes["full"].get(k) for k in struct.get("provided_full_inchikey_values", []) + [struct.get("rdkit_full_inchikey")])
        iso_hit = bool(struct.get("canonical_isomeric_smiles") and indexes["iso"].get(struct["canonical_isomeric_smiles"]))
        conn_keys = struct.get("provided_connectivity_key_values", []) + [struct.get("rdkit_connectivity_key")]
        formula_hits = any(indexes["conn_formula"].get((conn, fkey)) for conn in conn_keys if conn for fkey in struct.get("formula_keys", []))
        conn_hits = any(indexes["conn"].get(conn) for conn in conn_keys if conn)
        raw_condition_counts.update({
            "full_inchikey_exact": int(bool(full_hit)),
            "canonical_isomeric_smiles_exact": int(iso_hit),
            "connectivity_key_plus_formula": int(formula_hits),
            "connectivity_key_only": int(conn_hits),
        })

        match = match_structure(struct, indexes)
        if match:
            matches.append(match)
            if struct["strict_spectrum_count"]:
                consistency = class_consistency(struct.get("lipid_class_values", []), match["lmsd_records"])
                strict_rows.append({
                    "molecule_id": struct["molecule_id"],
                    "structure_record_id": struct["structure_record_id"],
                    "strict_class_values": [x for x in struct.get("lipid_class_values", []) if x != "P-lipid-unresolved"],
                    "strict_spectrum_count": struct["strict_spectrum_count"],
                    "match_tier": match["match_tier"],
                    "match_method": match["match_method"],
                    "candidate_count": match["candidate_count"],
                    "lm_ids": match["lm_ids"],
                    "exact_reference_match": match["exact_reference_match"],
                    "connectivity_only_match": match["connectivity_only_match"],
                    "class_consistency": consistency,
                    "class_conflict": bool(consistency) and not any(consistency.values()),
                    "lmsd_text_excerpt": " ".join(lmsd_text(ref) for ref in match["lmsd_records"])[:600],
                })
        else:
            unmatched.append({
                "molecule_id": struct["molecule_id"],
                "structure_record_id": struct["structure_record_id"],
                "connectivity_id": struct.get("connectivity_id"),
                "rdkit_full_inchikey": struct.get("rdkit_full_inchikey"),
                "rdkit_connectivity_key": struct.get("rdkit_connectivity_key"),
                "provided_connectivity_key_values": struct.get("provided_connectivity_key_values"),
                "formula_keys": struct.get("formula_keys"),
                "spectrum_count": struct.get("spectrum_count"),
                "lipid_class_values": struct.get("lipid_class_values"),
            })
        audit = audit_false_positive(struct, match)
        if audit:
            false_positive_rows.append(audit)

    invalid = [x for x in structures if x.get("identity_fallback")]
    strict_total = sum(1 for x in structures if x["strict_spectrum_count"])
    tier_counts = Counter(m["match_tier"] for m in matches)
    strict_conflicts = [x for x in strict_rows if x["class_conflict"]]
    exact_matches = [m for m in matches if m["exact_reference_match"]]
    v1_summary_path = Path("data/structure_labeling/phase1/summary.json")
    v1_summary = json.loads(v1_summary_path.read_text(encoding="utf-8")) if v1_summary_path.exists() else None
    comparison = {
        "v1_available": v1_summary is not None,
        "v1_unique_molecules": (v1_summary or {}).get("molecule_counts", {}).get("unique_molecules"),
        "v2_unique_structures": len(structures),
        "v1_matched_molecules": (v1_summary or {}).get("matching", {}).get("matched_molecules"),
        "v2_matched_structures": len(matches),
        "v1_unmatched_molecules": (v1_summary or {}).get("matching", {}).get("unmatched_molecules"),
        "v2_unmatched_structures": len(unmatched),
        "v1_tier_counts": (v1_summary or {}).get("matching", {}).get("tier_counts"),
        "v2_tier_counts": dict(tier_counts),
        "v1_strict_unique_molecules": (v1_summary or {}).get("strict_calibration", {}).get("strict_unique_molecules"),
        "v2_strict_unique_structures": strict_total,
        "v2_matched_strict_structures": len(strict_rows),
    }

    summary = {
        "conda_env": "lipidforge-chem",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python_executable": str(Path(__import__("sys").executable).resolve()),
        "rdkit_version": rdBase.rdkitVersion,
        "input_hash_before": {str(input_path): before_hash},
        "input_hash_after": {str(input_path): sha256_file(input_path)},
        "input_hash_unchanged": before_hash == sha256_file(input_path),
        "input_records": {
            "total": len(records),
            "strict": sum(1 for r in records if r["lipid_class"] != "P-lipid-unresolved"),
            "unresolved": sum(1 for r in records if r["lipid_class"] == "P-lipid-unresolved"),
        },
        "lmsd_parse": lmsd_parse,
        "molecule_counts": {
            "unique_structures": len(structures),
            "unique_connectivity_ids": len({x.get("connectivity_id") for x in structures}),
            "multiple_fragments": sum(1 for x in structures if x.get("has_multiple_fragments")),
            "rdkit_invalid_structures": len(invalid),
            "with_strict_records": sum(1 for x in structures if x["strict_spectrum_count"]),
            "with_unresolved_records": sum(1 for x in structures if x["unresolved_spectrum_count"]),
            "provided_full_inchikey_connectivity_mismatch_records": sum(
                1 for r in records
                if r.get("provided_full_inchikey") and r.get("rdkit_connectivity_key")
                and r["provided_full_inchikey"].split("-", 1)[0] != r["rdkit_connectivity_key"]
            ),
            "provided_connectivity_key_mismatch_records": sum(
                1 for r in records
                if r.get("provided_connectivity_key") and r.get("rdkit_connectivity_key")
                and r["provided_connectivity_key"] != r["rdkit_connectivity_key"]
            ),
        },
        "matching": {
            "matched_structures": len(matches),
            "unmatched_structures": len(unmatched),
            "exact_reference_matched_structures": len(exact_matches),
            "tier_counts": dict(tier_counts),
            "raw_condition_counts": dict(raw_condition_counts),
        },
        "strict_calibration": {
            "strict_unique_structures": strict_total,
            "matched_strict_structures": len(strict_rows),
            "exact_reference_matches": sum(1 for x in strict_rows if x["exact_reference_match"]),
            "connectivity_only_matches": sum(1 for x in strict_rows if x["connectivity_only_match"]),
            "class_consistent_structures": sum(1 for x in strict_rows if any(x["class_consistency"].values())),
            "class_conflicts": len(strict_conflicts),
            "conflict_examples": strict_conflicts[:20],
        },
        "suspicious_false_positives": {
            "flagged_structures": len(false_positive_rows),
            "flag_counts": dict(Counter(flag for row in false_positive_rows for flag in row["flags"])),
            "examples": false_positive_rows[:20],
        },
        "identity_consistency": {
            "massspecgym_14_char_identifiers_are_connectivity_blocks": True,
            "full_inchikey_connectivity_mismatches": sum(
                1 for r in records
                if r.get("provided_full_inchikey") and r.get("rdkit_connectivity_key")
                and r["provided_full_inchikey"].split("-", 1)[0] != r["rdkit_connectivity_key"]
            ),
            "connectivity_key_mismatches": sum(
                1 for r in records
                if r.get("provided_connectivity_key") and r.get("rdkit_connectivity_key")
                and r["provided_connectivity_key"] != r["rdkit_connectivity_key"]
            ),
            "stable_id_check": stability_check(records),
            "sample_audits": sample_audits(structures),
        },
        "v1_phase1_vs_v2_phase1": comparison,
        "outputs": {
            "molecule_index": "molecule_index.jsonl",
            "spectrum_to_structure": "spectrum_to_structure.jsonl",
            "rdkit_invalid_structures": "rdkit_invalid_structures.jsonl",
            "lipidmaps_matches": "lipidmaps_matches.jsonl",
            "lipidmaps_unmatched": "lipidmaps_unmatched.jsonl",
            "strict_calibration": "strict_calibration.jsonl",
            "possible_false_positives": "possible_false_positives.jsonl",
            "summary": "summary.json",
            "v1_phase1_vs_v2_phase1": "v1_phase1_vs_v2_phase1.json",
        },
    }

    write_jsonl(out / "molecule_index.jsonl", structures)
    write_jsonl(out / "spectrum_to_structure.jsonl", spectrum_map)
    write_jsonl(out / "rdkit_invalid_structures.jsonl", invalid)
    write_jsonl(out / "lipidmaps_matches.jsonl", matches)
    write_jsonl(out / "lipidmaps_unmatched.jsonl", unmatched)
    write_jsonl(out / "strict_calibration.jsonl", strict_rows)
    write_jsonl(out / "possible_false_positives.jsonl", false_positive_rows)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "v1_phase1_vs_v2_phase1.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
