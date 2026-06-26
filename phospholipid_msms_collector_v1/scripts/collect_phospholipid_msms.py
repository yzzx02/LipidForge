#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import shutil
import sys
import time
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

try:
    from rdkit import Chem, rdBase
except ImportError:  # Keep the v1 collector importable outside lipidforge-chem.
    Chem = None
    rdBase = None

MASSBANK_RELEASE_URL = "https://github.com/MassBank/MassBank-data/archive/refs/tags/{release}.zip"
MASSSPECGYM_URL = "https://huggingface.co/datasets/roman-bushuiev/MassSpecGym/resolve/main/data/MassSpecGym1.5.tsv?download=true"

FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")
CHAIN_HINTS = re.compile(
    r"(?:\b\d{1,2}:\d{1,2}\b|phosphatid|glycerophosph|sphingo|ceramide|cardiolipin|"
    r"plasmalogen|archaetid|inositolphosphorylceramide|phosphonolipid)",
    re.I,
)
EXCLUDE_NAME = re.compile(
    r"\b(?:ATP|ADP|AMP|GTP|GDP|GMP|CTP|CDP(?![- ]?(?:DAG|diacylglycerol))|"
    r"UTP|UDP|UMP|coenzyme A|acetyl[- ]?CoA|nucleotide|oligonucleotide|DNA|RNA)\b",
    re.I,
)
FULL_INCHIKEY = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
CONNECTIVITY_BLOCK = re.compile(r"^[A-Z]{14}$")
ADDUCT_PATTERN = re.compile(r"(\[[^\]]+\](?:\d*[+-]))")
HASH_VERSION = "v2.0"
STANDARDIZATION_VERSION = "structure-connectivity-v1"


def stable_hash(prefix: str, payload: Any, length: int = 24) -> str:
    text = payload if isinstance(payload, str) else json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return f"{prefix}_{hashlib.sha256(text.encode()).hexdigest()[:length]}"


def normalize_identifier_fields(value: str | None) -> dict[str, Any]:
    raw = (value or "").strip() or None
    out = {
        "provided_identifier_raw": raw,
        "provided_full_inchikey": None,
        "provided_connectivity_key": None,
        "inchikey_semantics": "missing",
        "identity_warnings": [],
    }
    if not raw:
        return out
    token = raw.upper()
    if FULL_INCHIKEY.match(token):
        out["provided_full_inchikey"] = token
        out["provided_connectivity_key"] = token.split("-", 1)[0]
        out["inchikey_semantics"] = "full_inchikey"
    elif CONNECTIVITY_BLOCK.match(token):
        out["provided_connectivity_key"] = token
        out["inchikey_semantics"] = "connectivity_block"
        out["identity_warnings"].append("legacy inchikey field contains a 14-character connectivity block, not a full InChIKey")
    else:
        out["inchikey_semantics"] = "unrecognized_identifier"
        out["identity_warnings"].append("legacy inchikey field is neither a full InChIKey nor a 14-character connectivity block")
    return out


def canonicalize_adduct_token(token: str) -> str:
    compact = re.sub(r"\s+", "", token)
    m = re.match(r"^\[([^\]]+)\](\d*[+-])$", compact, flags=re.I)
    if not m:
        return compact
    body, charge = m.groups()
    body = re.sub(r"^m", "M", body, flags=re.I)
    replacements = {
        "nh4": "NH4",
        "ch3coo": "CH3COO",
        "hcoo": "HCOO",
        "na": "Na",
        "cl": "Cl",
        "k": "K",
        "h": "H",
    }

    def repl(match: re.Match[str]) -> str:
        return replacements.get(match.group(0).lower(), match.group(0))

    body = re.sub(r"NH4|CH3COO|HCOO|Na|Cl|K|H", repl, body, flags=re.I)
    return f"[{body}]{charge}"


def normalize_adduct_value(raw: str | None) -> tuple[str | None, str, str, list[str]]:
    text = (raw or "").strip()
    warnings: list[str] = []
    if not text:
        return None, "unknown", "missing", ["adduct missing"]
    if text.upper() in {"POSITIVE", "NEGATIVE", "POS", "NEG", "ION_MODE POSITIVE", "ION_MODE NEGATIVE"}:
        return text, "unknown", "ion_mode_not_adduct", ["ion mode was not used as adduct"]
    m = ADDUCT_PATTERN.search(text)
    if not m:
        return text, "unknown", "unparsed", [f"could not normalize adduct from raw value: {text}"]
    return text, canonicalize_adduct_token(m.group(1)), "normalized", warnings


def extract_massbank_adduct(fields: dict[str, list[str]]) -> dict[str, Any]:
    candidates: list[tuple[str, str]] = []
    for value in fields.get("MS$FOCUSED_ION", []):
        if value.upper().startswith("PRECURSOR_TYPE"):
            candidates.append(("MS$FOCUSED_ION: PRECURSOR_TYPE", value.split(" ", 1)[1].strip() if " " in value else value))
    for value in fields.get("MS$FOCUSED_ION", []):
        if value.upper().startswith("ION_TYPE"):
            candidates.append(("MS$FOCUSED_ION: ION_TYPE", value.split(" ", 1)[1].strip() if " " in value else value))
    for value in fields.get("AC$MASS_SPECTROMETRY", []):
        upper = value.upper()
        if upper.startswith("PRECURSOR_TYPE") or "ADDUCT" in upper:
            candidates.append(("AC$MASS_SPECTROMETRY", value.split(" ", 1)[1].strip() if " " in value else value))
    for key in ("ADDUCT", "PRECURSOR_TYPE", "ION_TYPE"):
        for value in fields.get(key, []):
            candidates.append((key, value))
    for source, raw in candidates:
        adduct_raw, adduct, status, warnings = normalize_adduct_value(raw)
        if adduct != "unknown":
            return {
                "adduct_raw": adduct_raw,
                "adduct": adduct,
                "adduct_source_field": source,
                "adduct_normalization_status": status,
                "adduct_warnings": warnings,
            }
    if candidates:
        source, raw = candidates[0]
        adduct_raw, adduct, status, warnings = normalize_adduct_value(raw)
        return {
            "adduct_raw": adduct_raw,
            "adduct": adduct,
            "adduct_source_field": source,
            "adduct_normalization_status": status,
            "adduct_warnings": warnings,
        }
    return {
        "adduct_raw": None,
        "adduct": "unknown",
        "adduct_source_field": None,
        "adduct_normalization_status": "missing",
        "adduct_warnings": ["no explicit adduct field found"],
    }


def normalize_collision_energy(value: Any) -> dict[str, Any]:
    raw = None if value is None else str(value).strip()
    if not raw:
        return {
            "collision_energy_raw": None,
            "collision_energy_normalized": None,
            "collision_energy_unit": None,
            "collision_energy_parse_status": "missing",
        }
    text = raw.replace("−", "-")
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    unit = "eV" if re.search(r"\beV\b", text, flags=re.I) else ("%" if "%" in text else None)
    if len(numbers) == 1 and not re.search(r"\b(?:ramp|range|to|-|/)\b", text, flags=re.I):
        return {
            "collision_energy_raw": raw,
            "collision_energy_normalized": float(numbers[0]),
            "collision_energy_unit": unit or "eV",
            "collision_energy_parse_status": "single_value",
        }
    if len(numbers) >= 2:
        return {
            "collision_energy_raw": raw,
            "collision_energy_normalized": None,
            "collision_energy_unit": unit,
            "collision_energy_parse_status": "range_or_compound",
        }
    return {
        "collision_energy_raw": raw,
        "collision_energy_normalized": None,
        "collision_energy_unit": unit,
        "collision_energy_parse_status": "free_text",
    }


def normalize_instrument_fields(instrument: str | None, instrument_type: str | None) -> dict[str, Any]:
    def norm(value: str | None) -> str | None:
        value = (value or "").strip()
        return re.sub(r"\s+", " ", value) if value else None

    return {
        "instrument_raw": instrument,
        "instrument": norm(instrument),
        "instrument_type_raw": instrument_type,
        "instrument_type": norm(instrument_type),
    }


def normalized_peaks_for_hash(peaks: Iterable[tuple[Any, Any]]) -> list[list[float]]:
    return sorted([round(float(m), 4), round(float(i), 6)] for m, i in peaks)


def component_heavy_atoms(mol: Any) -> int:
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)


def component_carbons(mol: Any) -> int:
    return sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "C")


def likely_counterion_smiles(smiles: str) -> bool:
    return smiles in {"[Na+]", "[K+]", "[Cl-]", "[H+]", "[Li+]", "[NH4+]", "[Br-]"}


def connectivity_component_smiles(mol: Any) -> str:
    copy = Chem.RWMol(mol)
    for atom in copy.GetAtoms():
        atom.SetIsotope(0)
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        atom.SetFormalCharge(0)
        atom.SetNoImplicit(False)
    neutral = copy.GetMol()
    Chem.SanitizeMol(neutral, catchErrors=True)
    Chem.RemoveStereochemistry(neutral)
    return Chem.MolToSmiles(neutral, canonical=True, isomericSmiles=False)


def component_selection_key(mol: Any) -> tuple[int, int, str, str]:
    return (
        component_carbons(mol),
        component_heavy_atoms(mol),
        connectivity_component_smiles(mol),
        Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
    )


def standardize_structure_identity(raw_smiles: str | None) -> dict[str, Any]:
    raw = (raw_smiles or "").strip()
    warnings: list[str] = []
    base = {
        "standardization_version": STANDARDIZATION_VERSION,
        "rdkit_version": getattr(rdBase, "rdkitVersion", None) if rdBase is not None else None,
        "identity_fallback": False,
        "identity_warnings": warnings,
        "structure_record_id": None,
        "connectivity_id": None,
        "canonical_isomeric_smiles": None,
        "canonical_nonisomeric_smiles": None,
        "rdkit_full_inchikey": None,
        "rdkit_connectivity_key": None,
        "fragment_count": None,
        "all_fragment_smiles": [],
        "major_component_smiles": None,
        "minor_fragment_smiles": [],
        "fragment_signature": None,
        "has_multiple_fragments": False,
        "likely_counterions": [],
        "raw_structure_group": stable_hash("raw", {"value": raw}) if raw else None,
        "provided_full_inchikey_group": None,
        "provided_connectivity_group": None,
        "canonical_isomeric_group": None,
        "canonical_nonisomeric_group": None,
    }
    if Chem is None:
        warnings.append("RDKit unavailable; used raw structure fallback")
        base["identity_fallback"] = True
        base["structure_record_id"] = stable_hash("str", f"rawstructure:v1:{raw}")
        base["connectivity_id"] = stable_hash("conn", f"rawconnectivity:v1:{raw}")
        return base
    if not raw:
        warnings.append("missing SMILES; used empty raw structure fallback")
        base["identity_fallback"] = True
        base["structure_record_id"] = stable_hash("str", "rawstructure:v1:")
        base["connectivity_id"] = stable_hash("conn", "rawconnectivity:v1:")
        return base
    mol = Chem.MolFromSmiles(raw, sanitize=False)
    if mol is None:
        warnings.append("RDKit parse failed; used raw structure fallback")
        base["identity_fallback"] = True
        base["structure_record_id"] = stable_hash("str", f"rawstructure:v1:{raw}")
        base["connectivity_id"] = stable_hash("conn", f"rawconnectivity:v1:{raw}")
        return base
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:
        warnings.append(f"RDKit sanitize failed; used raw structure fallback: {exc}")
        base["identity_fallback"] = True
        base["structure_record_id"] = stable_hash("str", f"rawstructure:v1:{raw}")
        base["connectivity_id"] = stable_hash("conn", f"rawconnectivity:v1:{raw}")
        return base
    iso = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    noniso = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    base["canonical_isomeric_smiles"] = iso
    base["canonical_nonisomeric_smiles"] = noniso
    base["canonical_isomeric_group"] = stable_hash("ciso", iso)
    base["canonical_nonisomeric_group"] = stable_hash("cnon", noniso)
    try:
        ik = Chem.MolToInchiKey(mol)
        base["rdkit_full_inchikey"] = ik
        base["rdkit_connectivity_key"] = ik.split("-", 1)[0] if ik else None
    except Exception as exc:
        warnings.append(f"RDKit InChIKey generation failed: {exc}")
    base["structure_record_id"] = stable_hash("str", f"structure:v1:{iso}")
    fragments = list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False))
    frag_smiles = sorted(Chem.MolToSmiles(f, canonical=True, isomericSmiles=True) for f in fragments)
    base["fragment_count"] = len(fragments)
    base["all_fragment_smiles"] = frag_smiles
    base["fragment_signature"] = stable_hash("frag", frag_smiles)
    base["has_multiple_fragments"] = len(fragments) > 1
    base["likely_counterions"] = [s for s in frag_smiles if likely_counterion_smiles(s)]
    if fragments:
        organic = [f for f in fragments if component_carbons(f) > 0]
        if organic:
            major = max(organic, key=component_selection_key)
        else:
            warnings.append("no carbon-containing fragment found; used largest fragment for connectivity")
            major = max(fragments, key=component_selection_key)
        major_noniso = connectivity_component_smiles(major)
        major_iso = Chem.MolToSmiles(major, canonical=True, isomericSmiles=True)
        minor_fragment_smiles = list(frag_smiles)
        if major_iso in minor_fragment_smiles:
            minor_fragment_smiles.remove(major_iso)
        base["major_component_smiles"] = major_noniso
        base["minor_fragment_smiles"] = minor_fragment_smiles
        base["connectivity_id"] = stable_hash("conn", f"connectivity:v1:{major_noniso}")
    else:
        warnings.append("no fragments returned; used full non-isomeric SMILES for connectivity")
        base["major_component_smiles"] = noniso
        base["connectivity_id"] = stable_hash("conn", f"connectivity:v1:{noniso}")
    return base


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[reuse] {dest}")
        return dest
    print(f"[download] {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "LipidForge-data-collector/1.0"})
    last_exc: Exception | None = None
    for attempt in range(1, 4):
        try:
            print(f"[download] attempt {attempt}/3 -> {dest}")
            with urllib.request.urlopen(req, timeout=180) as r, tmp.open("wb") as w:
                shutil.copyfileobj(r, w, length=1024 * 1024)
            tmp.replace(dest)
            return dest
        except Exception as exc:
            last_exc = exc
            print(f"[download-error] {url} attempt {attempt}/3: {exc}", file=sys.stderr)
            if attempt < 3:
                time.sleep(2 * attempt)
    raise RuntimeError(f"failed to download {url} after 3 attempts: {last_exc}")


def parse_formula(formula: str | None) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not formula:
        return counts
    clean = re.sub(r"[\+\-\.\s]", "", str(formula))
    for element, number in FORMULA_TOKEN.findall(clean):
        counts[element] = counts.get(element, 0) + (int(number) if number else 1)
    return counts


def phosphorus_lipid_formula_candidate(formula: str | None, name: str = "", smiles: str = "") -> bool:
    counts = parse_formula(formula)
    p, c, h, o = counts.get("P", 0), counts.get("C", 0), counts.get("H", 0), counts.get("O", 0)
    if p < 1 or c < 8 or o < 3:
        return False
    if c and h / c < 1.15 and not CHAIN_HINTS.search(name):
        return False
    if EXCLUDE_NAME.search(name):
        return False
    # SMILES 中有 P 时是额外支持证据；没有 SMILES 时不强制。
    if smiles and "P" not in smiles and "p" not in smiles:
        return False
    return True


def load_aliases(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def classify_name(name: str, aliases: dict[str, Any]) -> tuple[str | None, str | None, list[str]]:
    text = name or ""
    cls = family = None
    for item in aliases["classes"]:
        if any(re.search(p, text, flags=re.I) for p in item["patterns"]):
            cls, family = item["class"], item["family"]
            break
    mods: list[str] = []
    for item in aliases.get("linkage_modifications", []):
        if any(re.search(p, text, flags=re.I) for p in item["patterns"]):
            mods.append(item["label"])
    return cls, family, mods


def normalize_polarity(value: str | None, adduct: str | None = None) -> str | None:
    text = (value or "").strip().lower()
    if "pos" in text or text == "positive":
        return "positive"
    if "neg" in text or text == "negative":
        return "negative"
    ad = adduct or ""
    if ad.endswith("+") or "]+" in ad:
        return "positive"
    if ad.endswith("-") or "]-" in ad:
        return "negative"
    return None


def parse_float(value: Any) -> float | None:
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def clean_peaks(peaks: Iterable[tuple[Any, Any]], min_peaks: int) -> list[list[float]]:
    out: list[list[float]] = []
    for mz, inten in peaks:
        m = parse_float(mz)
        i = parse_float(inten)
        if m is None or i is None or m <= 0 or i < 0:
            continue
        out.append([m, i])
    if len(out) < min_peaks:
        return []
    return out


def spectrum_hash(record: dict[str, Any]) -> str:
    peaks = sorted((round(float(m), 4), round(float(i), 6)) for m, i in record["peaks_raw"])
    key = {
        "inchikey": record.get("inchikey"),
        "smiles": record.get("smiles"),
        "precursor_mz": round(float(record["precursor_mz"]), 4),
        "adduct": record.get("adduct"),
        "polarity": record.get("polarity"),
        "peaks": peaks,
    }
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()


def parse_massbank_record(
    lines: list[str],
    source_file: str,
    fallback_id: str,
    aliases: dict[str, Any],
    min_peaks: int,
) -> dict[str, Any] | None:
    fields: dict[str, list[str]] = {}
    peaks: list[tuple[float, float]] = []
    in_peaks = False
    for line in lines:
        if line.startswith("PK$PEAK:"):
            in_peaks = True
            continue
        if in_peaks:
            if line.startswith("//"):
                break
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    peaks.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    pass
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            fields.setdefault(k.strip(), []).append(v.strip())

    ms_type = " ".join(fields.get("AC$MASS_SPECTROMETRY", []))
    record_title = " ".join(fields.get("RECORD_TITLE", []))
    if "MS2" not in ms_type.upper() and "MS/MS" not in ms_type.upper() and "MS2" not in record_title.upper():
        return None

    names = fields.get("CH$NAME", [])
    name = names[0] if names else record_title
    formula = (fields.get("CH$FORMULA") or [None])[0]
    smiles = (fields.get("CH$SMILES") or [None])[0]
    exact_mass = parse_float((fields.get("CH$EXACT_MASS") or [None])[0])
    license_text = (fields.get("LICENSE") or fields.get("COPYRIGHT") or [None])[0]

    mass_lines = fields.get("AC$MASS_SPECTROMETRY", [])
    precursor_mz = None
    ion_mode = None
    collision_energy = None
    for x in mass_lines:
        ux = x.upper()
        if ux.startswith("PRECURSOR_M/Z"):
            precursor_mz = parse_float(x.split()[-1])
        elif ux.startswith("ION_MODE"):
            ion_mode = x.split()[-1]
        elif "COLLISION_ENERGY" in ux:
            collision_energy = x.split(" ", 1)[1] if " " in x else x

    if precursor_mz is None:
        m = re.search(r"(?:PRECURSOR_M/Z|MS\$FOCUSED_ION:\s*PRECURSOR_M/Z)\s+([0-9.]+)", "\n".join(lines), re.I)
        precursor_mz = parse_float(m.group(1)) if m else None
    cleaned = clean_peaks(peaks, min_peaks)
    if precursor_mz is None or not cleaned:
        return None

    lipid_class, family, mods = classify_name(" ; ".join(names + [record_title]), aliases)
    is_formula_candidate = phosphorus_lipid_formula_candidate(formula, name, smiles or "")
    if lipid_class is None and not is_formula_candidate:
        return None
    if lipid_class is None:
        lipid_class, family = "P-lipid-unresolved", "phosphorus_lipid_candidate"

    adduct_info = extract_massbank_adduct(fields)

    inchikey = None
    lipidmaps_id = None
    for link in fields.get("CH$LINK", []):
        if link.upper().startswith("INCHIKEY"):
            inchikey = link.split()[-1]
        if link.upper().startswith("LIPIDMAPS"):
            lipidmaps_id = link.split()[-1]

    return {
        "spectrum_id": (fields.get("ACCESSION") or [fallback_id])[0],
        "source": "MassBank-data",
        "source_file": source_file,
        "source_release": None,
        "license": license_text,
        "record_title": record_title,
        "name": name,
        "all_names": names,
        "lipid_class": lipid_class,
        "lipid_family": family,
        "linkage_modifications": mods,
        "classification_source": "name_alias" if lipid_class != "P-lipid-unresolved" else "formula_structure_filter",
        "polarity": normalize_polarity(ion_mode, None if adduct_info["adduct"] == "unknown" else adduct_info["adduct"]),
        "precursor_mz": precursor_mz,
        "adduct": adduct_info["adduct"],
        "adduct_raw": adduct_info["adduct_raw"],
        "adduct_source_field": adduct_info["adduct_source_field"],
        "adduct_normalization_status": adduct_info["adduct_normalization_status"],
        "adduct_warnings": adduct_info["adduct_warnings"],
        "collision_energy": collision_energy,
        "instrument": (fields.get("AC$INSTRUMENT") or [None])[0],
        "instrument_type": (fields.get("AC$INSTRUMENT_TYPE") or [None])[0],
        "formula": formula,
        "exact_mass": exact_mass,
        "smiles": smiles,
        "inchikey": inchikey,
        "lipidmaps_id": lipidmaps_id,
        "peaks_raw": cleaned,
        "num_peaks": len(cleaned),
        "experimental": True,
    }


def parse_massbank_file(path: Path, aliases: dict[str, Any], min_peaks: int) -> dict[str, Any] | None:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return parse_massbank_record(lines, str(path), path.stem, aliases, min_peaks)


def collect_massbank(work: Path, aliases: dict[str, Any], release: str, min_peaks: int) -> list[dict[str, Any]]:
    archive = download(MASSBANK_RELEASE_URL.format(release=release), work / f"MassBank-data-{release}.zip")
    records = []
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            if info.is_dir() or not info.filename.lower().endswith(".txt"):
                continue
            try:
                with zf.open(info) as handle:
                    lines = handle.read().decode("utf-8", errors="replace").splitlines()
                rec = parse_massbank_record(
                    lines,
                    f"{archive}!{info.filename}",
                    Path(info.filename).stem,
                    aliases,
                    min_peaks,
                )
            except Exception as exc:
                print(f"[warn] MassBank parse failed {info.filename}: {exc}", file=sys.stderr)
                continue
            if rec:
                rec["source_release"] = release
                records.append(rec)
    print(f"[MassBank] selected {len(records)} spectra")
    return records


def parse_csv_floats(text: str | None) -> list[float]:
    if not text:
        return []
    out = []
    for x in str(text).split(","):
        v = parse_float(x.strip())
        if v is not None:
            out.append(v)
    return out


def collect_massspecgym(work: Path, min_peaks: int) -> list[dict[str, Any]]:
    path = download(MASSSPECGYM_URL, work / "MassSpecGym1.5.tsv")
    records = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            formula = row.get("formula")
            smiles = row.get("smiles") or ""
            if not phosphorus_lipid_formula_candidate(formula, "", smiles):
                continue
            mzs = parse_csv_floats(row.get("mzs"))
            intensities = parse_csv_floats(row.get("intensities"))
            peaks = clean_peaks(zip(mzs, intensities), min_peaks)
            precursor_mz = parse_float(row.get("precursor_mz"))
            if precursor_mz is None or not peaks:
                continue
            adduct_raw, adduct, adduct_status, adduct_warnings = normalize_adduct_value(row.get("adduct"))
            records.append({
                "spectrum_id": row.get("identifier"),
                "source": "MassSpecGym1.5",
                "source_file": str(path),
                "source_release": "1.5",
                "license": "MIT dataset release; preserve upstream provenance",
                "record_title": None,
                "name": None,
                "all_names": [],
                "lipid_class": "P-lipid-unresolved",
                "lipid_family": "phosphorus_lipid_candidate",
                "linkage_modifications": [],
                "classification_source": "formula_structure_filter",
                "polarity": normalize_polarity(None, None if adduct == "unknown" else adduct),
                "precursor_mz": precursor_mz,
                "adduct": adduct,
                "adduct_raw": adduct_raw,
                "adduct_source_field": "MassSpecGym1.5: adduct",
                "adduct_normalization_status": adduct_status,
                "adduct_warnings": adduct_warnings,
                "collision_energy": row.get("collision_energy"),
                "instrument": None,
                "instrument_type": row.get("instrument_type"),
                "formula": formula,
                "exact_mass": parse_float(row.get("parent_mass")),
                "smiles": smiles,
                "inchikey": row.get("inchikey"),
                "lipidmaps_id": None,
                "peaks_raw": peaks,
                "num_peaks": len(peaks),
                "experimental": True,
                "upstream_fold": row.get("fold"),
                "simulation_challenge": row.get("simulation_challenge"),
            })
    print(f"[MassSpecGym] selected {len(records)} P-lipid candidates")
    return records


def parse_mgf(path: Path, aliases: dict[str, Any], min_peaks: int, source_name: str) -> list[dict[str, Any]]:
    records = []
    fields: dict[str, str] = {}
    peaks: list[tuple[float, float]] = []
    inside = False

    def flush():
        nonlocal fields, peaks
        if not inside:
            return
        name = fields.get("NAME") or fields.get("COMPOUND_NAME") or fields.get("TITLE") or ""
        formula = fields.get("FORMULA")
        smiles = fields.get("SMILES") or fields.get("CANONICALSMILES")
        cls, family, mods = classify_name(name, aliases)
        candidate = phosphorus_lipid_formula_candidate(formula, name, smiles or "")
        cleaned = clean_peaks(peaks, min_peaks)
        precursor = parse_float(fields.get("PEPMASS", "").split()[0] if fields.get("PEPMASS") else None)
        if precursor is not None and cleaned and (cls is not None or candidate):
            adduct_raw, adduct, adduct_status, adduct_warnings = normalize_adduct_value(fields.get("ADDUCT") or fields.get("PRECURSORTYPE"))
            records.append({
                "spectrum_id": fields.get("SPECTRUMID") or fields.get("SCANS") or f"{source_name}:{len(records)+1}",
                "source": source_name,
                "source_file": str(path),
                "source_release": None,
                "license": fields.get("LICENSE"),
                "record_title": fields.get("TITLE"),
                "name": name or None,
                "all_names": [name] if name else [],
                "lipid_class": cls or "P-lipid-unresolved",
                "lipid_family": family or "phosphorus_lipid_candidate",
                "linkage_modifications": mods,
                "classification_source": "name_alias" if cls else "formula_structure_filter",
                "polarity": normalize_polarity(fields.get("IONMODE"), None if adduct == "unknown" else adduct),
                "precursor_mz": precursor,
                "adduct": adduct,
                "adduct_raw": adduct_raw,
                "adduct_source_field": "MGF: ADDUCT/PRECURSORTYPE",
                "adduct_normalization_status": adduct_status,
                "adduct_warnings": adduct_warnings,
                "collision_energy": fields.get("COLLISIONENERGY"),
                "instrument": fields.get("INSTRUMENT"),
                "instrument_type": fields.get("INSTRUMENTTYPE"),
                "formula": formula,
                "exact_mass": parse_float(fields.get("EXACTMASS")),
                "smiles": smiles,
                "inchikey": fields.get("INCHIKEY"),
                "lipidmaps_id": fields.get("LIPIDMAPS"),
                "peaks_raw": cleaned,
                "num_peaks": len(cleaned),
                "experimental": True,
            })
        fields, peaks = {}, []

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if line.upper() == "BEGIN IONS":
                inside = True
                fields, peaks = {}, []
            elif line.upper() == "END IONS":
                flush()
                inside = False
            elif inside and "=" in line:
                k, v = line.split("=", 1)
                fields[k.strip().upper()] = v.strip()
            elif inside:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        peaks.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        pass
    print(f"[MGF:{source_name}] selected {len(records)} spectra")
    return records


def deduplicate(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen = set()
    out = []
    removed = 0
    for rec in records:
        h = spectrum_hash(rec)
        if h in seen:
            removed += 1
            continue
        seen.add(h)
        rec["spectrum_hash"] = h
        out.append(rec)
    return out, removed


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as w:
        for row in rows:
            w.write(json.dumps(row, ensure_ascii=False) + "\n")


def enrich_record_v2(record: dict[str, Any]) -> dict[str, Any]:
    rec = dict(record)
    rec["source_record_id"] = rec.get("spectrum_id") or rec.get("record_title") or rec.get("source_file")

    identifiers = normalize_identifier_fields(rec.get("inchikey"))
    identifier_warnings = identifiers.pop("identity_warnings", [])
    structure = standardize_structure_identity(rec.get("smiles"))
    structure_warnings = structure.pop("identity_warnings", [])
    rec.update(identifiers)
    rec.update(structure)
    rec["provided_full_inchikey_group"] = (
        stable_hash("pfik", f"provided_full_inchikey:v1:{rec['provided_full_inchikey']}")
        if rec.get("provided_full_inchikey") else None
    )
    rec["provided_connectivity_group"] = (
        stable_hash("pconn", f"provided_connectivity:v1:{rec['provided_connectivity_key']}")
        if rec.get("provided_connectivity_key") else None
    )

    warnings = list(identifier_warnings) + list(structure_warnings)
    rdkit_conn = rec.get("rdkit_connectivity_key")
    provided_conn = rec.get("provided_connectivity_key")
    rec["provided_connectivity_matches_rdkit"] = None
    if provided_conn and rdkit_conn:
        rec["provided_connectivity_matches_rdkit"] = provided_conn == rdkit_conn
        if provided_conn != rdkit_conn:
            warnings.append("provided connectivity key does not match RDKit-derived connectivity key")
    rec["provided_full_inchikey_matches_rdkit"] = None
    if rec.get("provided_full_inchikey") and rec.get("rdkit_full_inchikey"):
        rec["provided_full_inchikey_matches_rdkit"] = rec["provided_full_inchikey"] == rec["rdkit_full_inchikey"]
        if rec["provided_full_inchikey"] != rec["rdkit_full_inchikey"]:
            warnings.append("provided full InChIKey does not match RDKit-derived full InChIKey")
    rec["identity_warnings"] = sorted(dict.fromkeys(warnings))

    if "adduct_raw" not in rec:
        adduct_raw, adduct, status, adduct_warnings = normalize_adduct_value(rec.get("adduct"))
        rec["adduct_raw"] = adduct_raw
        rec["adduct"] = adduct
        rec["adduct_source_field"] = None
        rec["adduct_normalization_status"] = status
        rec["adduct_warnings"] = adduct_warnings
    if rec.get("polarity") is None:
        rec["polarity"] = normalize_polarity(None, None if rec.get("adduct") == "unknown" else rec.get("adduct"))

    rec.update(normalize_collision_energy(rec.get("collision_energy")))
    rec.update(normalize_instrument_fields(rec.get("instrument"), rec.get("instrument_type")))

    identity_for_hash = (
        rec.get("connectivity_id")
        or rec.get("structure_record_id")
        or rec.get("provided_connectivity_key")
        or rec.get("provided_full_inchikey")
        or rec.get("smiles")
        or rec.get("source_record_id")
    )
    # Peak identity is source-independent: no collision energy, instrument,
    # source, or source_record_id is included here.
    peak_key = {
        "hash_version": HASH_VERSION,
        "molecular_identity": identity_for_hash,
        "precursor_mz": round(float(rec["precursor_mz"]), 4),
        "adduct": rec.get("adduct") or "unknown",
        "polarity": rec.get("polarity") or "unknown",
        "peaks": normalized_peaks_for_hash(rec.get("peaks_raw", [])),
    }
    rec["peak_identity_hash"] = stable_hash("peak", peak_key)
    # Acquisition metadata keeps the acquisition conditions but not the
    # source_record_id. Raw CE and parse status are retained to avoid
    # collapsing ranges, free text, and missing values to the same hash.
    metadata_key = {
        "hash_version": HASH_VERSION,
        "peak_identity_hash": rec["peak_identity_hash"],
        "collision_energy_raw": rec.get("collision_energy_raw"),
        "collision_energy_normalized": rec.get("collision_energy_normalized"),
        "collision_energy_unit": rec.get("collision_energy_unit"),
        "collision_energy_parse_status": rec.get("collision_energy_parse_status"),
        "instrument": rec.get("instrument"),
        "instrument_type": rec.get("instrument_type"),
        "source": rec.get("source"),
    }
    rec["acquisition_metadata_hash"] = stable_hash("acqmeta", metadata_key)
    # Acquisition record identity adds the source-local record identifier.
    # Only exact repeated records with the same acquisition_record_hash are
    # removable by the optional acquisition-dedup view.
    rec["acquisition_record_hash"] = stable_hash("acqrec", {**metadata_key, "source_record_id": rec.get("source_record_id")})
    rec["duplicate_group_id"] = None
    rec["duplicate_group_size"] = 1
    rec["duplicate_relation"] = "unique"
    return rec


def classify_duplicate_relation(items: list[dict[str, Any]]) -> str:
    if len(items) == 1:
        return "unique"
    acquisition_hashes = {r["acquisition_record_hash"] for r in items}
    if len(acquisition_hashes) == 1:
        return "removable_exact_duplicate"
    ce_keys = {
        (
            r.get("collision_energy_raw"),
            r.get("collision_energy_normalized"),
            r.get("collision_energy_unit"),
            r.get("collision_energy_parse_status"),
        )
        for r in items
    }
    instrument_keys = {(r.get("instrument"), r.get("instrument_type")) for r in items}
    missing_meta = any(
        r.get("collision_energy_parse_status") == "missing" or (not r.get("instrument") and not r.get("instrument_type"))
        for r in items
    )
    metadata_hashes = {r.get("acquisition_metadata_hash") for r in items}
    sources = {r.get("source") for r in items}
    source_record_ids = {r.get("source_record_id") for r in items}
    if len(ce_keys) > 1:
        return "same_peaks_different_collision_energy"
    if len(instrument_keys) > 1:
        return "same_peaks_different_instrument"
    if len(sources) > 1:
        return "cross_source_peak_duplicate"
    if missing_meta:
        return "missing_acquisition_metadata"
    if len(sources) == 1 and len(metadata_hashes) == 1 and len(source_record_ids) > 1:
        return "same_source_acquisition_duplicate_candidate"
    return "same_source_acquisition_duplicate_candidate"


def assign_duplicate_groups(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        groups.setdefault(rec["peak_identity_hash"], []).append(rec)
    rows: list[dict[str, Any]] = []
    for peak_hash, items in groups.items():
        group_id = stable_hash("dup", f"duplicate_group:v1:{peak_hash}")
        relation = classify_duplicate_relation(items)
        for rec in items:
            rec["duplicate_group_id"] = group_id
            rec["duplicate_group_size"] = len(items)
            rec["duplicate_relation"] = relation
        if len(items) > 1:
            rows.append({
                "duplicate_group_id": group_id,
                "peak_identity_hash": peak_hash,
                "duplicate_group_size": len(items),
                "duplicate_relation": relation,
                "sources": sorted({r.get("source") or "unknown" for r in items}),
                "structure_record_ids": sorted({r.get("structure_record_id") or "unknown" for r in items}),
                "connectivity_ids": sorted({r.get("connectivity_id") or "unknown" for r in items}),
                "collision_energy_values": sorted({str(r.get("collision_energy_raw")) for r in items}),
                "instrument_values": sorted({str(r.get("instrument") or r.get("instrument_type")) for r in items}),
                "example_spectrum_ids": [r.get("spectrum_id") for r in items[:10]],
            })
    return rows


def acquisition_deduplicate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for rec in records:
        key = rec["acquisition_record_hash"]
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return out


def write_count_csv(path: Path, header: tuple[str, str], counts: Counter) -> None:
    with path.open("w", encoding="utf-8", newline="") as w:
        wr = csv.writer(w)
        wr.writerow(header)
        wr.writerows(counts.most_common())


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_v2_outputs(out: Path, records: list[dict[str, Any]], duplicate_groups: list[dict[str, Any]]) -> dict[str, Any]:
    strict = [r for r in records if r["lipid_class"] != "P-lipid-unresolved"]
    unresolved = [r for r in records if r["lipid_class"] == "P-lipid-unresolved"]
    acquisition_dedup = acquisition_deduplicate(records)

    write_jsonl(out / "phospholipid_msms_all_v2.jsonl", records)
    write_jsonl(out / "phospholipid_msms_strict_v2.jsonl", strict)
    write_jsonl(out / "phosphorus_lipid_candidates_unresolved_v2.jsonl", unresolved)
    write_jsonl(out / "phospholipid_msms_acquisition_dedup_v2.jsonl", acquisition_dedup)
    write_jsonl(out / "duplicate_groups_v2.jsonl", duplicate_groups)

    class_counts = Counter(r["lipid_class"] for r in records)
    family_counts = Counter(r["lipid_family"] for r in records)
    source_counts = Counter(r["source"] for r in records)
    adduct_counts = Counter(r.get("adduct") or "unknown" for r in records)
    polarity_counts = Counter(r.get("polarity") or "unknown" for r in records)
    duplicate_relation_counts = Counter(r.get("duplicate_relation") or "unknown" for r in records)
    inchikey_semantics_counts = Counter(r.get("inchikey_semantics") or "missing" for r in records)
    adduct_status_counts = Counter(r.get("adduct_normalization_status") or "missing" for r in records)

    v1_summary = load_json_if_exists(Path("data/expanded_phospholipids/summary.json"))
    v1_v2 = {
        "v1_summary_available": v1_summary is not None,
        "v1_total_after_exact_dedup": v1_summary.get("total_after_exact_dedup") if v1_summary else None,
        "v2_total_all_acquisitions": len(records),
        "v2_total_acquisition_dedup": len(acquisition_dedup),
        "v1_strict_named_class": v1_summary.get("strict_named_class") if v1_summary else None,
        "v2_strict_named_class": len(strict),
        "v1_unresolved_structure_candidates": v1_summary.get("unresolved_structure_candidates") if v1_summary else None,
        "v2_unresolved_structure_candidates": len(unresolved),
        "main_reason": "v2 retains all acquisition records and separates acquisition-level deduplication from peak-identity duplicate groups.",
    }
    (out / "v1_v2_comparison.json").write_text(json.dumps(v1_v2, ensure_ascii=False, indent=2), encoding="utf-8")

    full_mismatch = sum(
        1 for r in records
        if r.get("provided_full_inchikey") and r.get("rdkit_connectivity_key")
        and r["provided_full_inchikey"].split("-", 1)[0] != r["rdkit_connectivity_key"]
    )
    conn_mismatch = sum(
        1 for r in records
        if r.get("provided_connectivity_key") and r.get("rdkit_connectivity_key")
        and r["provided_connectivity_key"] != r["rdkit_connectivity_key"]
    )

    summary = {
        "schema_version": "v2",
        "hash_version": HASH_VERSION,
        "standardization_version": STANDARDIZATION_VERSION,
        "total_all_acquisition_records": len(records),
        "total_after_acquisition_dedup": len(acquisition_dedup),
        "acquisition_duplicates_removed_in_optional_file": len(records) - len(acquisition_dedup),
        "removable_exact_duplicate_records": len(records) - len(acquisition_dedup),
        "strict_named_class": len(strict),
        "unresolved_structure_candidates": len(unresolved),
        "unique_structure_record_ids": len({r.get("structure_record_id") for r in records}),
        "unique_connectivity_ids": len({r.get("connectivity_id") for r in records}),
        "identity_fallback_records": sum(1 for r in records if r.get("identity_fallback")),
        "multi_fragment_records": sum(1 for r in records if r.get("has_multiple_fragments")),
        "provided_full_inchikey_connectivity_mismatches": full_mismatch,
        "provided_connectivity_key_mismatches": conn_mismatch,
        "duplicate_group_count": len(duplicate_groups),
        "class_counts": dict(class_counts.most_common()),
        "family_counts": dict(family_counts.most_common()),
        "source_counts": dict(source_counts.most_common()),
        "adduct_counts": dict(adduct_counts.most_common()),
        "adduct_normalization_status_counts": dict(adduct_status_counts.most_common()),
        "polarity_counts": dict(polarity_counts.most_common()),
        "duplicate_relation_counts": dict(duplicate_relation_counts.most_common()),
        "inchikey_semantics_counts": dict(inchikey_semantics_counts.most_common()),
        "v1_v2_comparison": v1_v2,
        "warnings": [
            "MassSpecGym legacy inchikey values that are 14 uppercase characters are treated as connectivity keys, not invalid full InChIKeys.",
            "Main v2 files retain all acquisition records; phospholipid_msms_acquisition_dedup_v2.jsonl removes only repeated acquisition_record_hash records.",
            "Duplicate groups are peak-identity groups. same_source_acquisition_duplicate_candidate is a provenance/acquisition duplicate candidate, not a safe-delete label.",
            "No LIPID MAPS class labels are written back into source acquisition records.",
        ],
    }
    (out / "summary_v2.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_count_csv(out / "class_counts_v2.csv", ("lipid_class", "spectra"), class_counts)
    write_count_csv(out / "source_counts_v2.csv", ("source", "spectra"), source_counts)
    write_count_csv(out / "adduct_counts_v2.csv", ("adduct", "spectra"), adduct_counts)
    return summary


def main():
    ap = argparse.ArgumentParser(description="Collect broad phospholipid MS/MS spectra")
    ap.add_argument("--out", default="phospholipid_msms_expanded")
    ap.add_argument("--work", default="_downloads")
    ap.add_argument("--aliases", default=str(Path(__file__).resolve().parents[1] / "config" / "class_aliases.json"))
    ap.add_argument("--massbank-release", default="2026.03")
    ap.add_argument("--skip-massbank", action="store_true")
    ap.add_argument("--skip-massspecgym", action="store_true")
    ap.add_argument("--local-mgf", action="append", default=[], help="Additional GNPS/MoNA/local MGF file; repeatable")
    ap.add_argument("--min-peaks", type=int, default=3)
    ap.add_argument("--schema-version", choices=["v1", "v2"], default="v1")
    args = ap.parse_args()

    out = Path(args.out)
    work = Path(args.work)
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    aliases = load_aliases(Path(args.aliases))

    records: list[dict[str, Any]] = []
    if not args.skip_massbank:
        records.extend(collect_massbank(work, aliases, args.massbank_release, args.min_peaks))
    if not args.skip_massspecgym:
        records.extend(collect_massspecgym(work, args.min_peaks))
    for mgf in args.local_mgf:
        p = Path(mgf)
        records.extend(parse_mgf(p, aliases, args.min_peaks, source_name=f"local_mgf:{p.stem}"))

    if args.schema_version == "v2":
        records = [enrich_record_v2(r) for r in records]
        duplicate_groups = assign_duplicate_groups(records)
        summary = write_v2_outputs(out, records, duplicate_groups)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    records, removed = deduplicate(records)
    strict = [r for r in records if r["lipid_class"] != "P-lipid-unresolved"]
    unresolved = [r for r in records if r["lipid_class"] == "P-lipid-unresolved"]

    write_jsonl(out / "phospholipid_msms_all.jsonl", records)
    write_jsonl(out / "phospholipid_msms_strict.jsonl", strict)
    write_jsonl(out / "phosphorus_lipid_candidates_unresolved.jsonl", unresolved)

    class_counts = Counter(r["lipid_class"] for r in records)
    family_counts = Counter(r["lipid_family"] for r in records)
    source_counts = Counter(r["source"] for r in records)
    polarity_counts = Counter(r.get("polarity") or "unknown" for r in records)

    summary = {
        "total_after_exact_dedup": len(records),
        "exact_duplicates_removed": removed,
        "strict_named_class": len(strict),
        "unresolved_structure_candidates": len(unresolved),
        "class_counts": dict(class_counts.most_common()),
        "family_counts": dict(family_counts.most_common()),
        "source_counts": dict(source_counts.most_common()),
        "polarity_counts": dict(polarity_counts.most_common()),
        "warnings": [
            "MassSpecGym rows have structures but no lipid class names; they remain unresolved until graph-based classification.",
            "Do not split train/test by spectrum row. Group by full InChIKey or connectivity key to prevent molecular leakage.",
            "Preserve source-specific license fields. Do not commit bulk third-party spectra to a public repository without review."
        ]
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with (out / "class_counts.csv").open("w", encoding="utf-8", newline="") as w:
        wr = csv.writer(w)
        wr.writerow(["lipid_class", "spectra"])
        wr.writerows(class_counts.most_common())

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
