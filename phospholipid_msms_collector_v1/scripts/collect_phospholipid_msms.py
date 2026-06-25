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
    adduct = None
    ion_mode = None
    collision_energy = None
    for x in mass_lines:
        ux = x.upper()
        if ux.startswith("PRECURSOR_M/Z"):
            precursor_mz = parse_float(x.split()[-1])
        elif ux.startswith("PRECURSOR_TYPE"):
            adduct = x.split(" ", 1)[1].strip() if " " in x else None
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
        "polarity": normalize_polarity(ion_mode, adduct),
        "precursor_mz": precursor_mz,
        "adduct": adduct,
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
                "polarity": normalize_polarity(None, row.get("adduct")),
                "precursor_mz": precursor_mz,
                "adduct": row.get("adduct"),
                "collision_energy": parse_float(row.get("collision_energy")),
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
                "polarity": normalize_polarity(fields.get("IONMODE"), fields.get("ADDUCT") or fields.get("PRECURSORTYPE")),
                "precursor_mz": precursor,
                "adduct": fields.get("ADDUCT") or fields.get("PRECURSORTYPE"),
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
