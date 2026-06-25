#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from collections import Counter
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("jsonl")
    p.add_argument("--out", default="dataset_audit")
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    counts = {k: Counter() for k in ["lipid_class","lipid_family","source","polarity","adduct","license"]}
    n = 0
    unique_molecules = set()
    with Path(a.jsonl).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line); n += 1
            for k in counts:
                counts[k][str(r.get(k) or "unknown")] += 1
            unique_molecules.add(r.get("inchikey") or r.get("smiles") or r.get("name") or r.get("spectrum_id"))
    report = {"spectra": n, "unique_molecule_keys": len(unique_molecules),
              **{k: dict(v.most_common()) for k,v in counts.items()}}
    (out/"audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, c in counts.items():
        with (out/f"{name}_counts.csv").open("w", encoding="utf-8", newline="") as w:
            wr=csv.writer(w); wr.writerow([name,"count"]); wr.writerows(c.most_common())
    print(json.dumps(report, ensure_ascii=False, indent=2))
if __name__ == "__main__":
    main()
