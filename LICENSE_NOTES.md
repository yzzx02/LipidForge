# License Notes

This repository contains source code for the LipidForge baseline and a tiny
pilot MS/MS fixture copied to `data/pilot/experimental_ms2_pilot.jsonl`.

The pilot spectrum rows retain source and license metadata in each JSONL record.
They come from MassBank-derived records:

- Five Chubu records are marked `CC BY-NC-SA`.
- One RIKEN PA record is marked `CC BY-SA`.

Because the pilot includes `CC BY-NC-SA` records, treat the redistributed pilot
data as non-commercial and share-alike unless those rows are removed or
separately re-licensed. The pilot data is included only to make parser, batching,
forward, loss, backward, and smoke-test behavior reproducible.

The larger local `glycerophospholipid_pilot_v1/` package, raw MassBank records,
LIPID MAPS structure seeds, and bulk MGF files are intentionally not uploaded by
this repository. Rebuild or download full datasets from their original sources
and follow each source license.
