# Data

The repository includes only the tiny pilot spectrum file:

```text
data/pilot/experimental_ms2_pilot.jsonl
```

It contains 6 experimental MS/MS records, one each for PA, PC, PE, PG, PI, and
PS. Every row retains its `source` and `license` fields. The file is included so
the baseline data loader, padding logic, model forward pass, loss, backward pass,
and smoke tests can run without the larger local data package.

The local `glycerophospholipid_pilot_v1/` package, raw MassBank records, LIPID
MAPS structure seeds, and bulk MGF files are intentionally not committed. The
structure seed files have no usable MS/MS peak lists and must not be treated as
spectra.

Because the pilot includes records marked `CC BY-NC-SA`, treat redistributed
pilot data as non-commercial and share-alike unless those rows are removed or
separately re-licensed.
