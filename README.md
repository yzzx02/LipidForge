# LipidForge

LipidForge is a small research baseline for glycerophospholipid MS/MS modeling.
The first version predicts a headgroup class and simple fatty-acyl chain labels
from spectrum polarity, precursor m/z, fragment m/z values, and fragment
intensities.

This repository is intentionally simple. It uses native PyTorch modules,
standard-library JSONL parsing, and a compact Transformer encoder. It does not
use Hugging Face Transformers, PyTorch Lightning, Hydra, custom CUDA/HIP code,
SMILES generation, graph neural networks, or candidate reranking.

## Data

The committed pilot file is `data/pilot/experimental_ms2_pilot.jsonl`. It has 6
experimental MS/MS spectra: PA, PC, PE, PG, PI, and PS. The LIPID MAPS seed files
from the local data package are structure-only labels with no usable peak lists
and are not committed.

See `LICENSE_NOTES.md` before redistributing the pilot data. It includes records
marked `CC BY-NC-SA`, so treat the pilot data as non-commercial/share-alike unless
you remove or separately re-license those rows.

## Environment

Use an existing environment that already provides PyTorch. Do not install,
uninstall, or replace PyTorch from this project.

```text
PyTorch is provided by the user's existing environment.
```

ROCm PyTorch still uses `torch.cuda` and `torch.device("cuda")`, so the same code
path works for NVIDIA CUDA, AMD ROCm, and CPU fallback tests.

## Quick Checks

```bash
pytest -q
python train.py --config configs/smoke_test.yaml --fast-dev-run
python train.py --config configs/smoke_test.yaml --overfit-small-batch
```

The smoke and overfit modes only verify that the data pipeline, model, losses,
and backward pass work. They are not model-performance measurements.

## Formal Training Later

After adding enough real spectra, copy the project to the RX 9070 XT WSL machine
and run:

```bash
python train.py --config configs/baseline.yaml
```

The baseline config keeps FP32 enabled by default. Test AMP only after standard
forward, backward, optimizer step, validation, checkpoint save/load, no NaN, and
no OOM have all been verified.
