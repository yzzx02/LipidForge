# LipidForge Agent Instructions

## Working directory

- The project root is `/home/administrator/projects/LipidForge`.
- Run project commands from the repository root unless a task explicitly requires another directory.

## Python and ROCm environment

- The only approved Python environment for LipidForge model code is the Conda environment `torch-rocm721`.
- The canonical command prefix is:

  `/home/administrator/miniconda3/bin/conda run --no-capture-output -n torch-rocm721`

- Run Python scripts as:

  `/home/administrator/miniconda3/bin/conda run --no-capture-output -n torch-rocm721 python <script>`

- Run tests as:

  `/home/administrator/miniconda3/bin/conda run --no-capture-output -n torch-rocm721 pytest -q`

- Do not use `/usr/bin/python3`.
- Do not use the Miniconda base Python.
- Do not invoke `/home/administrator/miniconda3/envs/torch-rocm721/bin/python` directly for GPU tasks, because direct execution may bypass the environment activation scripts required by ROCDXG.
- Do not create additional Conda environments without explicit user approval.
- Do not install, uninstall, replace, or upgrade PyTorch, ROCm, CUDA, HIP, AMD drivers, or system GPU packages without explicit user approval.
- Do not use `pip install`, `conda install`, or `apt install` without explicit user approval.

## Structure chemistry environment

- Use `torch-rocm721` only for model training, GPU checks, and PyTorch workflows.
- Use the independent CPU Conda environment `lipidforge-chem` for RDKit structure normalization and reference-structure matching.
- The canonical RDKit command prefix is:

  `/home/administrator/miniconda3/bin/conda run --no-capture-output -n lipidforge-chem`

- Do not install RDKit into `torch-rocm721` without explicit user approval.
- Do not run model training or GPU workloads from `lipidforge-chem`.

## ROCm behavior

- This machine uses AMD Radeon RX 9070 XT through ROCm 7.2.1 and ROCDXG under WSL2.
- PyTorch ROCm still uses `torch.cuda`, `torch.device("cuda")`, and `.to("cuda")`.
- Do not use `device="rocm"` or `device="hip"`.
- The Conda environment activation script sets `HSA_ENABLE_DXG_DETECTION=1`.
- Prefer `conda run` so that the activation script is applied.
- Do not set `HSA_OVERRIDE_GFX_VERSION`.

## Required GPU preflight

Before any GPU training command, verify:

```bash
/home/administrator/miniconda3/bin/conda run --no-capture-output -n torch-rocm721 python -c "import torch; print(torch.__version__); print(torch.version.hip); print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

Expected:

* `torch.version.hip` is not empty.
* `torch.cuda.is_available()` is `True`.
* Device count is at least 1.
* Device name contains `AMD Radeon RX 9070 XT`.

If the GPU preflight fails, stop before training and report the complete error. Do not silently fall back to CPU when `--require-gpu` is requested.

## Approved project commands

Fast development run:

```bash
/home/administrator/miniconda3/bin/conda run --no-capture-output -n torch-rocm721 \
  python train.py \
  --config configs/smoke_test.yaml \
  --fast-dev-run \
  --require-gpu
```

Small-batch overfit test:

```bash
/home/administrator/miniconda3/bin/conda run --no-capture-output -n torch-rocm721 \
  python train.py \
  --config configs/smoke_test.yaml \
  --overfit-small-batch \
  --require-gpu
```

Tests:

```bash
/home/administrator/miniconda3/bin/conda run --no-capture-output -n torch-rocm721 pytest -q
```

Do not run `configs/baseline.yaml` for formal training unless the user explicitly requests it.

## Data protection

* Do not commit files under `data/_downloads/`.
* Do not commit files under `data/expanded_phospholipids/`.
* Do not commit large `.zip`, `.tsv`, `.mgf`, checkpoint, or generated JSONL files.
* Preserve the original source, license, spectrum ID, and provenance fields.
* Do not relabel `P-lipid-unresolved` records as PC, PE, PG, PI, PS, PA, or another lipid class without structure-based evidence.
* Do not randomly split spectra by row. Future train/validation/test splits must be grouped by molecular identity such as InChIKey.

## Git safety

* Preserve current branch, existing uncommitted changes, and Git history.
* Do not commit, push, merge, delete branches, or modify the remote unless explicitly requested.
* Do not delete any external or Windows-side project backup unless the user explicitly requests deletion after integrity verification.
