from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

from lipidforge.dataset import collate_spectra, featurize_record
from lipidforge.environment import resolve_device
from lipidforge.labels import (
    HEADGROUPS,
    decode_chain_count,
    decode_chains,
    format_chain_text,
    format_display_name,
)
from lipidforge.model import LipidTransformer
from train import load_simple_yaml


MODEL_CONFIG_KEYS = {
    "peak_feature_dim",
    "d_model",
    "nhead",
    "num_layers",
    "dim_feedforward",
    "dropout",
    "activation",
    "norm_first",
}
PREPROCESSING_CONFIG_KEYS = {"max_peaks", "mz_scale"}


def load_records(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _check_config_conflicts(
    external_model_config: dict[str, Any],
    checkpoint_model_config: dict[str, Any],
    checkpoint_preprocessing_config: dict[str, Any],
) -> None:
    conflicts: list[str] = []
    for key in sorted(MODEL_CONFIG_KEYS):
        if (
            key in checkpoint_model_config
            and key in external_model_config
            and checkpoint_model_config[key] != external_model_config[key]
        ):
            conflicts.append(
                f"model.{key}: checkpoint={checkpoint_model_config[key]!r}, "
                f"external={external_model_config[key]!r}"
            )
    for key in sorted(PREPROCESSING_CONFIG_KEYS):
        if (
            key in checkpoint_preprocessing_config
            and key in external_model_config
            and checkpoint_preprocessing_config[key] != external_model_config[key]
        ):
            conflicts.append(
                f"preprocessing.{key}: "
                f"checkpoint={checkpoint_preprocessing_config[key]!r}, "
                f"external={external_model_config[key]!r}"
            )
    if conflicts:
        joined = "; ".join(conflicts)
        raise ValueError(f"External config conflicts with checkpoint: {joined}")


def _build_model_from_config(config: dict[str, Any]) -> LipidTransformer:
    return LipidTransformer(
        peak_feature_dim=int(config.get("peak_feature_dim", 3)),
        d_model=int(config.get("d_model", 128)),
        nhead=int(config.get("nhead", 4)),
        num_layers=int(config.get("num_layers", 4)),
        dim_feedforward=int(config.get("dim_feedforward", 256)),
        dropout=float(config.get("dropout", 0.10)),
        activation=str(config.get("activation", "gelu")),
        norm_first=bool(config.get("norm_first", True)),
        max_peaks=int(config.get("max_peaks", 200)),
        mz_scale=float(config.get("mz_scale", 1000.0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", default="configs/smoke_test.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output")
    parser.add_argument("--confidence-threshold", type=float)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    config = load_simple_yaml(args.config)
    external_model_config = config.get("model", {})
    model_config = dict(external_model_config)
    preprocessing_config = {
        "max_peaks": int(model_config.get("max_peaks", 200)),
        "mz_scale": float(model_config.get("mz_scale", 1000.0)),
    }
    checkpoint: dict[str, Any] | None = None
    state: dict[str, torch.Tensor] | None = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        state = checkpoint.get("model_state_dict", checkpoint)
        checkpoint_model_config = checkpoint.get("model_config") or {}
        checkpoint_preprocessing_config = checkpoint.get("preprocessing_config") or {}
        _check_config_conflicts(
            external_model_config,
            checkpoint_model_config,
            checkpoint_preprocessing_config,
        )
        model_config.update(checkpoint_model_config)
        preprocessing_config.update(checkpoint_preprocessing_config)

    max_peaks = int(preprocessing_config.get("max_peaks", 200))
    mz_scale = float(preprocessing_config.get("mz_scale", 1000.0))
    model_config["max_peaks"] = max_peaks
    model_config["mz_scale"] = mz_scale
    threshold = (
        args.confidence_threshold
        if args.confidence_threshold is not None
        else float(model_config.get("confidence_threshold", 0.50))
    )

    device = torch.device("cpu") if args.cpu else resolve_device(require_gpu=False)
    model = _build_model_from_config(model_config).to(device)
    if state is not None:
        model.load_state_dict(state)
    else:
        print(
            "Warning: no checkpoint was provided; predictions use random weights.",
            file=sys.stderr,
        )

    records = load_records(args.input)
    samples = [
        featurize_record(
            record,
            max_peaks=max_peaks,
            mz_scale=mz_scale,
            require_labels=False,
        )
        for record in records
    ]
    batch = collate_spectra(samples, max_peaks=max_peaks)
    batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }

    model.eval()
    with torch.no_grad():
        outputs = model(batch)
        headgroup_probs = torch.softmax(outputs["headgroup_logits"], dim=-1)
        chain_count_indices = outputs["chain_count_logits"].argmax(dim=-1)
        carbon_indices = outputs["chain_carbon_logits"].argmax(dim=-1)
        double_bond_indices = outputs["chain_double_bond_logits"].argmax(dim=-1)
        linkage_indices = outputs["chain_linkage_logits"].argmax(dim=-1)

    predictions: list[dict[str, Any]] = []
    for row in range(len(records)):
        probability, headgroup_index = headgroup_probs[row].max(dim=-1)
        chain_count = decode_chain_count(int(chain_count_indices[row]))
        chains = decode_chains(
            carbon_indices[row].cpu(),
            double_bond_indices[row].cpu(),
            linkage_indices[row].cpu(),
            chain_count=chain_count,
        )
        if float(probability) < threshold:
            predictions.append(
                {
                    "headgroup": "low_confidence",
                    "headgroup_probability": float(probability),
                    "display_name": (
                        f"Unknown-P-headgroup({format_chain_text(chains)})"
                    ),
                    "chain_count": chain_count,
                    "chains": chains,
                }
            )
            continue

        headgroup = HEADGROUPS[int(headgroup_index)]
        predictions.append(
            {
                "headgroup": headgroup,
                "headgroup_probability": float(probability),
                "display_name": format_display_name(headgroup, chains),
                "chain_count": chain_count,
                "chains": chains,
            }
        )

    lines = "\n".join(
        json.dumps(prediction, ensure_ascii=False) for prediction in predictions
    )
    if args.output:
        Path(args.output).write_text(lines + "\n", encoding="utf-8")
    else:
        print(lines)


if __name__ == "__main__":
    main()
