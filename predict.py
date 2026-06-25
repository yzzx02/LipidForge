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


def load_records(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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
    model_config = config.get("model", {})
    max_peaks = int(model_config.get("max_peaks", 200))
    mz_scale = float(model_config.get("mz_scale", 1000.0))
    threshold = (
        args.confidence_threshold
        if args.confidence_threshold is not None
        else float(model_config.get("confidence_threshold", 0.50))
    )

    device = torch.device("cpu") if args.cpu else resolve_device(require_gpu=False)
    model = LipidTransformer(
        peak_feature_dim=int(model_config.get("peak_feature_dim", 3)),
        d_model=int(model_config.get("d_model", 128)),
        nhead=int(model_config.get("nhead", 4)),
        num_layers=int(model_config.get("num_layers", 4)),
        dim_feedforward=int(model_config.get("dim_feedforward", 256)),
        dropout=float(model_config.get("dropout", 0.10)),
        activation=str(model_config.get("activation", "gelu")),
        norm_first=bool(model_config.get("norm_first", True)),
        max_peaks=max_peaks,
        mz_scale=mz_scale,
    ).to(device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        state = checkpoint.get("model_state_dict", checkpoint)
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
