from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from lipidforge.dataset import LipidSpectrumDataset, make_collate_fn
from lipidforge.environment import print_environment_info, resolve_device
from lipidforge.losses import compute_losses
from lipidforge.model import LipidTransformer, count_parameters


def _parse_scalar(value: str) -> Any:
    text = value.strip()
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        if any(marker in text for marker in [".", "e", "E"]):
            return float(text)
        return int(text)
    except ValueError:
        return text.strip("\"'")


def load_simple_yaml(path: str | Path) -> dict[str, dict[str, Any]]:
    config: dict[str, dict[str, Any]] = {}
    current_section: str | None = None
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" "):
            key = line.rstrip(":").strip()
            config[key] = {}
            current_section = key
            continue
        if current_section is None or ":" not in line:
            raise ValueError(f"Unsupported config line: {raw_line!r}")
        key, value = line.strip().split(":", 1)
        config[current_section][key.strip()] = _parse_scalar(value)
    return config


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=device.type == "cuda")
        else:
            moved[key] = value
    return moved


def has_nonfinite_tensor(outputs: dict[str, torch.Tensor]) -> bool:
    return any(not bool(torch.isfinite(tensor).all()) for tensor in outputs.values())


def print_batch_shapes(batch: dict[str, Any], outputs: dict[str, torch.Tensor]) -> None:
    print("Input shapes:")
    for key in [
        "peak_features",
        "peak_padding_mask",
        "precursor_mz",
        "polarity",
        "headgroup_label",
        "chain_count_label",
        "chain_present",
        "chain_carbon_labels",
        "chain_double_bond_labels",
        "chain_linkage_labels",
        "chain_mask",
    ]:
        if key in batch:
            print(f"  {key}: {tuple(batch[key].shape)}")
    print("Output shapes:")
    for key, value in outputs.items():
        print(f"  {key}: {tuple(value.shape)}")
    print("Layer shapes:")
    print("  peak encoder output: [batch_size, 200, 128]")
    print("  transformer input: [batch_size, 201, 128]")
    print("  transformer padding mask: [batch_size, 201]")
    print("  cls spectrum vector: [batch_size, 128]")


def build_loader(
    dataset: LipidSpectrumDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
    max_peaks: int,
) -> DataLoader:
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
        "collate_fn": make_collate_fn(max_peaks=max_peaks),
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **loader_kwargs)


def build_model(config: dict[str, Any]) -> LipidTransformer:
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


def print_run_header(
    model: LipidTransformer,
    device: torch.device,
    training_config: dict[str, Any],
    model_config: dict[str, Any],
    use_amp: bool,
) -> None:
    print_environment_info()
    total_params, trainable_params = count_parameters(model)
    print(f"Actual device: {device}")
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Batch size: {training_config.get('batch_size')}")
    print(f"Max peaks: {model_config.get('max_peaks', 200)}")
    print(f"AMP enabled: {use_amp}")


def run_optimizer_step(
    model: LipidTransformer,
    batch: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    gradient_clip_norm: float,
) -> tuple[float, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=use_amp,
    ):
        outputs = model(batch)
        loss, losses = compute_losses(outputs, batch)

    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite loss detected: {float(loss.detach().cpu())}")
    if has_nonfinite_tensor(outputs):
        raise RuntimeError("Non-finite model output detected")

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=gradient_clip_norm,
    )
    scaler.step(optimizer)
    scaler.update()
    return float(loss.detach().cpu()), losses, outputs


def run_fast_dev(
    model: LipidTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    gradient_clip_norm: float,
) -> None:
    model.train()
    for step, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        loss_value, losses, outputs = run_optimizer_step(
            model,
            batch,
            optimizer,
            scaler,
            device,
            use_amp,
            gradient_clip_norm,
        )
        if step == 1:
            print_batch_shapes(batch, outputs)
        loss_text = ", ".join(
            f"{name}={float(value.detach().cpu()):.4f}"
            for name, value in losses.items()
        )
        print(f"fast-dev-run step {step}: {loss_text}")
        if step >= 2:
            break
    print(f"fast-dev-run completed on {device}")


def run_overfit_small_batch(
    model: LipidTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    gradient_clip_norm: float,
    steps: int,
) -> None:
    model.train()
    batch = move_batch_to_device(next(iter(loader)), device)
    losses: list[float] = []
    for step in range(1, steps + 1):
        loss_value, _, _ = run_optimizer_step(
            model,
            batch,
            optimizer,
            scaler,
            device,
            use_amp,
            gradient_clip_norm,
        )
        losses.append(loss_value)
        if step == 1 or step == steps or step % 10 == 0:
            print(f"overfit step {step}: loss={loss_value:.4f}")

    print(
        "overfit-small-batch loss change: "
        f"{losses[0]:.4f} -> {losses[-1]:.4f}"
    )


def run_regular_training(
    model: LipidTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    gradient_clip_norm: float,
    epochs: int,
    checkpoint_path: str | None,
) -> None:
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            loss_value, _, _ = run_optimizer_step(
                model,
                batch,
                optimizer,
                scaler,
                device,
                use_amp,
                gradient_clip_norm,
            )
            epoch_losses.append(loss_value)
        mean_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        print(f"epoch {epoch}: loss={mean_loss:.4f}")
        if device.type == "cuda":
            allocated = torch.cuda.max_memory_allocated() / 1024**3
            reserved = torch.cuda.max_memory_reserved() / 1024**3
            print(f"GPU memory allocated: {allocated:.2f} GB")
            print(f"GPU memory reserved: {reserved:.2f} GB")

    if checkpoint_path:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict()}, path)
        loaded = torch.load(path, map_location=device)
        model.load_state_dict(loaded["model_state_dict"])
        print(f"Checkpoint saved and loaded: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke_test.yaml")
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--overfit-small-batch", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()

    config = load_simple_yaml(args.config)
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    data_config = config.get("data", {})

    require_gpu = bool(training_config.get("require_gpu", False) or args.require_gpu)
    device = resolve_device(require_gpu=require_gpu)
    max_peaks = int(model_config.get("max_peaks", 200))
    mz_scale = float(model_config.get("mz_scale", 1000.0))

    dataset = LipidSpectrumDataset(
        data_config.get(
            "train_jsonl",
            "glycerophospholipid_pilot_v1/experimental_ms2_pilot.jsonl",
        ),
        max_peaks=max_peaks,
        mz_scale=mz_scale,
    )
    batch_size = int(training_config.get("batch_size", 2))
    if args.overfit_small_batch:
        batch_size = min(batch_size, len(dataset))

    loader = build_loader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=not args.overfit_small_batch,
        num_workers=int(training_config.get("num_workers", 0)),
        device=device,
        max_peaks=max_peaks,
    )

    model = build_model(model_config).to(device)
    use_amp = bool(training_config.get("mixed_precision", False)) and device.type == "cuda"
    print_run_header(model, device, training_config, model_config, use_amp)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 3.0e-4)),
        weight_decay=float(training_config.get("weight_decay", 1.0e-4)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    gradient_clip_norm = float(training_config.get("gradient_clip_norm", 1.0))

    try:
        if args.fast_dev_run:
            run_fast_dev(
                model,
                loader,
                optimizer,
                scaler,
                device,
                use_amp,
                gradient_clip_norm,
            )
            return
        if args.overfit_small_batch:
            run_overfit_small_batch(
                model,
                loader,
                optimizer,
                scaler,
                device,
                use_amp,
                gradient_clip_norm,
                steps=int(training_config.get("overfit_steps", 60)),
            )
            return

        run_regular_training(
            model,
            loader,
            optimizer,
            scaler,
            device,
            use_amp,
            gradient_clip_norm,
            epochs=int(training_config.get("epochs", 2)),
            checkpoint_path=training_config.get("checkpoint_path"),
        )
    except RuntimeError as exc:
        message = str(exc).lower()
        if "out of memory" in message or "oom" in message:
            print(f"OOM at batch size {batch_size}. Try reducing 64 to 32.")
        raise


if __name__ == "__main__":
    main()
