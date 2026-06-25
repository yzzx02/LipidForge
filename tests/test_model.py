from __future__ import annotations

import torch

from lipidforge.dataset import collate_spectra, featurize_record
from lipidforge.losses import compute_losses
from lipidforge.model import LipidTransformer


def _record(peaks, chains=None):
    return {
        "spectrum_id": "demo",
        "prototype_headgroup": "PC",
        "polarity": "negative",
        "precursor_mz": 760.0,
        "chains": chains
        or [
            {"carbon": 16, "double_bonds": 0},
            {"carbon": 18, "double_bonds": 1},
        ],
        "chain_linkage_summary": "ester",
        "peaks_raw": peaks,
        "usable_for_pilot_training": True,
    }


def _batch(max_peaks=8):
    samples = [
        featurize_record(_record([[100.0, 1.0], [250.0, 4.0]])),
        featurize_record(_record([[125.0, 2.0], [300.0, 8.0], [400.0, 1.0]])),
    ]
    return collate_spectra(samples, max_peaks=max_peaks)


def test_cpu_batch_forward_shapes():
    model = LipidTransformer(max_peaks=8)
    batch = _batch(max_peaks=8)
    outputs = model(batch)

    assert outputs["headgroup_logits"].shape == (2, 6)
    assert outputs["chain_count_logits"].shape == (2, 2)
    assert outputs["chain_carbon_logits"].shape == (2, 2, 39)
    assert outputs["chain_double_bond_logits"].shape == (2, 2, 13)
    assert outputs["chain_linkage_logits"].shape == (2, 2, 3)


def test_single_spectrum_forward():
    model = LipidTransformer(max_peaks=8)
    sample = featurize_record(_record([[100.0, 1.0]]))
    batch = collate_spectra([sample], max_peaks=8)
    outputs = model(batch)
    assert outputs["headgroup_logits"].shape == (1, 6)


def test_gpu_forward_when_available():
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    model = LipidTransformer(max_peaks=8).to(device)
    batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in _batch(max_peaks=8).items()
    }
    outputs = model(batch)
    assert outputs["headgroup_logits"].is_cuda


def test_padding_values_do_not_change_valid_output():
    torch.manual_seed(7)
    model = LipidTransformer(max_peaks=8)
    model.eval()
    batch = _batch(max_peaks=8)
    noisy_batch = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    noisy_batch["peak_features"][batch["peak_padding_mask"]] = torch.randn(
        int(batch["peak_padding_mask"].sum()),
        3,
    )

    with torch.no_grad():
        plain = model(batch)
        noisy = model(noisy_batch)

    for key in plain:
        assert torch.allclose(plain[key], noisy[key], atol=1e-6)


def test_loss_backward_and_gradients_are_finite():
    torch.manual_seed(11)
    model = LipidTransformer(max_peaks=8)
    batch = _batch(max_peaks=8)
    outputs = model(batch)
    loss, losses = compute_losses(outputs, batch)

    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in losses.values())
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
