from __future__ import annotations

import torch

from lipidforge.losses import compute_losses


def test_compute_losses_masks_absent_chain_slot():
    outputs = {
        "headgroup_logits": torch.randn(1, 6, requires_grad=True),
        "chain_count_logits": torch.randn(1, 2, requires_grad=True),
        "chain_carbon_logits": torch.randn(1, 2, 39, requires_grad=True),
        "chain_double_bond_logits": torch.randn(1, 2, 13, requires_grad=True),
        "chain_linkage_logits": torch.randn(1, 2, 3, requires_grad=True),
    }
    batch = {
        "headgroup_label": torch.tensor([1]),
        "chain_count_label": torch.tensor([0]),
        "chain_carbon_labels": torch.tensor([[14, 0]]),
        "chain_double_bond_labels": torch.tensor([[0, 0]]),
        "chain_linkage_labels": torch.tensor([[0, 0]]),
        "chain_mask": torch.tensor([[True, False]]),
        "chain_linkage_mask": torch.tensor([[True, False]]),
    }

    total, losses = compute_losses(outputs, batch)
    assert torch.isfinite(total)
    assert set(losses) == {
        "headgroup",
        "chain_count",
        "chain_carbon",
        "chain_double_bond",
        "chain_linkage",
        "total",
    }
    total.backward()
    assert outputs["chain_carbon_logits"].grad is not None
    assert torch.count_nonzero(outputs["chain_carbon_logits"].grad[0, 1]) == 0


def test_masked_losses_allow_empty_chain_mask():
    outputs = {
        "headgroup_logits": torch.randn(1, 6, requires_grad=True),
        "chain_count_logits": torch.randn(1, 2, requires_grad=True),
        "chain_carbon_logits": torch.randn(1, 2, 39, requires_grad=True),
        "chain_double_bond_logits": torch.randn(1, 2, 13, requires_grad=True),
        "chain_linkage_logits": torch.randn(1, 2, 3, requires_grad=True),
    }
    batch = {
        "headgroup_label": torch.tensor([1]),
        "chain_count_label": torch.tensor([0]),
        "chain_carbon_labels": torch.tensor([[0, 0]]),
        "chain_double_bond_labels": torch.tensor([[0, 0]]),
        "chain_linkage_labels": torch.tensor([[0, 0]]),
        "chain_mask": torch.tensor([[False, False]]),
        "chain_linkage_mask": torch.tensor([[False, False]]),
    }

    total, losses = compute_losses(outputs, batch)
    assert torch.isfinite(total)
    assert float(losses["chain_carbon"].detach()) == 0.0
    assert float(losses["chain_double_bond"].detach()) == 0.0
    assert float(losses["chain_linkage"].detach()) == 0.0


def test_unknown_linkage_mask_produces_no_linkage_gradient():
    outputs = {
        "headgroup_logits": torch.randn(1, 6, requires_grad=True),
        "chain_count_logits": torch.randn(1, 2, requires_grad=True),
        "chain_carbon_logits": torch.randn(1, 2, 39, requires_grad=True),
        "chain_double_bond_logits": torch.randn(1, 2, 13, requires_grad=True),
        "chain_linkage_logits": torch.randn(1, 2, 3, requires_grad=True),
    }
    batch = {
        "headgroup_label": torch.tensor([1]),
        "chain_count_label": torch.tensor([1]),
        "chain_carbon_labels": torch.tensor([[14, 16]]),
        "chain_double_bond_labels": torch.tensor([[0, 1]]),
        "chain_linkage_labels": torch.tensor([[0, 0]]),
        "chain_mask": torch.tensor([[True, True]]),
        "chain_linkage_mask": torch.tensor([[False, False]]),
    }

    total, losses = compute_losses(outputs, batch)
    assert float(losses["chain_linkage"].detach()) == 0.0
    total.backward()

    assert outputs["chain_linkage_logits"].grad is not None
    assert torch.count_nonzero(outputs["chain_linkage_logits"].grad) == 0
