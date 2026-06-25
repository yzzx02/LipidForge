from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if not bool(mask.any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[mask], targets[mask])


def compute_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    headgroup_loss = F.cross_entropy(
        outputs["headgroup_logits"],
        batch["headgroup_label"],
    )
    chain_count_loss = F.cross_entropy(
        outputs["chain_count_logits"],
        batch["chain_count_label"],
    )
    chain_present_loss = nn.BCEWithLogitsLoss()(
        outputs["chain_present_logits"],
        batch["chain_present"],
    )
    chain_carbon_loss = _masked_cross_entropy(
        outputs["chain_carbon_logits"],
        batch["chain_carbon_labels"],
        batch["chain_mask"],
    )
    chain_double_bond_loss = _masked_cross_entropy(
        outputs["chain_double_bond_logits"],
        batch["chain_double_bond_labels"],
        batch["chain_mask"],
    )
    chain_linkage_loss = _masked_cross_entropy(
        outputs["chain_linkage_logits"],
        batch["chain_linkage_labels"],
        batch["chain_mask"],
    )

    losses = {
        "headgroup": headgroup_loss,
        "chain_count": chain_count_loss,
        "chain_present": chain_present_loss,
        "chain_carbon": chain_carbon_loss,
        "chain_double_bond": chain_double_bond_loss,
        "chain_linkage": chain_linkage_loss,
    }
    total = sum(losses.values())
    losses["total"] = total
    return total, losses
