from __future__ import annotations

import torch
from torch import nn

from .labels import CARBON_CLASSES, DOUBLE_BOND_CLASSES, HEADGROUPS, LINKAGES


class PeakEncoder(nn.Module):
    def __init__(self, peak_feature_dim: int = 3, d_model: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(peak_feature_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, peak_features: torch.Tensor) -> torch.Tensor:
        return self.net(peak_features)


class LipidTransformer(nn.Module):
    def __init__(
        self,
        peak_feature_dim: int = 3,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.10,
        activation: str = "gelu",
        norm_first: bool = True,
        max_peaks: int = 200,
        mz_scale: float = 1000.0,
        num_headgroups: int = len(HEADGROUPS),
        num_carbon_classes: int = len(CARBON_CLASSES),
        num_double_bond_classes: int = len(DOUBLE_BOND_CLASSES),
        num_linkage_classes: int = len(LINKAGES),
    ) -> None:
        super().__init__()
        self.max_peaks = max_peaks
        self.mz_scale = mz_scale
        self.d_model = d_model

        self.peak_encoder = PeakEncoder(peak_feature_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.precursor_encoder = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.polarity_embedding = nn.Embedding(2, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        self.headgroup_head = nn.Linear(d_model, num_headgroups)
        self.chain_count_head = nn.Linear(d_model, 2)
        self.chain_carbon_head = nn.Linear(d_model, 2 * num_carbon_classes)
        self.chain_double_bond_head = nn.Linear(
            d_model,
            2 * num_double_bond_classes,
        )
        self.chain_linkage_head = nn.Linear(d_model, 2 * num_linkage_classes)

        self.num_carbon_classes = num_carbon_classes
        self.num_double_bond_classes = num_double_bond_classes
        self.num_linkage_classes = num_linkage_classes
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def forward(
        self,
        batch: dict[str, torch.Tensor] | None = None,
        *,
        peak_features: torch.Tensor | None = None,
        peak_padding_mask: torch.Tensor | None = None,
        precursor_mz: torch.Tensor | None = None,
        polarity: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if batch is not None:
            peak_features = batch["peak_features"]
            peak_padding_mask = batch["peak_padding_mask"]
            precursor_mz = batch["precursor_mz"]
            polarity = batch["polarity"]

        if (
            peak_features is None
            or peak_padding_mask is None
            or precursor_mz is None
            or polarity is None
        ):
            raise ValueError("Missing model input tensors")

        batch_size = peak_features.shape[0]
        peak_tokens = self.peak_encoder(peak_features)
        precursor_input = (precursor_mz / self.mz_scale).view(batch_size, 1)
        cls = (
            self.cls_token.expand(batch_size, -1, -1)
            + self.precursor_encoder(precursor_input).unsqueeze(1)
            + self.polarity_embedding(polarity).unsqueeze(1)
        )

        tokens = torch.cat([cls, peak_tokens], dim=1)
        cls_mask = torch.zeros(
            batch_size,
            1,
            dtype=torch.bool,
            device=peak_padding_mask.device,
        )
        src_key_padding_mask = torch.cat([cls_mask, peak_padding_mask], dim=1)
        encoded = self.encoder(
            tokens,
            src_key_padding_mask=src_key_padding_mask,
        )
        spectrum = encoded[:, 0]

        return {
            "headgroup_logits": self.headgroup_head(spectrum),
            "chain_count_logits": self.chain_count_head(spectrum),
            "chain_carbon_logits": self.chain_carbon_head(spectrum).view(
                batch_size,
                2,
                self.num_carbon_classes,
            ),
            "chain_double_bond_logits": self.chain_double_bond_head(spectrum).view(
                batch_size,
                2,
                self.num_double_bond_classes,
            ),
            "chain_linkage_logits": self.chain_linkage_head(spectrum).view(
                batch_size,
                2,
                self.num_linkage_classes,
            ),
        }


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable
