from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


HEADGROUPS = ["PA", "PC", "PE", "PG", "PI", "PS"]
HEADGROUP_TO_INDEX = {name: index for index, name in enumerate(HEADGROUPS)}

CHAIN_COUNTS = [1, 2]
CHAIN_COUNT_TO_INDEX = {count: index for index, count in enumerate(CHAIN_COUNTS)}

CARBON_CLASSES = list(range(2, 41))
CARBON_TO_INDEX = {value: index for index, value in enumerate(CARBON_CLASSES)}

DOUBLE_BOND_CLASSES = list(range(0, 13))
DOUBLE_BOND_TO_INDEX = {
    value: index for index, value in enumerate(DOUBLE_BOND_CLASSES)
}

LINKAGES = ["ester", "ether", "vinyl_ether"]
LINKAGE_TO_INDEX = {name: index for index, name in enumerate(LINKAGES)}
POLARITY_TO_INDEX = {"negative": 0, "positive": 1}


@dataclass(frozen=True)
class ChainLabel:
    carbon: int
    double_bonds: int
    linkage: str | None
    linkage_inferred: bool = False


def normalize_linkage(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip().lower().replace(" ", "_").replace("-", "_")
        if text in LINKAGE_TO_INDEX:
            return text
    return None


def explicit_record_linkage(record: dict[str, Any]) -> str | None:
    return normalize_linkage(record.get("chain_linkage_summary"))


def encode_polarity(value: Any) -> int:
    text = str(value).strip().lower()
    if text in POLARITY_TO_INDEX:
        return POLARITY_TO_INDEX[text]
    raise ValueError(f"Unsupported polarity: {value!r}")


def sort_chains(
    chains: list[dict[str, Any]],
    inferred_linkage: str | None = None,
) -> list[ChainLabel]:
    normalized: list[ChainLabel] = []
    for chain in chains:
        carbon = int(chain["carbon"])
        double_bonds = int(chain["double_bonds"])
        linkage = normalize_linkage(chain.get("linkage"))
        linkage_inferred = False
        if linkage is None and inferred_linkage is not None:
            linkage = inferred_linkage
            linkage_inferred = True
        normalized.append(
            ChainLabel(
                carbon=carbon,
                double_bonds=double_bonds,
                linkage=linkage,
                linkage_inferred=linkage_inferred,
            )
        )
    return sorted(
        normalized,
        key=lambda item: (
            item.carbon,
            item.double_bonds,
            LINKAGE_TO_INDEX[item.linkage]
            if item.linkage is not None
            else len(LINKAGES),
        ),
    )


def encode_record_labels(record: dict[str, Any]) -> dict[str, torch.Tensor]:
    headgroup = record.get("prototype_headgroup") or record.get("lipid_class")
    if headgroup not in HEADGROUP_TO_INDEX:
        raise ValueError(f"Unsupported headgroup: {headgroup!r}")

    raw_chains = list(record.get("chains") or [])
    record_linkage = explicit_record_linkage(record)
    inferred_linkage = record_linkage if len(raw_chains) == 1 else None
    chains = sort_chains(raw_chains, inferred_linkage=inferred_linkage)
    if len(chains) not in CHAIN_COUNT_TO_INDEX:
        raise ValueError(f"Unsupported chain count: {len(chains)}")

    chain_mask = torch.zeros(2, dtype=torch.bool)
    chain_linkage_mask = torch.zeros(2, dtype=torch.bool)
    carbon_labels = torch.zeros(2, dtype=torch.long)
    double_bond_labels = torch.zeros(2, dtype=torch.long)
    linkage_labels = torch.zeros(2, dtype=torch.long)

    for index, chain in enumerate(chains[:2]):
        if chain.carbon not in CARBON_TO_INDEX:
            raise ValueError(f"Carbon count out of range: {chain.carbon}")
        if chain.double_bonds not in DOUBLE_BOND_TO_INDEX:
            raise ValueError(f"Double-bond count out of range: {chain.double_bonds}")

        chain_mask[index] = True
        carbon_labels[index] = CARBON_TO_INDEX[chain.carbon]
        double_bond_labels[index] = DOUBLE_BOND_TO_INDEX[chain.double_bonds]
        if chain.linkage is not None:
            chain_linkage_mask[index] = True
            linkage_labels[index] = LINKAGE_TO_INDEX[chain.linkage]

    return {
        "headgroup_label": torch.tensor(
            HEADGROUP_TO_INDEX[headgroup],
            dtype=torch.long,
        ),
        "chain_count_label": torch.tensor(
            CHAIN_COUNT_TO_INDEX[len(chains)],
            dtype=torch.long,
        ),
        "chain_carbon_labels": carbon_labels,
        "chain_double_bond_labels": double_bond_labels,
        "chain_linkage_labels": linkage_labels,
        "chain_mask": chain_mask,
        "chain_linkage_mask": chain_linkage_mask,
    }


def decode_chain_count(index: int) -> int:
    return CHAIN_COUNTS[int(index)]


def decode_chains(
    carbon_indices: torch.Tensor,
    double_bond_indices: torch.Tensor,
    linkage_indices: torch.Tensor,
    chain_count: int,
) -> list[dict[str, int | str]]:
    chains: list[dict[str, int | str]] = []
    for slot in range(chain_count):
        chains.append(
            {
                "carbon": CARBON_CLASSES[int(carbon_indices[slot])],
                "double_bonds": DOUBLE_BOND_CLASSES[int(double_bond_indices[slot])],
                "linkage": LINKAGES[int(linkage_indices[slot])],
            }
        )
    sorted_chains = sort_chains(chains)
    return [
        {
            "carbon": item.carbon,
            "double_bonds": item.double_bonds,
            "linkage": item.linkage,
        }
        for item in sorted_chains
    ]


def format_chain_text(chains: list[dict[str, int | str]]) -> str:
    return "_".join(
        f"{chain['carbon']}:{chain['double_bonds']}" for chain in chains
    )


def format_display_name(headgroup: str, chains: list[dict[str, int | str]]) -> str:
    return f"{headgroup}({format_chain_text(chains)})"


def label_schema() -> dict[str, list[int] | list[str]]:
    return {
        "headgroups": HEADGROUPS,
        "chain_counts": CHAIN_COUNTS,
        "carbon_classes": CARBON_CLASSES,
        "double_bond_classes": DOUBLE_BOND_CLASSES,
        "linkages": LINKAGES,
    }


def build_structure_candidate(headgroup: str, chains: list[dict]) -> dict:
    raise NotImplementedError(
        "Structure generation is intentionally reserved for a later version."
    )
