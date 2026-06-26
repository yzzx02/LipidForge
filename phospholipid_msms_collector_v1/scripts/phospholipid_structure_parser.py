#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

PARSER_VERSION = "gpl_v0.1"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "headgroups_v0_1.json"
IGNORED_PARTITION_ATOMIC_NUMS = {3, 11, 12, 19, 20, 17, 35, 53}

SUPPORTED_CLASS_BY_HEADGROUP_AND_CHAIN_COUNT = {
    ("phosphate", 2): "PA",
    ("phosphate", 1): "LPA",
    ("phosphocholine", 2): "PC",
    ("phosphocholine", 1): "LPC",
    ("phosphoethanolamine", 2): "PE",
    ("phosphoethanolamine", 1): "LPE",
    ("phosphoglycerol", 2): "PG",
    ("phosphoglycerol", 1): "LPG",
    ("phosphoinositol", 2): "PI",
    ("phosphoinositol", 1): "LPI",
    ("phosphoserine", 2): "PS",
    ("phosphoserine", 1): "LPS",
}


@dataclass
class Chain:
    attachment_backbone_atom: int
    attachment_hetero_atom: int
    first_chain_atom: int | None
    linkage_type: str
    core_atom_indices: list[int]
    partition_atom_indices: list[int]
    backbone_position_candidate: int
    warnings: list[str]


@dataclass
class BackboneCandidate:
    backbone_atom_indices: tuple[int, int, int]
    phosphorus_atom: int
    headgroup_bridge_oxygen: int
    headgroup_attachment_site: dict[str, int]
    chains: list[Chain]
    free_hydroxyl_sites: list[dict[str, int]]
    score: int
    warnings: list[str]


def load_headgroup_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atom_symbol(mol: Chem.Mol, idx: int) -> str:
    return mol.GetAtomWithIdx(idx).GetSymbol()


def heavy_atom_indices(mol: Chem.Mol) -> list[int]:
    return [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() > 1 and atom.GetAtomicNum() not in IGNORED_PARTITION_ATOMIC_NUMS
    ]


def heavy_edge_set(mol: Chem.Mol) -> set[tuple[int, int]]:
    edges = set()
    ignored_atoms = ignored_partition_atoms(mol)
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        if a in ignored_atoms or b in ignored_atoms:
            continue
        if mol.GetAtomWithIdx(a).GetAtomicNum() > 1 and mol.GetAtomWithIdx(b).GetAtomicNum() > 1:
            edges.add(tuple(sorted((a, b))))
    return edges


def ignored_partition_atoms(mol: Chem.Mol) -> set[int]:
    return {atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() in IGNORED_PARTITION_ATOMIC_NUMS}


def nonaromatic_carbon(atom: Chem.Atom) -> bool:
    return atom.GetAtomicNum() == 6 and not atom.GetIsAromatic()


def oxygen_neighbors(mol: Chem.Mol, idx: int) -> list[int]:
    return [n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors() if n.GetAtomicNum() == 8]


def is_carbonyl_carbon(mol: Chem.Mol, idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetAtomicNum() != 6:
        return False
    for bond in atom.GetBonds():
        other = bond.GetOtherAtom(atom)
        if other.GetAtomicNum() == 8 and bond.GetBondType() == Chem.BondType.DOUBLE:
            return True
    return False


def has_carbon_carbon_double_bond(mol: Chem.Mol, idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    for bond in atom.GetBonds():
        other = bond.GetOtherAtom(atom)
        if other.GetAtomicNum() == 6 and bond.GetBondType() == Chem.BondType.DOUBLE:
            return True
    return False


def connected_component_from(mol: Chem.Mol, start: int, blocked: set[int]) -> list[int]:
    if start in blocked:
        return []
    seen = {start}
    queue: deque[int] = deque([start])
    while queue:
        idx = queue.popleft()
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if nbr_idx in blocked or nbr_idx in seen:
                continue
            seen.add(nbr_idx)
            queue.append(nbr_idx)
    return sorted(seen)


def fragment_smiles(mol: Chem.Mol, atoms: Iterable[int]) -> str | None:
    atom_list = sorted(set(atoms))
    if not atom_list:
        return None
    try:
        return Chem.MolFragmentToSmiles(mol, atomsToUse=atom_list, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def formula_for_atoms(mol: Chem.Mol, atoms: Iterable[int]) -> str | None:
    atom_list = sorted(set(atoms))
    if not atom_list:
        return None
    try:
        smiles = Chem.MolFragmentToSmiles(mol, atomsToUse=atom_list, canonical=True, isomericSmiles=True)
        frag = Chem.MolFromSmiles(smiles)
        if frag is None:
            return None
        return rdMolDescriptors.CalcMolFormula(frag)
    except Exception:
        return None


def formal_charge_for_atoms(mol: Chem.Mol, atoms: Iterable[int]) -> int:
    return sum(mol.GetAtomWithIdx(idx).GetFormalCharge() for idx in set(atoms))


def classify_chain(mol: Chem.Mol, backbone_c: int, oxygen_idx: int, backbone_set: set[int], position: int) -> Chain:
    external = []
    for nbr in mol.GetAtomWithIdx(oxygen_idx).GetNeighbors():
        nbr_idx = nbr.GetIdx()
        if nbr_idx == backbone_c or nbr_idx in backbone_set:
            continue
        if nbr.GetAtomicNum() > 1:
            external.append(nbr_idx)
    warnings: list[str] = []
    if not external:
        return Chain(
            attachment_backbone_atom=backbone_c,
            attachment_hetero_atom=oxygen_idx,
            first_chain_atom=None,
            linkage_type="none",
            core_atom_indices=[],
            partition_atom_indices=[oxygen_idx],
            backbone_position_candidate=position,
            warnings=[],
        )
    if len(external) > 1:
        warnings.append("attachment_oxygen_has_multiple_external_heavy_neighbors")
    first = external[0]
    if atom_symbol(mol, first) != "C":
        return Chain(
            attachment_backbone_atom=backbone_c,
            attachment_hetero_atom=oxygen_idx,
            first_chain_atom=first,
            linkage_type="unsupported",
            core_atom_indices=connected_component_from(mol, first, backbone_set | {oxygen_idx}),
            partition_atom_indices=sorted({oxygen_idx, first}),
            backbone_position_candidate=position,
            warnings=["chain_attachment_is_not_carbon"],
        )
    if is_carbonyl_carbon(mol, first):
        linkage_type = "ester"
    elif has_carbon_carbon_double_bond(mol, first):
        linkage_type = "vinyl_ether"
    else:
        linkage_type = "alkyl_ether"
    core = connected_component_from(mol, first, backbone_set | {oxygen_idx})
    return Chain(
        attachment_backbone_atom=backbone_c,
        attachment_hetero_atom=oxygen_idx,
        first_chain_atom=first,
        linkage_type=linkage_type,
        core_atom_indices=core,
        partition_atom_indices=sorted({oxygen_idx, *core}),
        backbone_position_candidate=position,
        warnings=warnings,
    )


def chain_to_dict(mol: Chem.Mol, chain: Chain, chain_index: int) -> dict[str, Any]:
    carbon_atoms = [idx for idx in chain.core_atom_indices if atom_symbol(mol, idx) == "C"]
    cc_double_bonds = 0
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        if a in chain.core_atom_indices and b in chain.core_atom_indices:
            if bond.GetBondType() == Chem.BondType.DOUBLE and atom_symbol(mol, a) == "C" and atom_symbol(mol, b) == "C":
                cc_double_bonds += 1
    ring_info = mol.GetRingInfo()
    ring_count = sum(1 for ring in ring_info.AtomRings() if set(ring) & set(chain.core_atom_indices))
    hetero_atoms = [idx for idx in chain.core_atom_indices if mol.GetAtomWithIdx(idx).GetAtomicNum() not in {1, 6}]
    oxygen_atoms = [idx for idx in chain.core_atom_indices if atom_symbol(mol, idx) == "O"]
    expected_oxygen = 1 if chain.linkage_type == "ester" and any(is_carbonyl_carbon(mol, idx) for idx in carbon_atoms) else 0
    branch_count = 0
    core_set = set(chain.core_atom_indices)
    for idx in carbon_atoms:
        carbon_degree = sum(1 for n in mol.GetAtomWithIdx(idx).GetNeighbors() if n.GetIdx() in core_set and n.GetAtomicNum() == 6)
        if carbon_degree > 2:
            branch_count += 1
    return {
        "chain_index": chain_index,
        "attachment_backbone_atom": chain.attachment_backbone_atom,
        "attachment_hetero_atom": chain.attachment_hetero_atom,
        "linkage_type": chain.linkage_type,
        "chain_atom_indices": chain.core_atom_indices,
        "chain_partition_atom_indices": chain.partition_atom_indices,
        "canonical_chain_smiles": fragment_smiles(mol, chain.core_atom_indices),
        "carbon_count": len(carbon_atoms),
        "graph_double_bond_count": cc_double_bonds,
        "nomenclature_double_bond_count": None,
        "ring_count": ring_count,
        "heteroatom_count": len(hetero_atoms),
        "additional_oxygen_count": max(0, len(oxygen_atoms) - expected_oxygen),
        "branch_count": branch_count,
        "backbone_position_candidate": chain.backbone_position_candidate,
        "warnings": chain.warnings,
    }


def glycerol_paths(mol: Chem.Mol) -> list[tuple[int, int, int]]:
    paths = []
    seen: set[frozenset[int]] = set()
    for middle in mol.GetAtoms():
        if not nonaromatic_carbon(middle) or middle.IsInRing():
            continue
        c_neighbors = [n.GetIdx() for n in middle.GetNeighbors() if nonaromatic_carbon(n) and not n.IsInRing()]
        for i, left in enumerate(c_neighbors):
            for right in c_neighbors[i + 1 :]:
                key = frozenset({left, middle.GetIdx(), right})
                if len(key) != 3 or key in seen:
                    continue
                seen.add(key)
                endpoints = sorted([left, right])
                paths.append((endpoints[0], middle.GetIdx(), endpoints[1]))
    return paths


def build_backbone_candidate(mol: Chem.Mol, path: tuple[int, int, int]) -> BackboneCandidate | None:
    backbone_set = set(path)
    p_sites: list[dict[str, int]] = []
    chains: list[Chain] = []
    free_sites: list[dict[str, int]] = []
    warnings: list[str] = []
    oxygen_site_count = 0
    for position, carbon_idx in enumerate(path):
        oxygens = [idx for idx in oxygen_neighbors(mol, carbon_idx) if idx not in backbone_set]
        oxygen_site_count += len(oxygens)
        if len(oxygens) != 1:
            warnings.append(f"backbone_carbon_{carbon_idx}_oxygen_neighbor_count_{len(oxygens)}")
        for oxygen_idx in oxygens:
            p_neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(oxygen_idx).GetNeighbors() if n.GetAtomicNum() == 15]
            if p_neighbors:
                p_sites.append({"backbone_atom": carbon_idx, "oxygen_atom": oxygen_idx, "phosphorus_atom": p_neighbors[0], "position": position})
                continue
            chain = classify_chain(mol, carbon_idx, oxygen_idx, backbone_set, position)
            if chain.linkage_type == "none":
                free_sites.append({"backbone_atom": carbon_idx, "oxygen_atom": oxygen_idx, "position": position})
            else:
                chains.append(chain)
    if len(p_sites) != 1:
        return None
    usable_chains = [chain for chain in chains if chain.linkage_type in {"ester", "alkyl_ether", "vinyl_ether"}]
    if len(usable_chains) not in {1, 2}:
        return None
    abnormal = sum(1 for chain in chains if chain.linkage_type not in {"ester", "alkyl_ether", "vinyl_ether"})
    score = 100
    score += len(usable_chains) * 25
    score += min(oxygen_site_count, 3) * 5
    score += max(0, 2 - abs(len(free_sites) - (2 - len(usable_chains)))) * 3
    score -= 20 * abnormal
    score -= 5 * len(warnings)
    p_site = p_sites[0]
    return BackboneCandidate(
        backbone_atom_indices=path,
        phosphorus_atom=p_site["phosphorus_atom"],
        headgroup_bridge_oxygen=p_site["oxygen_atom"],
        headgroup_attachment_site=p_site,
        chains=usable_chains,
        free_hydroxyl_sites=free_sites,
        score=score,
        warnings=warnings,
    )


def find_backbone_candidates(mol: Chem.Mol) -> list[BackboneCandidate]:
    candidates = []
    seen: set[tuple[frozenset[int], int, tuple[tuple[int, str], ...]]] = set()
    for path in glycerol_paths(mol):
        candidate = build_backbone_candidate(mol, path)
        if candidate is None:
            continue
        key = (
            frozenset(candidate.backbone_atom_indices),
            candidate.phosphorus_atom,
            tuple(sorted((chain.attachment_hetero_atom, chain.linkage_type) for chain in candidate.chains)),
        )
        if key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)
    return sorted(candidates, key=lambda c: c.score, reverse=True)


def p_oxygen_neighbors(mol: Chem.Mol, p_idx: int) -> list[int]:
    return [n.GetIdx() for n in mol.GetAtomWithIdx(p_idx).GetNeighbors() if n.GetAtomicNum() == 8]


def branch_component_from_p_oxygen(mol: Chem.Mol, p_idx: int, oxygen_idx: int, blocked: set[int]) -> list[int]:
    local_blocked = set(blocked)
    local_blocked.add(p_idx)
    return connected_component_from(mol, oxygen_idx, local_blocked)


def branch_stats(mol: Chem.Mol, atoms: Iterable[int]) -> dict[str, Any]:
    atom_set = set(atoms)
    carbon_atoms = [idx for idx in atom_set if atom_symbol(mol, idx) == "C"]
    oxygen_atoms = [idx for idx in atom_set if atom_symbol(mol, idx) == "O"]
    nitrogen_atoms = [idx for idx in atom_set if atom_symbol(mol, idx) == "N"]
    carbonyl_carbons = [idx for idx in carbon_atoms if is_carbonyl_carbon(mol, idx)]
    carboxyl_carbons = []
    for idx in carbonyl_carbons:
        atom = mol.GetAtomWithIdx(idx)
        has_single_o = any(
            bond.GetOtherAtom(atom).GetAtomicNum() == 8 and bond.GetBondType() == Chem.BondType.SINGLE for bond in atom.GetBonds()
        )
        if has_single_o:
            carboxyl_carbons.append(idx)
    ring_info = mol.GetRingInfo()
    carbon_six_rings = [
        ring for ring in ring_info.AtomRings() if len(ring) == 6 and set(ring) <= atom_set and all(atom_symbol(mol, idx) == "C" for idx in ring)
    ]
    n_methyl_counts = {}
    n_acyl_atoms = []
    for n_idx in nitrogen_atoms:
        methyl_count = 0
        for nbr in mol.GetAtomWithIdx(n_idx).GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if nbr_idx not in atom_set or nbr.GetAtomicNum() != 6:
                continue
            heavy_degree = sum(1 for nn in nbr.GetNeighbors() if nn.GetAtomicNum() > 1)
            if heavy_degree == 1:
                methyl_count += 1
            if is_carbonyl_carbon(mol, nbr_idx):
                n_acyl_atoms.append(nbr_idx)
        n_methyl_counts[n_idx] = methyl_count
    return {
        "atom_indices": sorted(atom_set),
        "carbon_count": len(carbon_atoms),
        "oxygen_count": len(oxygen_atoms),
        "nitrogen_atoms": nitrogen_atoms,
        "nitrogen_count": len(nitrogen_atoms),
        "nitrogen_formal_charges": [mol.GetAtomWithIdx(idx).GetFormalCharge() for idx in nitrogen_atoms],
        "n_methyl_counts": n_methyl_counts,
        "n_acyl_atoms": sorted(set(n_acyl_atoms)),
        "carbonyl_count": len(carbonyl_carbons),
        "carboxyl_count": len(carboxyl_carbons),
        "negative_oxygen_count": sum(1 for idx in oxygen_atoms if mol.GetAtomWithIdx(idx).GetFormalCharge() < 0),
        "carbon_six_ring_count": len(carbon_six_rings),
    }


def phosphorus_charge_tier(mol: Chem.Mol, p_idx: int, charged_user_variant: bool) -> int:
    if not charged_user_variant:
        return 1
    for oxygen_idx in p_oxygen_neighbors(mol, p_idx):
        if mol.GetAtomWithIdx(oxygen_idx).GetFormalCharge() < 0:
            return 1
    return 3


def classify_polar_branch_headgroup(mol: Chem.Mol, p_idx: int) -> dict[str, Any] | None:
    for oxygen_idx in p_oxygen_neighbors(mol, p_idx):
        atoms = branch_component_from_p_oxygen(mol, p_idx, oxygen_idx, set())
        stats = branch_stats(mol, atoms)
        if not stats["carbon_count"]:
            continue
        if stats["nitrogen_count"]:
            n_charges = stats["nitrogen_formal_charges"]
            max_methyl_count = max(stats["n_methyl_counts"].values() or [0])
            if max_methyl_count >= 3 and any(charge > 0 for charge in n_charges):
                tier = phosphorus_charge_tier(mol, p_idx, charged_user_variant=True)
                return {
                    "headgroup_id": "phosphocholine",
                    "headgroup_match_tier": tier,
                    "charge_normalization_used": tier >= 3,
                    "headgroup_atom_indices": sorted({p_idx, *atoms}),
                }
            if stats["carboxyl_count"]:
                charged_serine = any(charge > 0 for charge in n_charges) or stats["negative_oxygen_count"] > 0
                tier = 1 if charged_serine else 3
                return {
                    "headgroup_id": "phosphoserine",
                    "headgroup_match_tier": tier,
                    "charge_normalization_used": tier >= 3,
                    "headgroup_atom_indices": sorted({p_idx, *atoms}),
                }
            if max_methyl_count == 2:
                return {
                    "headgroup_id": "n_dimethyl_phosphoethanolamine",
                    "headgroup_match_tier": 1,
                    "charge_normalization_used": False,
                    "headgroup_atom_indices": sorted({p_idx, *atoms}),
                }
            if max_methyl_count == 1:
                return {
                    "headgroup_id": "n_methyl_phosphoethanolamine",
                    "headgroup_match_tier": 1,
                    "charge_normalization_used": False,
                    "headgroup_atom_indices": sorted({p_idx, *atoms}),
                }
            return {
                "headgroup_id": "phosphoethanolamine",
                "headgroup_match_tier": 1,
                "charge_normalization_used": False,
                "headgroup_atom_indices": sorted({p_idx, *atoms}),
            }
    return None


def has_sphingoid_like_features(mol: Chem.Mol) -> bool:
    nitrogen_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7)
    return nitrogen_count >= 2


def classify_headgroup(mol: Chem.Mol, candidate: BackboneCandidate) -> dict[str, Any]:
    p_idx = candidate.phosphorus_atom
    bridge_o = candidate.headgroup_bridge_oxygen
    blocked = set(candidate.backbone_atom_indices)
    branches = []
    for oxygen_idx in p_oxygen_neighbors(mol, p_idx):
        if oxygen_idx == bridge_o:
            continue
        atoms = branch_component_from_p_oxygen(mol, p_idx, oxygen_idx, blocked)
        stats = branch_stats(mol, atoms)
        if stats["carbon_count"]:
            branches.append({"attachment_oxygen": oxygen_idx, "atoms": atoms, "stats": stats})
    p_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 15)
    if p_count != 1:
        return {
            "status": "unsupported_topology",
            "reason": "multiple_phosphorus_atoms" if p_count > 1 else "no_phosphorus_atom",
            "headgroup_id": None,
            "match_tier": None,
            "warnings": [f"phosphorus_count={p_count}"],
        }
    if len(branches) == 0:
        return {
            "status": "matched",
            "headgroup_id": "phosphate",
            "headgroup_core": "phosphate",
            "match_tier": 1,
            "charge_normalization_used": False,
            "attachment_normalization_used": False,
            "warnings": [],
        }
    if len(branches) > 1:
        return {
            "status": "unsupported_topology",
            "reason": "multiple_phosphate_external_branches",
            "headgroup_id": None,
            "match_tier": None,
            "warnings": ["multiple carbon-bearing branches leave the target phosphate"],
        }
    branch = branches[0]
    stats = branch["stats"]
    warnings: list[str] = []
    if stats["n_acyl_atoms"]:
        return {
            "status": "unsupported_topology",
            "reason": "n_acyl_phosphoethanolamine_not_supported",
            "headgroup_id": "phosphoethanolamine",
            "match_tier": 1,
            "warnings": ["headgroup nitrogen carries an acyl substituent"],
        }
    if stats["nitrogen_count"]:
        n_charges = stats["nitrogen_formal_charges"]
        max_methyl_count = max(stats["n_methyl_counts"].values() or [0])
        if max_methyl_count >= 3 and any(charge > 0 for charge in n_charges):
            tier = phosphorus_charge_tier(mol, p_idx, charged_user_variant=True)
            return {
                "status": "matched",
                "headgroup_id": "phosphocholine",
                "headgroup_core": "phosphocholine",
                "match_tier": tier,
                "charge_normalization_used": tier >= 3,
                "attachment_normalization_used": False,
                "warnings": warnings,
            }
        if stats["carboxyl_count"]:
            charged_serine = any(charge > 0 for charge in n_charges) or stats["negative_oxygen_count"] > 0
            tier = 1 if charged_serine else 3
            return {
                "status": "matched",
                "headgroup_id": "phosphoserine",
                "headgroup_core": "phosphoserine",
                "match_tier": tier,
                "charge_normalization_used": tier >= 3,
                "attachment_normalization_used": False,
                "warnings": warnings,
            }
        if max_methyl_count == 2:
            return {
                "status": "matched",
                "headgroup_id": "n_dimethyl_phosphoethanolamine",
                "headgroup_core": "phosphoethanolamine",
                "match_tier": 1,
                "charge_normalization_used": False,
                "attachment_normalization_used": False,
                "warnings": warnings,
            }
        if max_methyl_count == 1:
            return {
                "status": "matched",
                "headgroup_id": "n_methyl_phosphoethanolamine",
                "headgroup_core": "phosphoethanolamine",
                "match_tier": 1,
                "charge_normalization_used": False,
                "attachment_normalization_used": False,
                "warnings": warnings,
            }
        return {
            "status": "matched",
            "headgroup_id": "phosphoethanolamine",
            "headgroup_core": "phosphoethanolamine",
            "match_tier": 1,
            "charge_normalization_used": False,
            "attachment_normalization_used": False,
            "warnings": warnings,
        }
    if stats["carbon_six_ring_count"] and stats["oxygen_count"] >= 5:
        return {
            "status": "matched",
            "headgroup_id": "phosphoinositol",
            "headgroup_core": "phosphoinositol",
            "match_tier": 1,
            "charge_normalization_used": False,
            "attachment_normalization_used": False,
            "warnings": warnings,
        }
    if stats["carbonyl_count"]:
        return {
            "status": "unsupported_topology",
            "reason": "acylated_or_carbonyl_headgroup_not_supported",
            "headgroup_id": None,
            "match_tier": None,
            "warnings": ["carbonyl-bearing phosphate external branch is not a v0.1 headgroup"],
        }
    if stats["carbon_count"] == 3 and stats["oxygen_count"] >= 3:
        return {
            "status": "matched",
            "headgroup_id": "phosphoglycerol",
            "headgroup_core": "phosphoglycerol",
            "match_tier": 1,
            "charge_normalization_used": False,
            "attachment_normalization_used": False,
            "warnings": warnings,
        }
    if stats["carbon_count"] == 1:
        return {
            "status": "matched",
            "headgroup_id": "phosphomethanol",
            "headgroup_core": "phosphomethanol",
            "match_tier": 1,
            "charge_normalization_used": False,
            "attachment_normalization_used": False,
            "warnings": warnings,
        }
    if stats["carbon_count"] == 2 and stats["oxygen_count"] <= 1:
        return {
            "status": "matched",
            "headgroup_id": "phosphoethanol",
            "headgroup_core": "phosphoethanol",
            "match_tier": 1,
            "charge_normalization_used": False,
            "attachment_normalization_used": False,
            "warnings": warnings,
        }
    return {
        "status": "failed",
        "reason": "unknown_headgroup_branch",
        "headgroup_id": None,
        "match_tier": None,
        "warnings": [f"unrecognized branch carbon_count={stats['carbon_count']} oxygen_count={stats['oxygen_count']}"],
    }


def headgroup_atoms(mol: Chem.Mol, candidate: BackboneCandidate, chain_partition_atoms: set[int]) -> list[int]:
    blocked = set(candidate.backbone_atom_indices) | set(chain_partition_atoms)
    atoms = connected_component_from(mol, candidate.phosphorus_atom, blocked)
    out = set(atoms)
    out.add(candidate.headgroup_bridge_oxygen)
    return sorted(out)


def derive_lipid_class(headgroup_id: str | None, chain_count: int) -> tuple[str | None, str]:
    if not headgroup_id:
        return None, "none"
    cls = SUPPORTED_CLASS_BY_HEADGROUP_AND_CHAIN_COUNT.get((headgroup_id, chain_count))
    if cls:
        return cls, "high"
    return None, "unsupported"


def partition_summary(mol: Chem.Mol, groups: dict[str, set[int]]) -> dict[str, Any]:
    heavy = set(heavy_atom_indices(mol))
    assigned_counter: Counter[int] = Counter()
    for atoms in groups.values():
        for idx in atoms:
            if idx in heavy:
                assigned_counter[idx] += 1
    overlap = sorted(idx for idx, count in assigned_counter.items() if count > 1)
    assigned = {idx for idx, count in assigned_counter.items() if count > 0}
    unassigned = sorted(heavy - assigned)
    return {
        "heavy_atom_coverage": (len(assigned - set(overlap)) / len(heavy)) if heavy else 1.0,
        "unassigned_atom_indices": unassigned,
        "overlapping_atom_indices": overlap,
        "unassigned_heavy_atom_count": len(unassigned),
        "overlap_heavy_atom_count": len(overlap),
    }


def reconstruction_summary(mol: Chem.Mol, groups: dict[str, set[int]]) -> dict[str, Any]:
    original_edges = heavy_edge_set(mol)
    atom_to_groups: dict[int, list[str]] = {}
    for name, atoms in groups.items():
        for idx in atoms:
            atom_to_groups.setdefault(idx, []).append(name)
    segment_edges = set()
    cut_bonds = []
    for edge in original_edges:
        a, b = edge
        ga = atom_to_groups.get(a, [])
        gb = atom_to_groups.get(b, [])
        if len(ga) == 1 and len(gb) == 1 and ga[0] == gb[0]:
            segment_edges.add(edge)
        else:
            cut_bonds.append({"atom_1": a, "atom_2": b, "groups_1": ga, "groups_2": gb})
    reconstructed = segment_edges | {tuple(sorted((bond["atom_1"], bond["atom_2"]))) for bond in cut_bonds}
    connectivity_exact = reconstructed == original_edges
    try:
        noniso = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        iso = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        noniso = None
        iso = None
    return {
        "cut_bonds": cut_bonds,
        "reconstruction_connectivity_exact": connectivity_exact,
        "reconstruction_nonisomeric_smiles_exact": connectivity_exact and bool(noniso),
        "reconstruction_isomeric_smiles_exact": connectivity_exact and bool(iso),
    }


def base_result(structure_record_id: str | None, smiles: str | None) -> dict[str, Any]:
    return {
        "structure_record_id": structure_record_id,
        "parser_version": PARSER_VERSION,
        "input_smiles": smiles,
        "parse_status": "failed",
        "failure_reasons": [],
        "headgroup_id": None,
        "headgroup_match_tier": None,
        "headgroup_atom_indices": [],
        "headgroup_canonical_smiles": None,
        "headgroup_formula": None,
        "headgroup_formal_charge": None,
        "headgroup_attachment_bond": None,
        "charge_normalization_used": False,
        "attachment_normalization_used": False,
        "backbone_family": None,
        "backbone_atom_indices": [],
        "headgroup_attachment_site": None,
        "chain_attachment_sites": [],
        "free_hydroxyl_sites": [],
        "backbone_confidence": "none",
        "chains": [],
        "chain_count": 0,
        "linkage_pattern": [],
        "cut_bonds": [],
        "unassigned_atom_indices": [],
        "overlapping_atom_indices": [],
        "heavy_atom_coverage": 0.0,
        "unassigned_heavy_atom_count": None,
        "overlap_heavy_atom_count": None,
        "sn_assignment_status": "unknown",
        "derived_lipid_class_candidate": None,
        "class_derivation_confidence": "none",
        "reconstruction_connectivity_exact": False,
        "reconstruction_nonisomeric_smiles_exact": False,
        "reconstruction_isomeric_smiles_exact": False,
        "warnings": [],
    }


def parse_glycerophospholipid_smiles(smiles: str | None, structure_record_id: str | None = None) -> dict[str, Any]:
    result = base_result(structure_record_id, smiles)
    if not smiles:
        result["parse_status"] = "invalid_input"
        result["failure_reasons"].append("missing_smiles")
        return result
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        result["parse_status"] = "invalid_input"
        result["failure_reasons"].append("invalid_smiles")
        return result
    p_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 15]
    if not p_atoms:
        result["parse_status"] = "unsupported_backbone"
        result["failure_reasons"].append("no_phosphorus_atom")
        return result
    if len(p_atoms) > 1:
        result["parse_status"] = "unsupported_topology"
        result["failure_reasons"].append("multiple_phosphorus_atoms")
        return result

    candidates = find_backbone_candidates(mol)
    if not candidates:
        unsupported_headgroup = classify_polar_branch_headgroup(mol, p_atoms[0])
        if unsupported_headgroup:
            result.update(unsupported_headgroup)
        result["backbone_family"] = "sphingoid_or_unsupported" if has_sphingoid_like_features(mol) else "unsupported"
        result["parse_status"] = "unsupported_backbone"
        result["failure_reasons"].append("no_typical_glycerol_phosphate_backbone")
        return result
    top_score = candidates[0].score
    top_candidates = [candidate for candidate in candidates if candidate.score == top_score]
    if len(top_candidates) > 1:
        result["parse_status"] = "ambiguous_backbone"
        result["failure_reasons"].append("multiple_equivalent_glycerol_backbone_candidates")
        result["backbone_atom_indices"] = [list(candidate.backbone_atom_indices) for candidate in top_candidates]
        return result
    candidate = top_candidates[0]
    headgroup = classify_headgroup(mol, candidate)
    if headgroup["status"] != "matched":
        result["parse_status"] = headgroup["status"]
        result["failure_reasons"].append(headgroup.get("reason") or "headgroup_not_matched")
        result["warnings"].extend(headgroup.get("warnings") or [])
        result["headgroup_id"] = headgroup.get("headgroup_id")
        result["headgroup_match_tier"] = headgroup.get("match_tier")
        result["backbone_family"] = "glycerol"
        result["backbone_atom_indices"] = list(candidate.backbone_atom_indices)
        result["backbone_confidence"] = "high"
        return result

    chains = sorted(candidate.chains, key=lambda chain: (chain.backbone_position_candidate, chain.attachment_hetero_atom))
    chain_count = len([chain for chain in chains if chain.linkage_type in {"ester", "alkyl_ether", "vinyl_ether"}])
    derived_class, confidence = derive_lipid_class(headgroup["headgroup_id"], chain_count)
    if derived_class is None:
        result["parse_status"] = "unsupported_topology"
        result["failure_reasons"].append("headgroup_chain_count_combination_not_supported")
        result["headgroup_id"] = headgroup["headgroup_id"]
        result["headgroup_match_tier"] = headgroup["match_tier"]
        result["backbone_family"] = "glycerol"
        result["backbone_atom_indices"] = list(candidate.backbone_atom_indices)
        result["chain_count"] = chain_count
        return result

    chain_dicts = [chain_to_dict(mol, chain, idx) for idx, chain in enumerate(chains)]
    chain_partition = set()
    for chain in chains:
        chain_partition.update(chain.partition_atom_indices)
    h_atoms = set(headgroup_atoms(mol, candidate, chain_partition))
    free_hydroxyl_atoms = {site["oxygen_atom"] for site in candidate.free_hydroxyl_sites}
    groups = {
        "headgroup": h_atoms,
        "backbone": set(candidate.backbone_atom_indices),
        "chains": set(chain_partition),
        "free_hydroxyls": free_hydroxyl_atoms,
    }
    partition = partition_summary(mol, groups)
    reconstruction = reconstruction_summary(mol, groups)
    result.update(
        {
            "parse_status": "success",
            "headgroup_id": headgroup["headgroup_id"],
            "headgroup_match_tier": headgroup["match_tier"],
            "headgroup_atom_indices": sorted(h_atoms),
            "headgroup_canonical_smiles": fragment_smiles(mol, h_atoms),
            "headgroup_formula": formula_for_atoms(mol, h_atoms),
            "headgroup_formal_charge": formal_charge_for_atoms(mol, h_atoms),
            "headgroup_attachment_bond": {
                "backbone_atom": candidate.headgroup_attachment_site["backbone_atom"],
                "oxygen_atom": candidate.headgroup_bridge_oxygen,
                "phosphorus_atom": candidate.phosphorus_atom,
            },
            "charge_normalization_used": headgroup["charge_normalization_used"],
            "attachment_normalization_used": headgroup["attachment_normalization_used"],
            "backbone_family": "glycerol",
            "backbone_atom_indices": list(candidate.backbone_atom_indices),
            "headgroup_attachment_site": candidate.headgroup_attachment_site,
            "chain_attachment_sites": [
                {
                    "backbone_atom": chain.attachment_backbone_atom,
                    "oxygen_atom": chain.attachment_hetero_atom,
                    "position": chain.backbone_position_candidate,
                    "linkage_type": chain.linkage_type,
                }
                for chain in chains
            ],
            "free_hydroxyl_sites": candidate.free_hydroxyl_sites,
            "free_hydroxyl_atom_indices": sorted(free_hydroxyl_atoms),
            "backbone_confidence": "high",
            "chains": chain_dicts,
            "chain_count": chain_count,
            "linkage_pattern": sorted(chain.linkage_type for chain in chains),
            "sn_assignment_status": "graph_position_only",
            "derived_lipid_class_candidate": derived_class,
            "class_derivation_confidence": confidence,
            "warnings": sorted(set(candidate.warnings + headgroup.get("warnings", []))),
        }
    )
    result.update(partition)
    result.update(reconstruction)
    if result["overlap_heavy_atom_count"] or result["unassigned_heavy_atom_count"]:
        result["parse_status"] = "unsupported_extra_substitution"
        result["failure_reasons"].append("atom_partition_incomplete")
        result["failure_reasons"].append("unsupported_extra_substitution")
    if not result["reconstruction_connectivity_exact"]:
        result["parse_status"] = "failed"
        result["failure_reasons"].append("reconstruction_connectivity_mismatch")
    return result


def parse_phospholipid_structure(smiles: str | None, structure_record_id: str | None = None) -> dict[str, Any]:
    return parse_glycerophospholipid_smiles(smiles, structure_record_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse typical glycerophospholipid structures from SMILES.")
    parser.add_argument("smiles", nargs="?", help="SMILES string to parse")
    parser.add_argument("--structure-record-id")
    args = parser.parse_args()
    print(json.dumps(parse_glycerophospholipid_smiles(args.smiles, args.structure_record_id), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
