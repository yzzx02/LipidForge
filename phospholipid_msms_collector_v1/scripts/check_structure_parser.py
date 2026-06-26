#!/usr/bin/env python3
from __future__ import annotations

from rdkit import Chem

from phospholipid_structure_parser import parse_glycerophospholipid_smiles, parse_phospholipid_structure


PC_DIESTER = "CCCCCCCCCCCCCCCC(=O)OCC(COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCCCCCCCCCC"
LPC_MONOESTER = "CCCCCCCCCC(=O)OCC(O)COP(=O)(O)OCC[N+](C)(C)C"
PE_DIESTER = "CCCCCCCCCCCCCCCC(=O)OCC(COP(=O)(O)OCCN)OC(=O)CCCCCCCCCCCCCCC"
PS_NEUTRAL = "CCCCCCCCCCCCCCCC(=O)OCC(COP(=O)(O)OCC(N)C(=O)O)OC(=O)CCCCCCCCCCCCCCC"
PS_ZWITTERIONIC = "CCCCCCCCCCCCCCCC(=O)OCC(COP(=O)(O)OCC([NH3+])C(=O)[O-])OC(=O)CCCCCCCCCCCCCCC"
PC_ALKYL_ETHER = "CCCCCCCCCCCCCCCCOCC(COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCCCCCCCCCC"
PC_VINYL_ETHER = "CCCCCCCCCCCCCC/C=C/OCC(COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCCCCCCCCCC"
GLYCEROPHOSPHATE_ZERO_CHAIN = "O=P(O)(O)OCC(O)CO"
C_METHYL_PC = "CCCCCCCCCCCCCCCCOCC(C)(O)COP(=O)([O-])OCC[N+](C)(C)C"
BMP_LIKE_AMBIGUOUS = "CCCCCCCC(=O)OCC(O)COP(=O)(O)OCC(O)COC(=O)CCCCCCC"
MULTI_PHOSPHORUS = "O=P(O)(O)OCC(O)COP(=O)(O)O"
SM_NEGATIVE = "C[N+](C)(C)CCOP([O-])(=O)OC[C@H](NC(=O)CCCCCCCCCCCCCCCCC)[C@H](O)\\C=C\\CCCCCCCCCCCCC"
S1P_NEGATIVE = "CCCCCCCCCCCC/C=C/[C@@H](O)[C@H](N)COP(=O)(O)O"
SPHINGOSYL_PE = "[H]OC([H])(C([H])=C([H])CCCCCCCCCCCC)C([H])(N([H])[H])C([H])([H])OP(=O)([O-])OCC[N+]([H])([H])[H]"


def heavy_edges(mol: Chem.Mol) -> set[tuple[int, int]]:
    ignored_atomic_nums = {3, 11, 12, 19, 20, 17, 35, 53}
    out = set()
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        if mol.GetAtomWithIdx(a).GetAtomicNum() in ignored_atomic_nums or mol.GetAtomWithIdx(b).GetAtomicNum() in ignored_atomic_nums:
            continue
        if mol.GetAtomWithIdx(a).GetAtomicNum() > 1 and mol.GetAtomWithIdx(b).GetAtomicNum() > 1:
            out.add(tuple(sorted((a, b))))
    return out


def assert_partition_reconstructs(row: dict) -> None:
    mol = Chem.MolFromSmiles(row["input_smiles"])
    assert mol is not None
    groups = {
        "headgroup": set(row["headgroup_atom_indices"]),
        "backbone": set(row["backbone_atom_indices"]),
        "free_hydroxyl": set(row.get("free_hydroxyl_atom_indices") or []),
    }
    chain_atoms = set()
    for chain in row["chains"]:
        chain_atoms.update(chain["chain_partition_atom_indices"])
    groups["chains"] = chain_atoms
    atom_to_group = {}
    for group, atoms in groups.items():
        for idx in atoms:
            atom_to_group.setdefault(idx, []).append(group)
    segment_edges = set()
    for edge in heavy_edges(mol):
        a, b = edge
        if atom_to_group.get(a) == atom_to_group.get(b) and len(atom_to_group.get(a, [])) == 1:
            segment_edges.add(edge)
    cut_edges = {tuple(sorted((int(bond["atom_1"]), int(bond["atom_2"])))) for bond in row["cut_bonds"]}
    assert segment_edges | cut_edges == heavy_edges(mol), row


def assert_missing_partition_atom_blocks_high_confidence(row: dict) -> None:
    mol = Chem.MolFromSmiles(row["input_smiles"])
    assert mol is not None
    assigned = set(row["headgroup_atom_indices"]) | set(row["backbone_atom_indices"]) | set(row.get("free_hydroxyl_atom_indices") or [])
    for chain in row["chains"]:
        assigned.update(chain["chain_partition_atom_indices"])
    removed = next(iter(row["chains"][0]["chain_atom_indices"]))
    assigned.remove(removed)
    heavy = {idx for edge in heavy_edges(mol) for idx in edge}
    assert removed in heavy - assigned, row
    assert len(assigned & heavy) / len(heavy) < 1.0, row


def assert_success(row: dict, expected_class: str, expected_headgroup: str, chain_count: int) -> None:
    assert row["parse_status"] == "success", row
    assert row["derived_lipid_class_candidate"] == expected_class, row
    assert row["headgroup_id"] == expected_headgroup, row
    assert row["chain_count"] == chain_count, row
    assert row["overlap_heavy_atom_count"] == 0, row
    assert row["unassigned_heavy_atom_count"] == 0, row
    assert row["reconstruction_connectivity_exact"] is True, row


def test_pc_diester() -> None:
    row = parse_glycerophospholipid_smiles(PC_DIESTER)
    assert_success(row, "PC", "phosphocholine", 2)
    assert row["linkage_pattern"] == ["ester", "ester"], row


def test_lpc_monoester_and_free_hydroxyl() -> None:
    row = parse_glycerophospholipid_smiles(LPC_MONOESTER)
    assert_success(row, "LPC", "phosphocholine", 1)
    assert row["linkage_pattern"] == ["ester"], row
    assert len(row["free_hydroxyl_sites"]) == 1, row
    assert row["headgroup_match_tier"] == 3, row


def test_zero_chain_glycerophosphate_is_not_pa() -> None:
    row = parse_glycerophospholipid_smiles(GLYCEROPHOSPHATE_ZERO_CHAIN)
    assert row["parse_status"] == "unsupported_backbone", row
    assert row["derived_lipid_class_candidate"] is None, row
    assert row["chain_count"] == 0, row


def test_pc_label_would_not_override_lpc_graph() -> None:
    row = parse_glycerophospholipid_smiles(LPC_MONOESTER, structure_record_id="strict_label_pc_but_graph_lpc")
    assert_success(row, "LPC", "phosphocholine", 1)


def test_pe_pc_headgroup_distinction() -> None:
    pc = parse_glycerophospholipid_smiles(PC_DIESTER)
    pe = parse_glycerophospholipid_smiles(PE_DIESTER)
    assert pc["headgroup_id"] == "phosphocholine", pc
    assert pe["headgroup_id"] == "phosphoethanolamine", pe
    assert pc["derived_lipid_class_candidate"] == "PC", pc
    assert pe["derived_lipid_class_candidate"] == "PE", pe


def test_ps_neutral_and_zwitterionic_equivalence() -> None:
    neutral = parse_glycerophospholipid_smiles(PS_NEUTRAL)
    zwitterionic = parse_glycerophospholipid_smiles(PS_ZWITTERIONIC)
    assert_success(neutral, "PS", "phosphoserine", 2)
    assert_success(zwitterionic, "PS", "phosphoserine", 2)
    assert neutral["headgroup_match_tier"] == 3, neutral
    assert zwitterionic["headgroup_match_tier"] == 1, zwitterionic
    assert neutral["charge_normalization_used"] is True, neutral


def test_linkage_types() -> None:
    ester = parse_glycerophospholipid_smiles(PC_DIESTER)
    alkyl = parse_glycerophospholipid_smiles(PC_ALKYL_ETHER)
    vinyl = parse_glycerophospholipid_smiles(PC_VINYL_ETHER)
    assert ester["linkage_pattern"] == ["ester", "ester"], ester
    assert alkyl["linkage_pattern"] == ["alkyl_ether", "ester"], alkyl
    assert vinyl["linkage_pattern"] == ["ester", "vinyl_ether"], vinyl
    assert_partition_reconstructs(ester)
    assert_partition_reconstructs(alkyl)
    assert_partition_reconstructs(vinyl)
    assert_missing_partition_atom_blocks_high_confidence(ester)


def test_sphingosyl_pe_is_unsupported_backbone() -> None:
    row = parse_glycerophospholipid_smiles(SPHINGOSYL_PE)
    assert row["parse_status"] == "unsupported_backbone", row
    assert row["headgroup_id"] == "phosphoethanolamine", row
    assert row["backbone_family"] == "sphingoid_or_unsupported", row
    assert row["derived_lipid_class_candidate"] is None, row


def test_c_methyl_pc_is_not_silent_success() -> None:
    row = parse_glycerophospholipid_smiles(C_METHYL_PC)
    assert row["parse_status"] == "unsupported_extra_substitution", row
    assert row["failure_reasons"] == ["atom_partition_incomplete", "unsupported_extra_substitution"], row
    assert row["unassigned_heavy_atom_count"] == 1, row
    assert row["reconstruction_connectivity_exact"] is True, row


def test_ambiguous_backbone() -> None:
    row = parse_glycerophospholipid_smiles(BMP_LIKE_AMBIGUOUS)
    assert row["parse_status"] in {"ambiguous_backbone", "unsupported_topology"}, row
    assert row["derived_lipid_class_candidate"] not in {"PG", "LPG"}, row


def test_negative_controls() -> None:
    sm = parse_glycerophospholipid_smiles(SM_NEGATIVE)
    s1p = parse_glycerophospholipid_smiles(S1P_NEGATIVE)
    multi_p = parse_glycerophospholipid_smiles(MULTI_PHOSPHORUS)
    assert sm["derived_lipid_class_candidate"] != "PC", sm
    assert sm["parse_status"] in {"unsupported_backbone", "unsupported_topology", "failed"}, sm
    assert s1p["derived_lipid_class_candidate"] not in {"PA", "LPA"}, s1p
    assert s1p["parse_status"] in {"unsupported_backbone", "unsupported_topology", "failed"}, s1p
    assert multi_p["parse_status"] == "unsupported_topology", multi_p
    assert multi_p["failure_reasons"] == ["multiple_phosphorus_atoms"], multi_p


def test_atom_order_stability() -> None:
    mol = Chem.MolFromSmiles(PC_DIESTER)
    assert mol is not None
    order = list(reversed(range(mol.GetNumAtoms())))
    renumbered = Chem.RenumberAtoms(mol, order)
    scrambled = Chem.MolToSmiles(renumbered, canonical=False, isomericSmiles=True)
    first = parse_glycerophospholipid_smiles(PC_DIESTER)
    second = parse_glycerophospholipid_smiles(scrambled)
    comparable_fields = [
        "parse_status",
        "headgroup_id",
        "chain_count",
        "linkage_pattern",
        "derived_lipid_class_candidate",
        "reconstruction_connectivity_exact",
    ]
    for field in comparable_fields:
        assert first[field] == second[field], (field, first, second)


def test_parser_does_not_read_lipid_class() -> None:
    row = parse_phospholipid_structure(PC_DIESTER, structure_record_id="not_a_label")
    assert row["derived_lipid_class_candidate"] == "PC", row
    assert "lipid_class" not in row, row


def test_malformed_smiles_returns_invalid_input() -> None:
    row = parse_glycerophospholipid_smiles("C1CC")
    assert row["parse_status"] == "invalid_input", row
    assert row["failure_reasons"] == ["invalid_smiles"], row


def main() -> None:
    test_pc_diester()
    test_lpc_monoester_and_free_hydroxyl()
    test_zero_chain_glycerophosphate_is_not_pa()
    test_pc_label_would_not_override_lpc_graph()
    test_pe_pc_headgroup_distinction()
    test_ps_neutral_and_zwitterionic_equivalence()
    test_linkage_types()
    test_sphingosyl_pe_is_unsupported_backbone()
    test_c_methyl_pc_is_not_silent_success()
    test_ambiguous_backbone()
    test_negative_controls()
    test_atom_order_stability()
    test_parser_does_not_read_lipid_class()
    test_malformed_smiles_returns_invalid_input()
    print("structure parser self-checks passed")


if __name__ == "__main__":
    main()
