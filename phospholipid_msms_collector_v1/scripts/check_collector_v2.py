#!/usr/bin/env python3
from __future__ import annotations

from collect_phospholipid_msms import (
    assign_duplicate_groups,
    classify_duplicate_relation,
    enrich_record_v2,
    extract_massbank_adduct,
    normalize_adduct_value,
    normalize_identifier_fields,
    standardize_structure_identity,
)


def minimal_record(**overrides):
    rec = {
        "spectrum_id": "test:1",
        "source": "unit",
        "source_file": "unit",
        "source_release": None,
        "license": "unit test",
        "record_title": "unit",
        "name": "unit",
        "all_names": ["unit"],
        "lipid_class": "PC",
        "lipid_family": "glycerophospholipid",
        "linkage_modifications": [],
        "classification_source": "unit",
        "polarity": "positive",
        "precursor_mz": 123.456789,
        "adduct": "[M+H]+",
        "adduct_raw": "[M+H]+",
        "adduct_source_field": "unit",
        "adduct_normalization_status": "normalized",
        "adduct_warnings": [],
        "collision_energy": "25 eV",
        "instrument": "Q Exactive",
        "instrument_type": "Orbitrap",
        "formula": "C2H4O2",
        "exact_mass": 60.0,
        "smiles": "CC(=O)O",
        "inchikey": None,
        "lipidmaps_id": None,
        "peaks_raw": [[100.0, 50.0], [101.0, 10.0], [102.0, 5.0]],
        "num_peaks": 3,
        "experimental": True,
    }
    rec.update(overrides)
    return rec


def test_adduct_priority_and_formats() -> None:
    info = extract_massbank_adduct({
        "MS$FOCUSED_ION": ["ION_TYPE [M+Na]+", "PRECURSOR_TYPE [M+H]+"],
        "AC$MASS_SPECTROMETRY": ["PRECURSOR_TYPE [M+K]+"],
    })
    assert info["adduct"] == "[M+H]+"
    assert info["adduct_source_field"] == "MS$FOCUSED_ION: PRECURSOR_TYPE"

    raw, adduct, status, warnings = normalize_adduct_value("POSITIVE")
    assert raw == "POSITIVE"
    assert adduct == "unknown"
    assert status == "ion_mode_not_adduct"
    assert warnings

    raw, adduct, status, warnings = normalize_adduct_value("NEGATIVE")
    assert raw == "NEGATIVE"
    assert adduct == "unknown"
    assert status == "ion_mode_not_adduct"
    assert warnings

    raw, adduct, status, warnings = normalize_adduct_value("")
    assert raw is None
    assert adduct == "unknown"
    assert status == "missing"
    assert warnings

    raw, adduct, status, warnings = normalize_adduct_value("mystery")
    assert raw == "mystery"
    assert adduct == "unknown"
    assert status == "unparsed"
    assert warnings

    examples = {
        "[M+H]+": "[M+H]+",
        "[M-H]-": "[M-H]-",
        "[M+Na]+": "[M+Na]+",
        "[m + nh4]+": "[M+NH4]+",
        "[M+K]+": "[M+K]+",
        "[M+Cl]-": "[M+Cl]-",
        "[M+CH3COO]-": "[M+CH3COO]-",
        "[M+HCOO]-": "[M+HCOO]-",
        "[M-2H]2-": "[M-2H]2-",
    }
    for raw_value, expected in examples.items():
        assert normalize_adduct_value(raw_value)[1] == expected


def test_identifier_semantics() -> None:
    full = normalize_identifier_fields("BSYNRYMUTXBXSQ-UHFFFAOYSA-N")
    assert full["provided_full_inchikey"] == "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
    assert full["provided_connectivity_key"] == "BSYNRYMUTXBXSQ"
    assert full["inchikey_semantics"] == "full_inchikey"

    conn = normalize_identifier_fields("VFMQMACUYWGDOJ")
    assert conn["provided_full_inchikey"] is None
    assert conn["provided_connectivity_key"] == "VFMQMACUYWGDOJ"
    assert conn["inchikey_semantics"] == "connectivity_block"


def test_structure_and_connectivity_ids() -> None:
    acid = standardize_structure_identity("CC(=O)O")
    salt = standardize_structure_identity("CC(=O)[O-].[Na+]")
    assert acid["connectivity_id"] == salt["connectivity_id"]
    assert acid["structure_record_id"] != salt["structure_record_id"]
    assert salt["has_multiple_fragments"]
    assert "[Na+]" in salt["likely_counterions"]

    left = standardize_structure_identity("C[C@H](O)C(=O)O")
    right = standardize_structure_identity("C[C@@H](O)C(=O)O")
    assert left["connectivity_id"] == right["connectivity_id"]
    assert left["structure_record_id"] != right["structure_record_id"]
    assert standardize_structure_identity("CC(=O)O")["structure_record_id"] == acid["structure_record_id"]

    forward = standardize_structure_identity("CCO.CCN")
    reverse = standardize_structure_identity("CCN.CCO")
    assert forward["connectivity_id"] == reverse["connectivity_id"]
    assert forward["major_component_smiles"] == reverse["major_component_smiles"]

    duplicated_fragment = standardize_structure_identity("CCO.CCO")
    assert duplicated_fragment["minor_fragment_smiles"] == ["CCO"]


def test_hash_boundaries() -> None:
    first = enrich_record_v2(minimal_record())
    repeat = enrich_record_v2(minimal_record())
    reordered_peaks = enrich_record_v2(minimal_record(peaks_raw=[[102.0, 5.0], [100.0, 50.0], [101.0, 10.0]]))
    changed_source_record = enrich_record_v2(minimal_record(spectrum_id="test:2"))
    changed_ce = enrich_record_v2(minimal_record(collision_energy="35 eV"))
    changed_instrument = enrich_record_v2(minimal_record(instrument="Orbitrap Exploris"))
    changed_peak = enrich_record_v2(minimal_record(peaks_raw=[[100.0, 51.0], [101.0, 10.0], [102.0, 5.0]]))

    assert first["structure_record_id"] == repeat["structure_record_id"]
    assert first["connectivity_id"] == repeat["connectivity_id"]
    assert first["peak_identity_hash"] == repeat["peak_identity_hash"]
    assert first["acquisition_record_hash"] == repeat["acquisition_record_hash"]
    assert first["peak_identity_hash"] == reordered_peaks["peak_identity_hash"]
    assert first["peak_identity_hash"] == changed_source_record["peak_identity_hash"]
    assert first["acquisition_metadata_hash"] == changed_source_record["acquisition_metadata_hash"]
    assert first["acquisition_record_hash"] != changed_source_record["acquisition_record_hash"]
    assert first["peak_identity_hash"] == changed_ce["peak_identity_hash"]
    assert first["acquisition_metadata_hash"] != changed_ce["acquisition_metadata_hash"]
    assert first["peak_identity_hash"] == changed_instrument["peak_identity_hash"]
    assert first["acquisition_metadata_hash"] != changed_instrument["acquisition_metadata_hash"]
    assert first["peak_identity_hash"] != changed_peak["peak_identity_hash"]


def test_duplicate_relation_names() -> None:
    first = enrich_record_v2(minimal_record())
    repeat = enrich_record_v2(minimal_record())
    assign_duplicate_groups([first, repeat])
    assert first["duplicate_group_size"] == 2
    assert first["duplicate_relation"] == "removable_exact_duplicate"

    first = enrich_record_v2(minimal_record())
    same_acquisition_different_record = enrich_record_v2(minimal_record(spectrum_id="test:2"))
    assign_duplicate_groups([first, same_acquisition_different_record])
    assert first["duplicate_relation"] == "same_source_acquisition_duplicate_candidate"

    first = enrich_record_v2(minimal_record())
    different_ce = enrich_record_v2(minimal_record(spectrum_id="test:2", collision_energy="35 eV"))
    assign_duplicate_groups([first, different_ce])
    assert first["duplicate_relation"] == "same_peaks_different_collision_energy"

    first = enrich_record_v2(minimal_record())
    different_instrument = enrich_record_v2(minimal_record(spectrum_id="test:2", instrument="Orbitrap Exploris"))
    assign_duplicate_groups([first, different_instrument])
    assert first["duplicate_relation"] == "same_peaks_different_instrument"

    assert classify_duplicate_relation([enrich_record_v2(minimal_record())]) == "unique"


def test_v2_schema_fields() -> None:
    rec = enrich_record_v2(minimal_record())
    expected_fields = {
        "source",
        "source_record_id",
        "license",
        "provided_identifier_raw",
        "provided_full_inchikey",
        "provided_connectivity_key",
        "inchikey_semantics",
        "structure_record_id",
        "connectivity_id",
        "canonical_isomeric_smiles",
        "canonical_nonisomeric_smiles",
        "major_component_smiles",
        "minor_fragment_smiles",
        "fragment_signature",
        "collision_energy_raw",
        "collision_energy_normalized",
        "collision_energy_unit",
        "collision_energy_parse_status",
        "instrument",
        "instrument_type",
        "peak_identity_hash",
        "acquisition_metadata_hash",
        "acquisition_record_hash",
        "duplicate_group_id",
        "duplicate_relation",
    }
    missing = sorted(expected_fields - rec.keys())
    assert not missing, missing


def main() -> None:
    test_adduct_priority_and_formats()
    test_identifier_semantics()
    test_structure_and_connectivity_ids()
    test_hash_boundaries()
    test_duplicate_relation_names()
    test_v2_schema_fields()
    print("collector v2 self-checks passed")


if __name__ == "__main__":
    main()
