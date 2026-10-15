#!/usr/bin/env python3
"""Phase 3C fixed B1 transfer runner for UATD v3 (MFLS)."""

from architecture_transfer_common import DatasetRegistration, ROOT, run_cli


REGISTRATION = DatasetRegistration(
    dataset_id="uatd_v3",
    dataset_name="UATD v3",
    modality="MFLS",
    num_classes=10,
    clean_root=ROOT / "datasets/UATD/clean_phase3b",
    split_manifest=ROOT / "datasets/UATD/metadata/phase3b_clean_split_manifest.json",
    output_dir=ROOT / "results/transfer/UATD",
    expected_identity="UATD v3",
)


if __name__ == "__main__":
    run_cli(REGISTRATION)
