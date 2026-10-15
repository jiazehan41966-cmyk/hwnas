#!/usr/bin/env python3
"""Phase 3C fixed B1 transfer runner for classic SCTD / SCTD 1.0 (SSS)."""

from architecture_transfer_common import DatasetRegistration, ROOT, run_cli


REGISTRATION = DatasetRegistration(
    dataset_id="classic_sctd_1_0",
    dataset_name="classic SCTD / SCTD 1.0",
    modality="SSS",
    num_classes=3,
    clean_root=ROOT / "datasets/SCTD/clean_phase3b_v2",
    split_manifest=ROOT / "datasets/SCTD/metadata/phase3b_clean_split_manifest_v2.json",
    output_dir=ROOT / "results/transfer/SCTD",
    expected_identity="classic SCTD / SCTD 1.0",
)


if __name__ == "__main__":
    run_cli(REGISTRATION)
