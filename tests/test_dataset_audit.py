from pathlib import Path

from PIL import Image

from hwnas_fpga.data.audit import audit_nksid_protocol


def _write_dataset(root: Path) -> Path:
    dataset = root / "NKSID"
    dataset.mkdir()
    rows = []
    for label in range(2):
        class_dir = dataset / f"class_{label}"
        class_dir.mkdir()
        for number in range(4):
            path = class_dir / f"img_{number}.png"
            Image.new("L", (8, 8), color=label * 64 + number).save(path)
            rows.append((path.relative_to(dataset).as_posix(), label))
    (dataset / "train_abs.txt").write_text(
        "".join(f"{path} {label}\n" for path, label in rows),
        encoding="utf-8",
    )
    # Two-fold cycle repeated twice, intentionally image-index based.
    val_rows = ([0, 2, 4, 6], [1, 3, 5, 7]) * 2
    train_rows = ([1, 3, 5, 7], [0, 2, 4, 6]) * 2
    (dataset / "kfold_train.txt").write_text(
        "".join(" ".join(map(str, row)) + "\n" for row in train_rows),
        encoding="utf-8",
    )
    (dataset / "kfold_val.txt").write_text(
        "".join(" ".join(map(str, row)) + "\n" for row in val_rows),
        encoding="utf-8",
    )
    return dataset


def test_protocol_audit_identifies_repeated_cycle_and_adjacency(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path)
    result = audit_nksid_protocol(dataset, fold=0, hash_files=True)

    assert result["critical_failures"] == []
    assert result["split_protocol"]["record_count"] == 4
    assert result["split_protocol"]["inferred_k"] == 2
    assert result["split_protocol"]["inferred_repeats"] == 2
    assert result["selected_fold"]["filename_neighbor_fraction"] == 1.0
    assert "fold_records_are_repeated_splits_not_a_single_kfold_cycle" in result["warnings"]
    assert "image_level_split_has_high_filename_neighbor_leakage_risk" in result["warnings"]


def test_protocol_audit_rejects_negative_fold_index(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path)
    train_path = dataset / "kfold_train.txt"
    rows = train_path.read_text(encoding="utf-8").splitlines()
    rows[0] = f"-1 {rows[0]}"
    train_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    result = audit_nksid_protocol(dataset, fold=0)

    assert "fold_index_integrity_failed" in result["critical_failures"]
    assert result["split_protocol"]["fold_rows"][0]["out_of_range_count"] == 1
