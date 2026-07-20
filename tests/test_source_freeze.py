from scripts.freeze_experiment_source import is_excluded


def test_vivado_transient_directory_is_excluded_from_source_freeze() -> None:
    assert is_excluded(".Xil/Vivado-123-workstation/.lpr")
    assert is_excluded(".Xil")


def test_source_files_are_not_excluded() -> None:
    assert not is_excluded("run_eval_protocol.py")
    assert not is_excluded("src/hwnas_fpga/training/trainer.py")
