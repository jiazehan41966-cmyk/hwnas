from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from architecture_transfer_common import B1_ID, V01_SPEC, load_b1
from hwnas_fpga.models.sonar_supernet import architecture_id


def test_transfer_architecture_is_exact_frozen_b1():
    binding = load_b1()
    assert binding["architecture_id"] == B1_ID
    assert architecture_id(binding["architecture"], spec=V01_SPEC) == B1_ID
    assert binding["architecture"]["output_combination"] == "P2_P3_P4_P5"
    assert binding["architecture"]["kernels"] == {
        "stage1": 3, "stage2": 5, "stage4": 5, "stage5": 5, "stage6": 5, "stage7": 5
    }


def test_transfer_runners_do_not_use_astar_term():
    for name in ("run_uatd_architecture_transfer.py", "run_sctd_architecture_transfer.py", "architecture_transfer_common.py"):
        text = (ROOT / "scripts" / name).read_text(encoding="utf-8").lower()
        assert "a_star" not in text
        assert "a*" not in text

