import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('rtl',Path(__file__).resolve().parents[1]/'tools/detector_preflight/rtl_equivalence.py')
rtl=importlib.util.module_from_spec(spec);spec.loader.exec_module(rtl)


def test_generated_ids_can_change_but_structure_is_preserved():
    assert rtl.compare_text('wire a_reg_12; assign a_reg_12 = 7;',
                            'wire a_reg_19; assign a_reg_19 = 7;')==1


@pytest.mark.parametrize('changed',[
    'wire a_reg_19; assign a_reg_19 = 8;',
    'wire a_reg_19; assign a_reg_20 = 7;',
    'wire a_reg_19; assign ap_start = 7;',
])
def test_semantic_changes_or_inconsistent_renaming_are_rejected(changed):
    with pytest.raises(ValueError):rtl.compare_text('wire a_reg_12; assign a_reg_12 = 7;',changed)
