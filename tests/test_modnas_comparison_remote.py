"""Check provenance and report gates before a finite remote HLS invocation."""
import json
from pathlib import Path
import shutil

import pytest

from tools.modnas_comparison.remote_worker import parse_report, verify_case

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / 'results/modnas_vs_ours_20261007/mapping_probe'


def test_all_prepared_cases_are_exact_and_registered():
    cases = sorted((PROBE / 'cases').iterdir())
    assert {verify_case(case)['PE'] for case in cases} == {8, 16, 32}
    assert len(cases) == 3


def test_changed_witness_is_rejected(tmp_path):
    source = next((PROBE / 'cases').iterdir())
    case = tmp_path / source.name
    shutil.copytree(source, case)
    (case / 'reference.bin').write_bytes(b'changed')
    with pytest.raises(ValueError, match='witness drift'):
        verify_case(case)


def test_existing_verified_report_keeps_units_and_evidence_scope():
    manifest = json.loads((PROBE / 'preparation_manifest.json').read_text(encoding='utf-8'))
    report = ROOT / manifest['baseline_exact_measurement']['source_report_path']
    result = parse_report(report)
    assert result['latency_us'] == manifest['baseline_exact_measurement']['latency_us']
    assert result['DSP'] == manifest['baseline_exact_measurement']['DSP']
    assert result['timing_eligible'] is True


def test_wrong_platform_report_is_rejected(tmp_path):
    manifest = json.loads((PROBE / 'preparation_manifest.json').read_text(encoding='utf-8'))
    original = ROOT / manifest['baseline_exact_measurement']['source_report_path']
    report = tmp_path / 'wrong.xml'
    report.write_bytes(original.read_bytes().replace(b'xczu7ev-ffvc1156-2-e', b'wrongpart'))
    with pytest.raises(ValueError, match='platform'):
        parse_report(report)
