"""Summarize retained campaign evidence without launching any work."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'results/modnas_vs_ours_20261007'


def record(relative):
    path = OUTPUT / relative
    if not path.is_file():
        return None
    return dict(path=path.relative_to(ROOT).as_posix(),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                value=json.loads(path.read_text(encoding='utf-8')))


def summarize():
    measurements, evidence = [], {}
    for folder in ('mapping_probe', 'mapping_inline_probe', 'mapping_inline_scale_probe'):
        bound = record(folder + '/verified_results.json')
        if bound is not None:
            evidence[folder] = dict(path=bound['path'], sha256=bound['sha256'])
            for row in bound['value']['results']:
                metrics = row.get('metrics') or {}
                measurements.append(dict(case_name=row['case_name'], PE=row['PE'], SIMD=row['SIMD'],
                    status=row['status'], numerical_verification=row['numerical_report_verification'],
                    latency_us=metrics.get('latency_us'), LUT=metrics.get('LUT'), DSP=metrics.get('DSP'),
                    BRAM18=metrics.get('BRAM18'), timing_eligible=metrics.get('timing_eligible'),
                    resource_eligible=metrics.get('resource_screen_eligible'),
                    serial_proxy_stem_budget_met=row.get('stem_meets_120ms_serial_proxy', False),
                    source_hash=row['source_hash'], board_measured=False, full_network_measured=False))
    eligible = [r for r in measurements if r['numerical_verification'] == 'PASS_RAW_SOURCE_INTEGER_AND_XML'
                and r['timing_eligible'] and r['resource_eligible']]
    best = min(eligible, key=lambda r: r['latency_us']) if eligible else None
    ptq = record('ptq_real_dataset_a/attempt_v2/real_data_ptq_gate.json')
    ptq_status = None
    if ptq:
        evidence['W8A8_real_train_gate'] = dict(path=ptq['path'], sha256=ptq['sha256'])
        ptq_status = dict(status=ptq['value']['status'],
            scope=ptq['value']['scope'], adapter_version=ptq['value']['adapter_version'],
            checkpoint_status='ONE_STEP_DIAGNOSTIC',
            original_E0_FP32_BN_fold_gate='FAILED_PRESERVED',
            replacement_scope='SEPARATELY_REGISTERED_FP64_MATHEMATICAL_REFERENCE',
            accuracy_retention='NOT_EVALUATED')
    mixed = record('ptq_real_dataset_a/mixed_head_v1/real_data_mixed_ptq_gate.json')
    mixed_status = None
    if mixed:
        evidence['all_profile_real_train_gate'] = dict(path=mixed['path'], sha256=mixed['sha256'])
        mixed_status = dict(status=mixed['value']['status'], scope=mixed['value']['scope'],
            profiles=mixed['value']['supported_software_profiles'],
            candidate_witnesses=[dict(macro=macro, profile=profile,
                compared_nodes=row['compared_nodes'], compared_elements=row['compared_elements'],
                mismatch_count=row['actual_mismatch_count'])
                for macro, candidate in mixed['value']['candidates'].items()
                for profile, row in candidate['profiles'].items()],
            calibration_images=64, integer_witness_images_per_candidate=1,
            accuracy_retention='NOT_EVALUATED', hardware_export='NOT_IMPLEMENTED')
    joint = record('joint_v2_newspace_preflight/reviewed_v2/plan.json')
    joint_status = None
    if joint:
        evidence['joint_v2_newspace_preflight'] = dict(path=joint['path'], sha256=joint['sha256'])
        joint_status = dict(status=joint['value']['status'],
            software_backend_status=joint['value']['software_backend']['status'],
            qualified_hardware_triples=joint['value']['qualified_hardware_triples'],
            eligible_hardware_triples=joint['value']['eligible_triples'],
            formal_launch_blockers=joint['value']['blockers'],
            training_interfaces='IMPLEMENTED_BUT_NOT_EXECUTED')
    result = dict(schema_version=1, updated_at_utc=datetime.now(timezone.utc).isoformat(),
        status='HARDWARE_AND_NEW_BACKEND_PREPARATION',
        user_choice=dict(comparison='FORMAL_ZCU104_HARDWARE_CONSTRAINTS',
                         ours='JOINT_V2_ARCHITECTURE_QUANTIZATION_MAPPING'),
        network='NEW_E0_E1_FIXED_COMPLETE_B1_48CH_FOUR_SCALE',
        canonical_architectures=72, joint_space_required_triples=1296,
        hls_measurements=measurements, best_measured_stem=best,
        W8A8_software_ptq=ptq_status, all_profile_software_ptq=mixed_status,
        joint_v2_newspace=joint_status,
        network_search_started=False, final_retraining_started=False,
        comparison_accuracy_results=None, test_evaluated=False,
        formal_comparison_ready=False, full_network_hardware_feasibility='UNKNOWN',
        evidence_layers=dict(software_integer=True, operator_csim=bool(measurements),
                             operator_csynth=bool(measurements), full_route=False,
                             board_runtime=False, measured_power=False),
        unresolved=['Complete new-space source-bound costs and qualified common B1 mapping',
                    'HEAD16 boundary/kernel hardware and full-network interface contracts',
                    'MODNAS joint precision loss/runtime and full tuple cost-table validator',
                    'Matched per-search total GPU budget and final retraining/model-selection freeze',
                    'Remaining datasets development split/model-head-specific cost freeze'],
        protocol='tools/modnas_comparison/PROTOCOL_DRAFT.md', evidence=evidence)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / 'campaign_status.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)
    return result


if __name__ == '__main__':
    result = summarize()
    print(json.dumps(dict(status=result['status'], completed_hls_cases=len(result['hls_measurements']),
                         best_stem=result['best_measured_stem'], network_search_started=False), ensure_ascii=False))
