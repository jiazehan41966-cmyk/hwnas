"""Joint HW-NAS candidate loop.

The 144 x 3 x 6 enumeration is FULL_SPACE_HARDWARE_COVERAGE. Final Pareto uses
only the frozen B1/C/D Q(x) domain. Missing measurements stay missing.
"""
import json
from pathlib import Path
from candidate_graphs import architectures
from common import ROOT, csvwrite, save
from cost_api import get_database
from pareto import PRIMARY_POLICIES, pareto_accuracy_peak_resources, pareto_max_min
from q_admission import FINAL_NETWORKS, PHASE5E_MANIFEST, annotate_final_eligibility, load_frozen_q
from quant_contracts import PROFILES

SEARCH_PROFILES=tuple(PROFILES)
OUT=ROOT/'results/fpga_h4_joint_search'


def _manifest(row, kind):
    if kind=='accuracy':
        reason=row.get('reason') or row.get('accuracy_status')
    else:
        reason=row.get('reason') or row.get('hardware_status')
    return dict(
        architecture_id=row.get('architecture_id'),
        architecture_hash=row.get('architecture_hash'),
        graph_hash=row.get('graph_hash'),
        quantization_config=row.get('quantization_config') or row.get('quant_config'),
        quant_contract_hash=row.get('quant_contract_hash'),
        mapping_config=row.get('mapping_config') or row.get('mapping_policy'),
        mapping_hash=row.get('mapping_hash'),
        target_device=row.get('target_device'),
        clock_constraint_ns=row.get('clock_constraint_ns'),
        toolchain_version=row.get('toolchain_version'),
        hardware_status=row.get('hardware_status'),
        accuracy_status=row.get('accuracy_status'),
        reason=reason,
        kind=kind)


def _public(row):
    keys=('final_network','network','architecture_id','profile','quant_config','quant_contract_hash',
        'mapping_policy','mapping_hash','q_status','q_reason','hardware_status','accuracy_status',
        'measurement_pair','eligible','Accuracy','serial_op_estimate_us','Latency','LUT','FF','DSP','BRAM',
        'measurement_scope','shell_reservation','deployment','measurement_ids')
    return {k:row.get(k) for k in keys}


def _canonical_full_grid(aids,profiles,policies):
    return (set(aids)==set(architectures()) and tuple(profiles)==SEARCH_PROFILES and tuple(policies)==tuple(PRIMARY_POLICIES))


CANDIDATE_FIELDS=['final_network','architecture_id','profile','quant_config','mapping_policy','q_status','q_reason',
    'hardware_status','accuracy_status','measurement_pair','eligible','Accuracy','serial_op_estimate_us',
    'LUT','FF','DSP','BRAM','measurement_scope','shell_reservation','deployment']


def run_joint_search(db=None, architecture_ids=None, profiles=None, policies=None):
    """Call the hardware agent once per real search candidate, then gate Pareto."""
    db=db or get_database()
    aids=list(architecture_ids) if architecture_ids is not None else list(architectures())
    profiles=tuple(profiles or SEARCH_PROFILES)
    policies=tuple(policies or PRIMARY_POLICIES)
    before=getattr(db,'agent_queries',0)
    rows=[]
    queried=[]
    for aid in aids:
        queried.append(aid)
        for profile in profiles:
            for policy in policies:
                rows.append(db.estimate_cost(aid,profile,policy))
    calls=db.agent_queries-before
    if calls!=len(rows) or queried!=aids:
        raise RuntimeError('Hardware Cost Agent was not called for every search candidate')
    for row in rows:
        annotate_final_eligibility(row)
    def count(pred):
        return sum(1 for row in rows if pred(row))
    final=[row for row in rows if row.get('eligible') is True]
    if _canonical_full_grid(aids,profiles,policies) and (len(final)!=30 or any(row.get('q_status')!='Q_ADMITTED' or row.get('hardware_status')!='MEASURED' or row.get('accuracy_status')!='MEASURED' for row in final)):
        raise RuntimeError('Frozen final co-design domain is not 30/30 fully measured')
    latency_front=pareto_max_min(final,'Accuracy','serial_op_estimate_us')
    fronts=dict(
        accuracy_vs_serial_op_estimate_us=latency_front,
        accuracy_vs_lut=pareto_max_min(final,'Accuracy','LUT'),
        accuracy_vs_ff=pareto_max_min(final,'Accuracy','FF'),
        accuracy_vs_dsp=pareto_max_min(final,'Accuracy','DSP'),
        accuracy_vs_bram=pareto_max_min(final,'Accuracy','BRAM'),
        accuracy_vs_peak_operator_lut_ff_dsp_bram=pareto_accuracy_peak_resources(final))
    frozen=load_frozen_q()
    summary=dict(
        enumeration_scope='FULL_SPACE_HARDWARE_COVERAGE',
        pareto_domain='FROZEN_FINAL_CODESIGN_DOMAIN',
        final_networks=list(FINAL_NETWORKS),
        frozen_q={label:list(profiles_for) for label,profiles_for in frozen.items()},
        total_candidates=len(rows),
        architectures=len(aids),
        architectures_queried=len(set(queried)),
        profiles=list(profiles),
        mapping_policies=list(policies),
        hardware_agent_calls=calls,
        hardware_measured=count(lambda row: row.get('hardware_status')=='MEASURED'),
        needs_measurement=count(lambda row: row.get('hardware_status')=='NEEDS_MEASUREMENT'),
        hardware_contract_mismatch=count(lambda row: row.get('hardware_status')=='CONTRACT_MISMATCH'),
        invalid=count(lambda row: row.get('hardware_status')=='INVALID'),
        accuracy_measured=count(lambda row: row.get('accuracy_status')=='MEASURED'),
        needs_accuracy=count(lambda row: row.get('accuracy_status')=='NEEDS_ACCURACY'),
        accuracy_contract_mismatch=count(lambda row: row.get('accuracy_status')=='CONTRACT_MISMATCH'),
        contract_mismatch=count(lambda row: row.get('hardware_status')=='CONTRACT_MISMATCH' or row.get('accuracy_status')=='CONTRACT_MISMATCH'),
        q_admitted=count(lambda row: row.get('q_status')=='Q_ADMITTED'),
        q_not_admitted=count(lambda row: row.get('q_status')=='Q_NOT_ADMITTED'),
        measurement_pair_candidates=count(lambda row: row.get('measurement_pair') is True),
        final_joint_candidates=len(final),
        eligible_candidates=len(final),
        pareto_candidates=len(latency_front),
        pareto_front_sizes={name:len(front) for name,front in fronts.items()},
        by_profile={profile:dict(
            candidates=count(lambda row, profile=profile: row.get('profile')==profile),
            hardware_measured=count(lambda row, profile=profile: row.get('profile')==profile and row.get('hardware_status')=='MEASURED'),
            needs_measurement=count(lambda row, profile=profile: row.get('profile')==profile and row.get('hardware_status')=='NEEDS_MEASUREMENT'),
            accuracy_measured=count(lambda row, profile=profile: row.get('profile')==profile and row.get('accuracy_status')=='MEASURED'),
            needs_accuracy=count(lambda row, profile=profile: row.get('profile')==profile and row.get('accuracy_status')=='NEEDS_ACCURACY'),
            q_admitted=count(lambda row, profile=profile: row.get('profile')==profile and row.get('q_status')=='Q_ADMITTED'),
            eligible=count(lambda row, profile=profile: row.get('profile')==profile and row.get('eligible') is True),
        ) for profile in profiles},
        measurement_scope='OP_LEVEL_HLS',
        latency_semantics='serial operator CSYNTH estimate',
        resource_semantics='operator-level peak proxy',
        board_measured_latency=False,
        routed_full_detector_resource=False,
        shell_reservation='UNKNOWN')
    return dict(summary=summary,rows=rows,eligible=final,final_candidates=final,pareto=latency_front,fronts=fronts,
        missing_measurements=[_manifest(row,'hardware') for row in rows if row.get('hardware_status')!='MEASURED'],
        missing_accuracy=[_manifest(row,'accuracy') for row in rows if row.get('accuracy_status')!='MEASURED'],
        queried_architecture_ids=queried)


def _audit(report):
    summary=report['summary']
    frozen=summary['frozen_q']
    manifest=json.loads(PHASE5E_MANIFEST.read_text(encoding='utf-8'))
    admission=manifest.get('admission') or {}
    excluded=[]
    seen=set()
    for row in report['rows']:
        if row.get('measurement_pair') is True and row.get('q_status')=='Q_NOT_ADMITTED':
            key=(row.get('final_network'),row.get('profile'))
            if key in seen:
                continue
            seen.add(key)
            excluded.append(dict(network=key[0],profile=key[1],q_reason=row.get('q_reason'),
                hardware_status=row.get('hardware_status'),accuracy_status=row.get('accuracy_status')))
    lines=[
        '# Q(x) eligibility audit',
        '',
        'Frozen source: `reports/phase5e_mixed_precision/Q_OF_X.json`.',
        'This gate is not recomputed from hardware cost, quantized accuracy, or Pareto membership.',
        '',
        '## Frozen Q(x)',
        '',
        '| Network | Admitted quant contracts |',
        '|---|---|',
    ]
    for label in FINAL_NETWORKS:
        lines.append('| '+label+' | '+', '.join(frozen[label])+' |')
    lines.extend([
        '',
        'Phase5E recorded admission used only as context for profiles already absent from Q(x):',
        '',
        '- C_Q1: `'+str(admission.get('C_Q1'))+'`',
        '- C_Q2 (head W16A16): `'+str(admission.get('C_Q2'))+'`',
        '',
        '## Gate',
        '',
        'A candidate is final-eligible only when all of the following hold:',
        '',
        '1. architecture is in X* = {B1, C, D}',
        '2. quant contract is in frozen Q(architecture)',
        '3. hardware_status == MEASURED',
        '4. accuracy_status == MEASURED',
        '',
        'Otherwise the Q gate status is `Q_NOT_ADMITTED`. Measured hardware and measured accuracy do not override it.',
        '',
        '## Counts',
        '',
        '| Scope | Result |',
        '|---|---|',
        '| Hardware coverage (`FULL_SPACE_HARDWARE_COVERAGE`) | '+str(summary['hardware_measured'])+'/'+str(summary['total_candidates'])+' MEASURED |',
        '| Accuracy coverage over the full enumeration | '+str(summary['accuracy_measured'])+'/'+str(summary['total_candidates'])+' MEASURED |',
        '| Q_NOT_ADMITTED | '+str(summary['q_not_admitted'])+' |',
        '| Frozen final co-design domain | '+str(summary['final_joint_candidates'])+'/30 fully measured |',
        '| Final Pareto (accuracy vs serial_op_estimate_us) | '+str(summary['pareto_candidates'])+' candidates, computed only on those '+str(summary['final_joint_candidates'])+' |',
        '',
        'The 144-architecture enumeration remains hardware coverage. It is not a quantified joint Pareto: 141 architectures have no quantized accuracy.',
        '',
        '## Measured profiles excluded by Q(x)',
        '',
        '| Network | Profile | Reason |',
        '|---|---|---|',
    ])
    if not excluded:
        lines.append('| — | — | none |')
    for item in excluded:
        lines.append('| '+str(item['network'])+' | '+str(item['profile'])+' | '+str(item['q_reason'])+' |')
    lines.extend([
        '',
        'B1 / HEAD_W16A16 is outside frozen Q(B1).',
        'C / HEAD_W16A16 is outside frozen Q(C). Phase5E records C_Q2 as `'+str(admission.get('C_Q2'))+'`.',
        'Those rows can remain in the historical accuracy database. They are absent from `FINAL_JOINT_CANDIDATES` and from every final Pareto front.',
        '',
        '## Metric semantics',
        '',
        '- Latency: `OP_LEVEL_HLS`, serial operator CSYNTH estimate (`serial_op_estimate_us`).',
        '- Resource: operator-level peak proxy for LUT, FF, DSP, and BRAM18.',
        '- Board latency: not measured.',
        '- Routed full-detector resource: not measured.',
        '- Shell reservation: UNKNOWN.',
        '',
        '## Stop',
        '',
        'No additional accuracy evaluation, HLS run, place-and-route, or board deployment is started from this audit.',
        '',
    ])
    return '\n'.join(lines)


def write_report(report, out=OUT):
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    final=sorted(report['final_candidates'],key=lambda row:(row.get('final_network') or '',row.get('profile') or '',row.get('mapping_policy') or ''))
    csvwrite(out/'FINAL_JOINT_CANDIDATES.csv',[{key:row.get(key) for key in CANDIDATE_FIELDS} for row in final],CANDIDATE_FIELDS)
    pareto_rows=report['pareto']
    csvwrite(out/'FINAL_JOINT_PARETO.csv',[{key:row.get(key) for key in CANDIDATE_FIELDS} for row in pareto_rows],CANDIDATE_FIELDS)
    payload=dict(report['summary'],fronts={name:[_public(row) for row in front] for name,front in report['fronts'].items()})
    save(out/'FINAL_JOINT_PARETO.json',payload)
    (out/'QX_ELIGIBILITY_AUDIT.md').write_text(_audit(report),encoding='utf-8')
    public=dict(report['summary'],pareto=[_public(row) for row in pareto_rows])
    save(out/'summary.json',public)
    save(out/'missing_measurements.json',report['missing_measurements'])
    save(out/'missing_accuracy.json',report['missing_accuracy'])
    return public


def main():
    report=run_joint_search()
    public=write_report(report)
    shown=('enumeration_scope','pareto_domain','hardware_measured','total_candidates','accuracy_measured',
        'q_admitted','q_not_admitted','final_joint_candidates','pareto_candidates','pareto_front_sizes','shell_reservation')
    print(json.dumps({key:public[key] for key in shown},indent=2))


if __name__=='__main__':
    main()
