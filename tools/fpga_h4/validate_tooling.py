"""Fail-closed H4 tooling tests: exact lookup only, unknown triples abstain."""
import copy,json
from common import *
from cost_api import atomic_usage, estimate_cost, get_database
from search import admit_rows, load_accuracy, load_h3, validate_h3_row

FORBIDDEN=('method==\'nearest\'',"method=='interpolation'","method=='linear'",'def predict','from cost_model import','nearest_neighbor')


def _src(*names):
    return '\n'.join((ROOT/'tools/fpga_h4'/name).read_text(encoding='utf-8') for name in names)

def _full_spec(row):
    return {key:row[key] for key in EXACT_MAPPING_FIELDS}


def checks():
    tests=[]
    db=get_database()
    man,_,_,info=load_h3()
    assert man['status']=='H3_DB_READY'
    tests.append('h3_ready')

    src=_src('cost_api.py','pareto.py','tooling.py')
    assert all(token not in src for token in FORBIDDEN)
    tests.append('no_surrogate_in_tooling_source')

    for label in PARETO_NETWORKS+['baseline']:
        derived,_=atomic_usage(db.graphs[label])
        frozen=info['usage'][label]
        assert [u['shape_id'] for u in derived]==[u['shape_id'] for u in frozen],label
        assert [u['node'] for u in derived]==[u['node'] for u in frozen],label
    tests.append('graph_usage_matches_h3_atomic')

    ok=estimate_cost('B1','NUMERIC_CONTRACT_V1','reference_min_latency')
    assert ok['status']=='LOOKED_UP' and ok['hardware_status']=='MEASURED'
    assert ok['shell_reservation']=='UNKNOWN' and ok['deployment']=='NOT_CLAIMED'
    assert ok['latency_scope'].startswith('HLS_CSYNTH')
    assert ok['measurement_scope']=='OP_LEVEL_HLS'
    assert ok['accuracy_status']=='MEASURED' and ok['quantized_accuracy_bound'] is True
    assert ok['accuracy_precision']=='W8A8'
    assert ok['Accuracy'] is not None and ok['Accuracy']!=load_accuracy()['B1']['mean_mAP50_95']
    assert ok['Latency']==ok['compute_latency_us']==ok['latency']['serial_op_estimate_us']
    assert ok['latency']['board_measured_ms'] is None
    assert ok['resource']['integrated_post_synth'] is None and ok['resource']['board_routed'] is None
    assert ok['resource']['peak_operator']['dsp']==ok['DSP']==ok['DSP_peak']
    assert ok['resource']['peak_operator']['lut']==ok['LUT']==ok['LUT_peak']
    assert ok['resource']['peak_operator']['bram']==ok['BRAM']==ok['BRAM18_peak']
    assert 'board_latency' not in ok and 'end_to_end_latency' not in ok and 'total_network_resource' not in ok
    tests.append('measured_triple_looked_up')

    constrained=estimate_cost('B1','NUMERIC_CONTRACT_V1',dict(policy='reference_min_latency',target_clock_ns=5))
    assert constrained['status']=='LOOKED_UP'
    bad_constraint=estimate_cost('B1','NUMERIC_CONTRACT_V1',dict(policy='reference_min_latency',target_clock_ns=10))
    assert bad_constraint['status']=='NEEDS_MEASUREMENT' and bad_constraint['Latency'] is None
    tests.append('named_policy_checks_explicit_constraints')

    graph_ok=estimate_cost(db.graphs['C'],'NUMERIC_CONTRACT_V1','reference_min_latency')
    named_ok=estimate_cost('C','NUMERIC_CONTRACT_V1','reference_min_latency')
    assert graph_ok['status']==named_ok['status']=='LOOKED_UP'
    assert graph_ok['compute_latency_us']==named_ok['compute_latency_us']
    tests.append('graph_dict_matches_label')

    broken=copy.deepcopy(db.graphs['B1'])
    broken['nodes']=broken['nodes'][:-1]
    bad_graph=estimate_cost(broken,'NUMERIC_CONTRACT_V1','reference_min_latency')
    assert bad_graph['status']=='NEEDS_MEASUREMENT' and bad_graph['Accuracy'] is None and bad_graph['Latency'] is None
    tests.append('modified_graph_does_not_inherit_frozen_accuracy')

    usage=db.info['usage']['B1']
    full_by_node={}
    expected=0.0
    chosen={}
    for u in usage:
        pool=sorted(db.options[u['shape_id']],key=lambda r:(r['PE'],r['SIMD'],r['point_id']))
        chosen[u['node']]=pool[0]
    n_low='neck.output_convs.0.0'
    n_high='head.towers.0.0'
    assert chosen[n_low]['shape_id']==chosen[n_high]['shape_id']
    same_pool=sorted(db.options[chosen[n_low]['shape_id']],key=lambda r:(r['PE'],r['SIMD'],r['point_id']))
    assert len(same_pool)>=2
    chosen[n_low]=same_pool[0]
    chosen[n_high]=same_pool[-1]
    for u in usage:
        full_by_node[u['node']]=_full_spec(chosen[u['node']])
        expected+=chosen[u['node']]['latency_us']
    mixed=estimate_cost('B1','NUMERIC_CONTRACT_V1',dict(name='adversarial_same_shape_by_node',by_node=full_by_node))
    assert mixed['status']=='LOOKED_UP'
    assert abs(mixed['compute_latency_us']-expected)<1e-6
    assert abs(mixed['compute_latency_us']-13065647.975)>1000
    assert abs(mixed['compute_latency_us']-15832290.455)<1e-3
    tests.append('by_node_same_shape_mappings_do_not_overwrite')

    unseen=estimate_cost('B1','NUMERIC_CONTRACT_V1',dict(PE=99,SIMD=99,buffer_profile='linebuffer_fifo4'))
    assert unseen['status']=='NEEDS_MEASUREMENT' and unseen['hardware_status']=='NEEDS_MEASUREMENT'
    assert unseen['eligible'] is False
    assert unseen['Latency'] is None and unseen['DSP'] is None and unseen['LUT'] is None and unseen['BRAM'] is None
    assert unseen['latency']['serial_op_estimate_us'] is None and unseen['latency']['board_measured_ms'] is None
    if unseen.get('accuracy_status')=='MEASURED':
        assert unseen['Accuracy']!=load_accuracy()['B1']['mean_mAP50_95']
    tests.append('unseen_mapping_needs_measurement')

    quant=estimate_cost('B1',dict(activation_bits=4,weight_bits=4,accumulator_bits=32,output_bits=4),'reference_min_latency')
    assert quant['status']=='NEEDS_MEASUREMENT' and quant['Latency'] is None
    tests.append('unmeasured_quant_needs_measurement')

    unknown_net=estimate_cost('not_a_frozen_candidate','NUMERIC_CONTRACT_V1','reference_min_latency')
    assert unknown_net['status']=='NEEDS_MEASUREMENT' and unknown_net['Latency'] is None
    tests.append('unknown_network_needs_measurement')

    mutated=copy.deepcopy(db.graphs['D'])
    mutated['nodes'].append(dict(id='fake.int4.op',op='int4_conv',Cin=3,Cout=3,H=8,W=8,kernel=1,stride=1,groups=1,activation='linear'))
    fake=estimate_cost(mutated,'NUMERIC_CONTRACT_V1','reference_min_latency')
    assert fake['status']=='NEEDS_MEASUREMENT' and fake['Latency'] is None
    tests.append('unknown_operator_needs_measurement')

    unspecified=estimate_cost('D','NUMERIC_CONTRACT_V1',None)
    assert unspecified['status']=='NEEDS_MEASUREMENT'
    tests.append('unspecified_mapping_needs_measurement')

    v1=json.loads((ROOT/'reports/fpga_h2_contract/numeric_contract_v1.json').read_text(encoding='utf-8'))
    hashed=estimate_cost('D',v1,'reference_low')
    assert hashed['status']=='LOOKED_UP'
    tests.append('v1_json_contract_accepted')

    explicit_shape=db.info['usage']['D'][0]['shape_id']
    hit=db.options[explicit_shape][0]
    one=estimate_cost('D','NUMERIC_CONTRACT_V1',dict(by_shape={explicit_shape:dict(PE=hit['PE'],SIMD=hit['SIMD'],buffer_profile=hit['buffer_profile'])}))
    assert one['status']=='NEEDS_MEASUREMENT'
    tests.append('partial_explicit_mapping_does_not_fill')

    full={u['node']:_full_spec(db.options[u['shape_id']][0]) for u in db.info['usage']['D']}
    first=next(iter(full))
    bad_source=copy.deepcopy(full)
    bad_source[first]=dict(bad_source[first],source_hash='BAD')
    bad_lookup=estimate_cost('D','NUMERIC_CONTRACT_V1',dict(by_node=bad_source))
    assert bad_lookup['status']=='NEEDS_MEASUREMENT' and bad_lookup['Latency'] is None
    tests.append('explicit_lookup_rejects_bad_source_hash')

    base=next(r for r in db.rows if r['operator_type'] not in AUX_OPS)
    for field,value in [
        ('source_hash','0'*64),
        ('source_report_sha256','0'*64),
        ('input_bitwidth',4),
        ('tool_version','2024.1'),
        ('target_clock_ns',10),
        ('numeric_contract_hash','0'*64),
        ('interface_contract_hash','0'*64),
    ]:
        row=dict(base);row[field]=value
        checked,errors=validate_h3_row(row,db.hashes)
        assert errors and not checked['measurement_valid'],field
    tests.append('h3_admission_rejects_provenance_bitwidth_tool_clock_contract_mutations')

    timing=dict(base);timing['timing_status']='ESTIMATED_TARGET_MISSED'
    checked,errors=validate_h3_row(timing,db.hashes)
    assert 'report_timing_mismatch' in errors and not checked['measurement_valid'] and not checked['timing_eligible']
    admitted=admit_rows([timing],db.hashes)
    assert not admitted
    tests.append('measurement_valid_separate_from_timing_eligible')

    return tests


def main():
    tests=checks()
    save(H4_TOOLING_OUT/'H4_TOOLING_VALIDATION.json',dict(status='PASS',tests=tests,n_tests=len(tests),
        shell_reservation='UNKNOWN',deployment='NOT_CLAIMED',latency='HLS_CSYNTH_estimate_not_board'))
    print('PASS',len(tests),'checks')


if __name__=='__main__':
    H4_TOOLING_OUT.mkdir(parents=True,exist_ok=True)
    main()
