"""Fail-closed H4 tests: exact lookup only, unseen mappings abstain, no deployment claim."""
import csv,json
from common import *
from search import admit_rows,load_h3,needs_measurement

def checks(include_manifest=True):
    tests=[]
    man,hashes,rows,info=load_h3()
    assert man['status']=='H3_DB_READY';tests.append('h3_ready')
    options=admit_rows(rows,hashes)
    sid=info['usage']['D'][0]['shape_id']
    assert needs_measurement(sid,options,pe=99,simd=99)['status']=='NEEDS_MEASUREMENT'
    tests.append('unseen_mapping_needs_measurement')
    src=(ROOT/'tools/fpga_h4/search.py').read_text(encoding='utf-8')
    assert 'def predict' not in src and "method=='nearest'" not in src
    tests.append('no_surrogate_in_h4_source')
    table=list(csv.DictReader((H4_OUT/'RESOURCE_LATENCY_TABLE.csv').open(encoding='utf-8')))
    assert table and all(r['shell_reservation']=='UNKNOWN' for r in table)
    assert all(r['latency_scope'].startswith('HLS_CSYNTH') for r in table)
    tests.append('shell_unknown_and_csynth_scope')
    ids={r['point_id'] for r in rows}
    for r in table:
        for pid in r['measurement_ids'].split('|'):
            assert pid in ids,pid
    tests.append('measurement_ids_bound_to_h3_clean')
    proto=(H4_OUT/'H4_PROTOCOL.md').read_text(encoding='utf-8')
    assert 'UNKNOWN' in proto and 'board' in proto.lower()
    tests.append('protocol_states_limitations')
    if include_manifest:
        man4=json.loads((H4_OUT/'H4_DATABASE_MANIFEST.json').read_text())
        assert man4['H4_deployment_ready']=='NO' and man4['stop']=='wait_for_human_review'
        tests.append('stop_condition_no_deployment')
    return tests,len(table)

def main():
    tests,n=checks(include_manifest=True)
    save(H4_OUT/'H4_LOCAL_VALIDATION.json',dict(status='PASS',tests=tests,named_policies=n))
    print('PASS',len(tests),'checks')

if __name__=='__main__':main()
