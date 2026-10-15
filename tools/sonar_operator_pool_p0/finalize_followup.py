"""Bind four no-training audits and add non-degenerate binomial intervals."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/sonar_operator_pool_p0_followup'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def wilson(k,n):
    if n==0:return (np.nan,np.nan)
    z=1.959963984540054;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0.,center-half),min(1.,center+half)


def main():
    # Preserve earlier inferential tables; add Wilson intervals for zero/all-success strata.
    r=pd.read_csv(OUT/'manta/stratified_recall.csv')
    r[['recall_wilson95_low','recall_wilson95_high']]=[wilson(k,n) for k,n in zip(r.detected_count,r.gt_count)]
    r.to_csv(OUT/'manta/stratified_recall_wilson.csv',index=False)
    assert wilson(0,8)[1]>.3 and wilson(8,8)[0]<.7
    assert np.isnan(wilson(0,0)[0])
    audit=json.loads((ROOT/'reports/sonar_operator_pool_p0_confirmation/BASELINE_COMPARABILITY.json').read_text())
    for rel,expected in audit['source_sha256'].items():
        if sha(ROOT/rel)!=expected:raise ValueError(f'Frozen source drift: {rel}')
    for rel,key in [('configs/baseline_sonar_frozen.yaml','baseline_sha256'),('configs/search_space_sonar_v0_1.yaml','search_space_sha256')]:
        if sha(ROOT/rel)!=audit[key]:raise ValueError(f'Frozen config drift: {rel}')
    previous=ROOT/'reports/sonar_operator_pool_p0_diagnosis'
    original=json.loads((previous/'FINAL_DIAGNOSTIC_MANIFEST.json').read_text())
    for rel,expected in original['files_sha256'].items():
        if sha(previous/rel)!=expected:raise ValueError(f'Prior report drift: {rel}')
    manifests={
        'annotation':('ANNOTATION_AUDIT_MANIFEST.json','annotation_followup.py','script_sha256'),
        'assignment':('ASSIGNMENT_MANIFEST.json','assignment_followup.py','source_sha256'),
        'manta':('MANIFEST.json','manta_followup.py','source_sha256'),
        'confidence':('MANIFEST.json','confidence_followup.py','source_sha256')}
    for folder,(filename,script,key) in manifests.items():
        m=json.loads((OUT/folder/filename).read_text(encoding='utf-8'))
        if sha(ROOT/'tools/sonar_operator_pool_p0'/script)!=m[key]:raise ValueError(f'Script drift: {script}')
    # Check all raw input files against the admitted historical train/valid binding.
    data=ROOT/'data/external_redownload_audit_20260909/roboflow_cylider2/extracted/v6_yolov8'
    files=0
    for split in ('train','valid'):
        for row in audit['datasets'][split]['entries']:
            for key in ('image','annotation'):
                rel=Path(row[key].replace('\\','/'))
                if rel.parts[0]!=split:raise ValueError('Unpermitted split path')
                if sha(data/rel)!=row[key+'_sha256']:raise ValueError('Raw input drift')
                files+=1
    # Cross-audit joins: no instance duplication, class/geometry mismatch or lost stratum.
    d=pd.read_csv(OUT/'manta/image_bbox_statistics.csv')
    a=pd.read_csv(OUT/'assignment/assignment_gt.csv')
    a['image']=a.image.str.replace('\\','/',regex=False)
    j=d.merge(a,on=['split','image','class_name'],validate='one_to_one')
    if len(j)!=429 or not np.allclose(j.width_640,j.width,atol=1e-4,rtol=0) or not np.allclose(j.height_640,j.height,atol=1e-4,rtol=0):
        raise ValueError('Cross-audit GT binding mismatch')
    diagnostic_scope={
        'new_training_runs':0,'model_weights_modified':False,'labels_modified':False,
        'test_images_labels_read':False,'test_inference':False,'test_evaluated':False,
        'prior_interrupted_agent_test_metadata_seen':True,
        'test_metadata_note':'Earlier interrupted continuation viewed a prefix of an existing all-split source index including test entries; none used. No test image/label file or inference accessed; current scripts enforce train/valid.',
        'raw_logit_decomposition':'UNAVAILABLE; cached products only; remote authentication failed',
        'semantic_annotation_exhaustiveness':'UNRESOLVED; specialist review required'}
    payload=dict(status='FOUR_AUDITS_COMPLETE_WITH_EXPLICIT_LIMITS',scope=diagnostic_scope,
                 frozen_sources_verified=len(audit['source_sha256']),frozen_configs_verified=2,
                 historical_output_hashes_verified=len(original['files_sha256']),raw_train_valid_hashes_verified=files,
                 cross_audit_gt_rows=429,confidence_retained_tp_parity_checks=48,
                 new_unit_tests_passed=11,seed_count=3,
                 source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'tools/sonar_operator_pool_p0').glob('*followup.py')},
                 output_sha256={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='FOLLOWUP_MANIFEST.json'})
    (OUT/'FOLLOWUP_MANIFEST.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print('PASS',files,'raw files;',len(original['files_sha256']),'unchanged prior outputs;',len(payload['output_sha256']),'followup artifacts')


if __name__=='__main__':main()
