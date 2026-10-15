"""Quantized-accuracy lookup. No FP32 substitution and no cross-contract hit.

A record is MEASURED only when the stored quantization contract, checkpoint,
calibration, split, and seed set match the requested candidate. Phase2F FP32
scores are not an accuracy source for this lookup.
"""
import csv
import json
import math
import re
from common import ROOT, ARCH, contracts, hashobj, sha
from candidate_graphs import architectures, SPEC_SHA
from quant_contracts import canonical_profile, quant_contract

PHASE5A_MANIFEST=ROOT/'reports/phase5a_quantization/PTQ_MANIFEST.json'
PHASE5E_RESULTS=ROOT/'reports/phase5e_mixed_precision/MIXED_PRECISION_RESULTS.csv'
_CACHE={}
CALIBRATION=PHASE5A_MANIFEST.parent/'CALIBRATION_MANIFEST.json'
QUANT_CONFIG=PHASE5A_MANIFEST.parent/'quant_config.yaml'
CHECKPOINTS=PHASE5A_MANIFEST.parent/'CHECKPOINT_SELECTION_MANIFEST.json'


def _provenance():
    """Local retained manifests bind remote checkpoint identities; no remote I/O."""
    import yaml
    frozen=json.loads(CHECKPOINTS.read_text(encoding='utf-8'))
    config=yaml.safe_load(QUANT_CONFIG.read_text(encoding='utf-8'))
    if frozen['common_provenance']['search_space_sha256']!=SPEC_SHA:
        raise ValueError('checkpoint_search_space_mismatch')
    policy=config['official_policy']
    if any(policy.get(k)!=v for k,v in dict(activation_bits=8,weight_bits=8,accumulator_bits=32,
            scheme='symmetric',zero_point=0,rounding='nearest_even').items()):
        raise ValueError('uniform_quant_config_mismatch')
    checkpoints={(r['architecture_id'],int(r['seed'])):r['best_pt']['sha256']
        for r in frozen['runs'] if r.get('verification_status')=='VERIFIED_ORIGINAL'}
    return checkpoints,sha(CALIBRATION),sha(QUANT_CONFIG),policy['numeric_contract_hash']


def _sha_valid(value):
    return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None


def _blank(status, **extra):
    row=dict(status=status,accuracy_status=status,Accuracy=None,mean_mAP50_95=None,
        std_mAP50_95=None,n_seeds=None,test_evaluated=False,quantized_accuracy_bound=False,
        accuracy_precision=None,accuracy_split=None,accuracy_source=None,
        eval_contract_hash=None,quant_contract_hash=None,architecture_id=None,
        architecture_hash=None,checkpoint_sha256=None,calibration_sha256=None,seeds=None)
    row.update(extra)
    return row


def _sample_std(values):
    mean=sum(values)/len(values)
    if len(values)<2:
        return None
    return (sum((x-mean)**2 for x in values)/(len(values)-1))**0.5


def _eval_hash(payload):
    return hashobj(payload)


def _admit_group(architecture_id, profile, seeds, body_hash, qhash, source, calibration, evaluator):
    if len(seeds)!=3:
        return None
    if {s['seed'] for s in seeds}!={42,43,44}:
        return None
    if any(s.get('split')!='valid' or s.get('test_evaluated') is not False for s in seeds):
        return None
    checkpoints,expected_calibration,_,_= _provenance()
    if calibration!=expected_calibration or not _sha_valid(calibration):
        return None
    if any(not _sha_valid(s.get('checkpoint_sha256')) or s['checkpoint_sha256']!=checkpoints.get((architecture_id,s['seed'])) for s in seeds):
        return None
    if any(s.get('calibration_sha256')!=calibration for s in seeds):
        return None
    ordered=sorted(seeds,key=lambda s:int(s['seed']))
    values=[float(s['mAP50_95']) for s in ordered]
    if any(not math.isfinite(v) or not 0<=v<=1 for v in values):
        return None
    mean=sum(values)/len(values)
    provenance=dict(evaluator=evaluator,dataset_split='official_valid',metric='mAP50_95',
        seeds=[int(s['seed']) for s in ordered],
        checkpoint_sha256=[s['checkpoint_sha256'] for s in ordered],
        calibration_sha256=calibration,numeric_contract_hash=body_hash,
        quant_contract_hash=qhash,test_evaluated=False,preprocessing='phase_calibration_manifest')
    return dict(status='MEASURED',accuracy_status='MEASURED',Accuracy=mean,mean_mAP50_95=mean,
        std_mAP50_95=_sample_std(values),n_seeds=3,test_evaluated=False,
        quantized_accuracy_bound=True,accuracy_precision=profile,accuracy_split='valid',
        accuracy_source=source,architecture_id=architecture_id,quant_contract_hash=qhash,
        eval_contract_hash=_eval_hash(provenance),checkpoint_sha256=provenance['checkpoint_sha256'],
        calibration_sha256=calibration,seeds=provenance['seeds'],
        accuracy_key=dict(architecture_id=architecture_id,quant_contract_hash=qhash,
            eval_contract_hash=_eval_hash(provenance)))


def _load_phase5a(body_hash, contracts_by_profile):
    if not PHASE5A_MANIFEST.is_file():
        return {}
    manifest=json.loads(PHASE5A_MANIFEST.read_text(encoding='utf-8'))
    _,calibration,qconfig,numeric=_provenance()
    if (manifest.get('numeric_contract_hash')!=body_hash or numeric!=body_hash
            or manifest.get('quant_config_sha256')!=qconfig
            or manifest.get('calibration_manifest_sha256')!=calibration):
        return {('__contract__','W8A8'):_blank('CONTRACT_MISMATCH',reason='phase5a_numeric_contract_mismatch',
            accuracy_source='phase5a_ptq')}
    grouped={}
    for run in manifest.get('runs') or []:
        if run.get('status')!='MEASURED' or run.get('int8_mAP50_95') is None:
            continue
        grouped.setdefault(run['architecture_id'],[]).append(dict(
            seed=int(run['seed']),mAP50_95=run['int8_mAP50_95'],split=run.get('split'),
            test_evaluated=run.get('test_evaluated'),
            checkpoint_sha256=run.get('checkpoint_sha256'),
            calibration_sha256=manifest.get('calibration_manifest_sha256')))
    qhash=contracts_by_profile['W8A8']['quant_contract_hash']
    out={}
    for aid,seeds in grouped.items():
        admitted=_admit_group(aid,'W8A8',seeds,body_hash,qhash,'phase5a_ptq',
            manifest.get('calibration_manifest_sha256'),'phase5a_quantization')
        if admitted:
            out[(aid,'W8A8')]=admitted
        else:
            out[(aid,'W8A8')]=_blank('CONTRACT_MISMATCH',reason='phase5a_evaluation_provenance_mismatch')
    return out


def _load_phase5e(body_hash, contracts_by_profile):
    if not PHASE5E_RESULTS.is_file():
        return {}
    grouped={}
    mismatched=False
    invalid=set()
    with PHASE5E_RESULTS.open(encoding='utf-8',newline='') as handle:
        for row in csv.DictReader(handle):
            if row.get('status')!='MEASURED':
                continue
            if row.get('numeric_contract_sha256')!=body_hash:
                mismatched=True
                continue
            weight,activation=int(row['head_weight_bits']),int(row['head_activation_bits'])
            if (weight,activation)==(8,16):
                profile='HEAD_W8A16'
            elif (weight,activation)==(16,16):
                profile='HEAD_W16A16'
            else:
                # W16A8 and any other head width are not the W8A16 or W16A16 contracts.
                continue
            semantics=dict(signed=True,symmetric=True,zero_point=0,saturation=[-32768,32767],
                rounding='nearest_even',scale_derivation='deterministic_power_of_two_from_train_calibration_or_weight_maxabs')
            qconfig=dict(profile=row.get('profile'),network=row.get('network'),
                body=dict(weight_bits=8,activation_bits=8),head=dict(weight_bits=weight,activation_bits=activation),
                int16_semantics=semantics)
            if (ARCH.get(row.get('network'))!=row.get('architecture_id')
                    or row.get('quant_profile_config_sha256')!=hashobj(qconfig)):
                invalid.add((row['architecture_id'],profile))
            grouped.setdefault((row['architecture_id'],profile),[]).append(dict(
                seed=int(row['seed']),mAP50_95=row['mixed_mAP50_95'],split=row.get('split'),
                test_evaluated=False if str(row.get('test_evaluated')).lower()=='false' else None,
                checkpoint_sha256=row.get('checkpoint_sha256'),
                calibration_sha256=row.get('calibration_manifest_sha256')))
    out={}
    if mismatched and not grouped:
        return {('__contract__','mixed'):_blank('CONTRACT_MISMATCH',reason='phase5e_numeric_contract_mismatch',
            accuracy_source='phase5e_mixed_precision')}
    for (aid,profile),seeds in grouped.items():
        calibration=seeds[0]['calibration_sha256']
        qhash=contracts_by_profile[profile]['quant_contract_hash']
        admitted=_admit_group(aid,profile,seeds,body_hash,qhash,'phase5e_mixed_precision',
            calibration,'phase5e_mixed_precision')
        if admitted:
            out[(aid,profile)]=admitted
        else:
            invalid.add((aid,profile))
    for key in invalid:
        out[key]=_blank('CONTRACT_MISMATCH',reason='phase5e_evaluation_provenance_mismatch')
    return out


def accuracy_index(body_hashes=None):
    body_hashes=body_hashes or contracts()
    key=hashobj(body_hashes)
    paths=(PHASE5A_MANIFEST,PHASE5E_RESULTS,CALIBRATION,QUANT_CONFIG,CHECKPOINTS)
    key=(key,tuple((str(p),p.stat().st_mtime_ns,p.stat().st_size) if p.is_file() else (str(p),None,None) for p in paths))
    if key in _CACHE:
        return _CACHE[key]
    by_profile={name:quant_contract(name,body_hashes) for name in ('W8A8','HEAD_W8A16','HEAD_W16A16')}
    index={}
    numeric=body_hashes['numeric_contract_hash']
    try:
        index.update(_load_phase5a(numeric,by_profile))
        index.update(_load_phase5e(numeric,by_profile))
    except (OSError,ValueError,KeyError,TypeError):
        index={('__contract__',p):_blank('CONTRACT_MISMATCH',reason='accuracy_provenance_invalid') for p in by_profile}
    _CACHE[key]=(index,by_profile)
    return _CACHE[key]


def lookup_accuracy(architecture_id, profile, body_hashes=None, quant_contract_hash=None, architecture_hash=None):
    """Return MEASURED quantized accuracy, or NEEDS_ACCURACY / CONTRACT_MISMATCH.

    Never falls back to a Phase2F FP32 score, another architecture, or another profile.
    """
    body_hashes=body_hashes or contracts()
    spec=architectures().get(architecture_id)
    expected_arch=hashobj(spec) if spec is not None else None
    if architecture_hash is not None and architecture_hash!=expected_arch:
        return _blank('CONTRACT_MISMATCH',reason='architecture_hash_mismatch',architecture_id=architecture_id)
    architecture_hash=expected_arch
    canonical=canonical_profile(profile)
    if canonical is None:
        return _blank('NEEDS_ACCURACY',reason='unknown_quant_profile',architecture_id=architecture_id,
            accuracy_precision=profile)
    index,by_profile=accuracy_index(body_hashes)
    expected=by_profile[canonical]['quant_contract_hash']
    if quant_contract_hash is not None and quant_contract_hash!=expected:
        return _blank('CONTRACT_MISMATCH',reason='quant_contract_hash_mismatch',
            architecture_id=architecture_id,accuracy_precision=canonical,
            quant_contract_hash=quant_contract_hash,expected_quant_contract_hash=expected,
            architecture_hash=architecture_hash)
    found=index.get((architecture_id,canonical))
    if found is None:
        contract_row=index.get(('__contract__',canonical)) or index.get(('__contract__','mixed'))
        if contract_row and canonical!='W8A8':
            return _blank('CONTRACT_MISMATCH',reason=contract_row.get('reason'),
                architecture_id=architecture_id,accuracy_precision=canonical,
                quant_contract_hash=expected,architecture_hash=architecture_hash)
        if contract_row and canonical=='W8A8' and ('__contract__','W8A8') in index:
            return _blank('CONTRACT_MISMATCH',reason=contract_row.get('reason'),
                architecture_id=architecture_id,accuracy_precision=canonical,
                quant_contract_hash=expected,architecture_hash=architecture_hash)
        return _blank('NEEDS_ACCURACY',reason='no_quantized_accuracy_for_contract',
            architecture_id=architecture_id,accuracy_precision=canonical,
            quant_contract_hash=expected,architecture_hash=architecture_hash)
    out=dict(found)
    out['architecture_hash']=architecture_hash
    if architecture_hash is not None:
        out['accuracy_key']=dict(out.get('accuracy_key') or {},architecture_hash=architecture_hash)
    return out
