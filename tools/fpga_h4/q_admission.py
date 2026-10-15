"""Frozen network-conditioned Q(x) gate for the final B1/C/D joint domain.

Q(x) is read from the Phase5E freeze. It is not recomputed from hardware cost,
quantized accuracy, or Pareto membership. A measured (x, q) outside this set is
Q_NOT_ADMITTED.
"""
import json
import math
from functools import lru_cache
from common import ARCH, ROOT, contracts, hashobj
from candidate_graphs import architectures
from quant_contracts import canonical_profile, quant_contract, quantization_regions

Q_OF_X_PATH=ROOT/'reports/phase5e_mixed_precision/Q_OF_X.json'
PHASE5E_MANIFEST=ROOT/'reports/phase5e_mixed_precision/PHASE5E_MANIFEST.json'
FINAL_NETWORKS=('B1','C','D')
# Frozen Phase5E Q(x). Uniform W8A8 stays inside every listed network.
FROZEN_Q={
    'B1':('W8A8','HEAD_W8A16'),
    'C':('W8A8',),
    'D':('W8A8','HEAD_W16A16'),
}
LABEL_BY_ID={ARCH[label]:label for label in FINAL_NETWORKS}
_CACHE=None


def load_frozen_q():
    """Return the frozen Q(x) lists. Refuse a file that disagrees with the freeze."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    payload=json.loads(Q_OF_X_PATH.read_text(encoding='utf-8'))
    recorded=payload.get('Q') or {}
    if set(recorded)!=set(FROZEN_Q):
        raise ValueError('Frozen Q(x) network set drifted')
    for label,profiles in FROZEN_Q.items():
        if tuple(recorded[label])!=profiles:
            raise ValueError('Frozen Q('+label+') drifted')
    if payload.get('NETWORK_CONDITIONED_QUANTIZATION_SPACE') is not True:
        raise ValueError('Phase5E Q(x) freeze flag missing')
    if payload.get('uniform_baseline_retained') is not True:
        raise ValueError('Phase5E uniform W8A8 retention flag drifted')
    _CACHE={label:tuple(recorded[label]) for label in FINAL_NETWORKS}
    return _CACHE


def network_of(row):
    label=row.get('network')
    aid=row.get('architecture_id')
    resolved=LABEL_BY_ID.get(aid)
    if resolved is None:
        return None
    if label is not None and label not in (resolved,aid,'baseline' if resolved=='D' else resolved):
        return None
    return resolved


def profile_of(row):
    names=[row[k] for k in ('profile','quant_config','accuracy_precision') if row.get(k) is not None]
    profiles=[canonical_profile(n) if isinstance(n,str) else None for n in names]
    if not profiles or None in profiles or len(set(profiles))!=1:
        return None
    profile=profiles[0]
    if row.get('quant_contract_hash')!=_identities()[0][profile]:
        return None
    if row.get('quantization_config') is not None and row['quantization_config']!=quantization_regions(profile):
        return None
    return profile


@lru_cache(maxsize=1)
def _identities():
    body=contracts()
    return ({p:quant_contract(p,body)['quant_contract_hash'] for p in ('W8A8','HEAD_W8A16','HEAD_W16A16')},
        {aid:hashobj(spec) for aid,spec in architectures().items()})


def classify_q(row, frozen=None):
    frozen=frozen or load_frozen_q()
    label=network_of(row)
    profile=profile_of(row)
    aid=row.get('architecture_id')
    if row.get('architecture_hash')!=_identities()[1].get(aid) or aid not in _identities()[1]:
        return 'Q_NOT_ADMITTED','architecture_identity_mismatch',label,profile
    key=row.get('accuracy_key') or {}
    if any(key.get(k)!=row.get(k) for k in ('architecture_id','architecture_hash','quant_contract_hash')):
        return 'Q_NOT_ADMITTED','accuracy_identity_mismatch',label,profile
    if label not in frozen:
        return 'Q_NOT_ADMITTED','architecture_outside_frozen_final_set',label,profile
    if profile not in frozen[label]:
        return 'Q_NOT_ADMITTED','quant_contract_not_in_frozen_Q',label,profile
    return 'Q_ADMITTED','quant_contract_in_frozen_Q',label,profile


def measurement_pair(row):
    latency=row.get('serial_op_estimate_us', row.get('Latency'))
    accuracy=row.get('Accuracy')
    if not isinstance(accuracy,(int,float)) or not math.isfinite(accuracy) or not 0<=accuracy<=1:
        return False
    if not isinstance(latency,(int,float)) or not math.isfinite(latency) or latency<0:
        return False
    return (row.get('hardware_status')=='MEASURED' and row.get('status')=='LOOKED_UP'
        and row.get('accuracy_status')=='MEASURED' and row.get('quantized_accuracy_bound') is True
        and row.get('Accuracy') is not None and latency is not None)


def annotate_final_eligibility(row):
    """Set q_status and eligible. Measured hardware plus accuracy cannot bypass Q(x)."""
    status,reason,label,profile=classify_q(row)
    peak=(row.get('resource') or {}).get('peak_operator') or {}
    if row.get('hardware_status')=='MEASURED':
        if row.get('serial_op_estimate_us') is None:
            serial=(row.get('latency') or {}).get('serial_op_estimate_us')
            if serial is None:
                serial=row.get('Latency')
            if serial is not None:
                row['serial_op_estimate_us']=serial
        if row.get('FF') is None and (row.get('FF_peak') is not None or peak.get('ff') is not None):
            row['FF']=row.get('FF_peak', peak.get('ff'))
    row['final_network']=label
    row['q_profile']=profile
    row['q_status']=status
    row['q_reason']=reason
    row['measurement_pair']=bool(measurement_pair(row))
    row['eligible']=status=='Q_ADMITTED' and row['measurement_pair']
    return row


def final_eligible(row):
    annotate_final_eligibility(row)
    return row.get('q_status')=='Q_ADMITTED' and row.get('eligible') is True and measurement_pair(row)
