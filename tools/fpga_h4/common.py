"""H4 mapping-search utilities. Exact lookup only; no NAS or unseen-mapping prediction."""
import json
import sys
from pathlib import Path
import importlib.util

ROOT=Path(__file__).resolve().parents[2]
_h3=ROOT/'tools/fpga_h3/common.py'
_spec=importlib.util.spec_from_file_location('fpga_h3_common',_h3)
_mod=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=_mod;_spec.loader.exec_module(_mod)
ART=_mod.ART
H3_OUT=_mod.OUT
PART=_mod.PART
ROOT=_mod.ROOT
contracts=_mod.contracts
csvwrite=_mod.csvwrite
hashobj=_mod.hashobj
legality=_mod.legality
save=_mod.save
sha=_mod.sha
shape_key=_mod.shape_key
_collect_spec=importlib.util.spec_from_file_location('fpga_h3_collect',ROOT/'tools/fpga_h3/collect.py')
_collect=importlib.util.module_from_spec(_collect_spec)
_collect_spec.loader.exec_module(_collect)
canonical_h3_mapping=_collect.canonical_mapping

H4_OUT=ROOT/'reports/fpga_h4_mapping_search_v2'
H4_TOOLING_OUT=ROOT/'reports/fpga_h4_tooling_v2'
H3_MANIFEST=H3_OUT/'H3_DATABASE_MANIFEST.json'
ACCURACY_SET=ROOT/'reports/phase2f_multiseed/network_candidate_set.json'
ACCURACY_CSV=ROOT/'reports/phase2f_multiseed/multiseed_results.csv'
DEVICE=dict(LUT=230400,FF=460800,DSP=1728,BRAM18=624,URAM=96)
NETWORKS=['baseline','B1','C','D']
PARETO_NETWORKS=['B1','C','D']
ARCH={
    'baseline':'P3_P4_P5_s1k3_s2k3_s4k3_s5k3_s6k3_s7k3',
    'B1':'P2_P3_P4_P5_s1k3_s2k5_s4k5_s5k5_s6k5_s7k5',
    'C':'P3_P4_P5_s1k3_s2k5_s4k5_s5k3_s6k5_s7k3',
    'D':'P3_P4_P5_s1k3_s2k3_s4k3_s5k3_s6k3_s7k3',
}
AUX_OPS={'ChainTowerHead','DMARepack'}
METRICS=['latency_us','DSP','LUT','FF','BRAM18','URAM']
MAPPING_KEY=(
    'numeric_contract_hash','interface_contract_hash',
    'input_bitwidth','weight_bitwidth','accumulator_bitwidth','output_bitwidth',
    'PE','SIMD','input_tile','output_tile','spatial_tile','buffer_profile',
    'part','tool_version','target_clock_ns','source_hash','source_report_sha256',
)

EXACT_MAPPING_FIELDS=MAPPING_KEY
V1_BITS=dict(input_bitwidth=8,weight_bitwidth=8,accumulator_bitwidth=32,output_bitwidth=8)
PLATFORM=dict(part=PART,tool_version='2022.2',target_clock_ns=5)

def canonical_graph_hash(graph):
    payload={k:graph.get(k) for k in ('architecture','nodes','edges','outputs','descriptors','numeric_contract_hash','interface_contract_hash')}
    return hashobj(payload)

def relpath_hash(rel):
    p=ROOT/rel
    return sha(p) if p.is_file() else None
