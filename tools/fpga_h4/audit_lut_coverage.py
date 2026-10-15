"""Operator LUT vs frozen V0.1 144-space. Exact lookup only; no INT8 fill-in."""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/fpga_h4'))
from candidate_graphs import architectures, candidate_graph
from cost_api import CostDatabase, atomic_usage
from quant_contracts import PROFILES, node_precision
from common import contracts

W8A8 = dict(input_bitwidth=8, weight_bitwidth=8, accumulator_bitwidth=32, output_bitwidth=8)
W8A16 = dict(input_bitwidth=16, weight_bitwidth=8, accumulator_bitwidth=32, output_bitwidth=16)
W16A16 = dict(input_bitwidth=16, weight_bitwidth=16, accumulator_bitwidth=32, output_bitwidth=16)
POLICIES = ['reference_low', 'reference_high', 'reference_min_latency',
            'reference_min_dsp', 'reference_min_lut', 'reference_min_bram']


def pool_for(db, shape_id, bits):
    rows = [r for r in db.options.get(shape_id, []) if all(r[k] == v for k, v in bits.items())]
    rows += [r for r in db.extension_rows if r['shape_id'] == shape_id and all(r[k] == v for k, v in bits.items())]
    return rows


def main():
    db = CostDatabase()
    hashes = contracts()
    aids = architectures()
    shapes = {}
    unknown = []
    pack_desc = set()
    for aid in aids:
        graph = candidate_graph(aid, hashes)
        usage, unk = atomic_usage(graph)
        if unk:
            unknown.append(dict(architecture_id=aid, unknown=unk))
        for u in usage:
            rec = shapes.setdefault(u['shape_id'], dict(u, nodes=set(), arches=set(), is_head=False))
            rec['nodes'].add(u['node'])
            rec['arches'].add(aid)
            rec['is_head'] = rec['is_head'] or str(u['node']).startswith('head.')
        for d in graph.get('descriptors') or []:
            pack_desc.add((d.get('name'), d.get('height'), d.get('width'), d.get('real_bytes'), d.get('padded_bytes')))

    missing_ops = []
    shape_rows = []
    for sid, rec in sorted(shapes.items(), key=lambda kv: (kv[1]['operator_type'], kv[1]['H'], kv[1]['kernel'], kv[0])):
        body = pool_for(db, sid, W8A8)
        row = dict(shape_id=sid, operator_type=rec['operator_type'], Cin=rec['Cin'], Cout=rec['Cout'],
                   H=rec['H'], W=rec['W'], kernel=rec['kernel'], stride=rec['stride'], groups=rec['groups'],
                   activation=rec['activation'], is_head=rec['is_head'], n_arch=len(rec['arches']),
                   nodes=sorted(rec['nodes']), n_w8a8=len(body), n_w8a16=0, n_w16a16=0)
        if not body:
            missing_ops.append(dict(row, precision='W8A8', reason='no_valid_csynth_row'))
        if rec['is_head']:
            h8 = pool_for(db, sid, W8A16)
            h16 = pool_for(db, sid, W16A16)
            row['n_w8a16'] = len(h8)
            row['n_w16a16'] = len(h16)
            if not h8:
                missing_ops.append(dict(row, precision='HEAD_W8A16', reason='no_valid_csynth_row'))
            if not h16:
                missing_ops.append(dict(row, precision='HEAD_W16A16', reason='no_valid_csynth_row'))
        shape_rows.append(row)

    pack_missing = []
    for d in sorted(pack_desc):
        match = [r for r in db.pack_rows if r['operator_type'] == 'DMARepack'
                 and r['H'] == d[1] and r['W'] == d[2] and r['real_bytes'] == d[3] and r['padded_bytes'] == d[4]]
        if not match:
            pack_missing.append(dict(name=d[0], height=d[1], width=d[2], real_bytes=d[3], padded_bytes=d[4]))

    net_miss = defaultdict(int)
    examples = []
    looked = 0
    total = 0
    for aid in aids:
        for q in PROFILES:
            for policy in POLICIES:
                total += 1
                r = db.estimate_cost(aid, q, policy)
                if r['status'] == 'LOOKED_UP':
                    looked += 1
                else:
                    net_miss[(q, policy, r['status'], r.get('reason'))] += 1
                    if len(examples) < 12:
                        examples.append(dict(architecture_id=aid, profile=q, mapping=policy,
                                             status=r['status'], reason=r.get('reason'),
                                             missing=(r.get('missing') or [])[:2]))

    report = dict(
        architectures=len(aids),
        unique_compute_shapes=len(shapes),
        head_shapes=sum(1 for r in shape_rows if r['is_head']),
        body_shapes=sum(1 for r in shape_rows if not r['is_head']),
        unknown_operators=unknown,
        missing_operator_rows=[{k: m[k] for k in ('shape_id', 'precision', 'operator_type', 'Cin', 'Cout',
                                                   'H', 'kernel', 'stride', 'groups', 'activation', 'n_arch', 'reason')
                                 if k in m} for m in missing_ops],
        packing_descriptors=len(pack_desc),
        packing_missing=pack_missing,
        network_lookups=total,
        network_looked_up=looked,
        network_missing=total - looked,
        network_miss_groups={str(k): v for k, v in net_miss.items()},
        network_miss_examples=examples,
        body_shapes_multi_mapping=sum(1 for r in shape_rows if not r['is_head'] and r['n_w8a8'] > 1),
        body_shapes_single_mapping=sum(1 for r in shape_rows if not r['is_head'] and r['n_w8a8'] == 1),
        head_mixed_singleton=sum(1 for r in shape_rows if r['is_head'] and (r['n_w8a16'] <= 1 or r['n_w16a16'] <= 1)),
        deployment='NOT_CLAIMED',
        latency_fidelity='HLS_CSYNTH_SERIAL_ESTIMATE',
        shapes=[{k: r[k] for k in r if k != 'nodes'} for r in shape_rows],
    )
    out = ROOT / 'results/sonar_joint_v1/lut_space_audit.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in (
        'architectures', 'unique_compute_shapes', 'head_shapes', 'body_shapes',
        'missing_operator_rows', 'packing_missing', 'network_lookups',
        'network_looked_up', 'network_missing', 'network_miss_groups',
        'body_shapes_multi_mapping', 'body_shapes_single_mapping',
        'head_mixed_singleton')}, indent=2))
    print('SHAPES')
    for r in shape_rows:
        print(f"{r['shape_id']} {r['operator_type']:12} Cin={r['Cin']:4} Cout={r['Cout']:4} "
              f"H={r['H']:3} k={r['kernel']} s={r['stride']} g={r['groups']:3} {r['activation']:12} "
              f"head={int(r['is_head'])} w8a8={r['n_w8a8']} w8a16={r['n_w8a16']} "
              f"w16a16={r['n_w16a16']} arch={r['n_arch']}")


if __name__ == '__main__':
    main()
