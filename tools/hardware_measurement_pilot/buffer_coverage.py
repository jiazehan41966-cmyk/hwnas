"""Analytic full-frame BRAM lower bound, never a routed feasibility result."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


def analyze(inventory, synthesis_report):
    inventory, synthesis_report = Path(inventory), Path(synthesis_report)
    data = json.loads(inventory.read_text(encoding='utf-8'))
    available = ET.parse(synthesis_report).find('.//AvailableResources')
    blocks = int(available.findtext('BRAM_18K'))
    bits = blocks * 18 * 1024
    implemented = {'conv1x1', 'DW3', 'DW5', 'nearest', 'requant_add', 'body8_to_head16'}
    rows = []
    for primitive in data['primitive_inventory']:
        shape, precision = primitive['shape'], primitive['precision']
        input_bits = sum(math.prod(tensor) for tensor in shape['input_shapes']) * precision['input_bitwidth']
        output_bits = math.prod(shape['output_shape']) * precision['output_bitwidth']
        total = input_bits + output_bits
        family = primitive['family']
        rows.append(dict(primitive_id=primitive['primitive_id'], family=family,
            input_shapes=shape['input_shapes'], output_shape=shape['output_shape'], precision=precision,
            declared_full_frame_activation_bits=total, theoretical_bram18_lower_bound=math.ceil(total / 18432),
            available_bram_fraction=total/bits, operator_template_exists=family in implemented,
            assessment='EXCEEDS_DEVICE_BRAM_UNDER_CURRENT_FULL_FRAME_TEMPLATE' if family in implemented and total > bits
                else 'LOWER_BOUND_ONLY_NOT_A_FIT_PROOF' if family in implemented else 'NO_PILOT_OPERATOR_TEMPLATE'))
    rows.sort(key=lambda row: -row['declared_full_frame_activation_bits'])
    return dict(schema='pilot_full_frame_buffer_coverage_v1', evidence_layer='ANALYTIC_STORAGE_LOWER_BOUND_NOT_SYNTHESIS_OR_ROUTE',
        source_inventory=str(inventory.resolve()), source_inventory_sha256=hashlib.sha256(inventory.read_bytes()).hexdigest(),
        capacity_report=str(synthesis_report.resolve()), capacity_report_sha256=hashlib.sha256(synthesis_report.read_bytes()).hexdigest(),
        available_bram18=blocks, available_bram_bits=bits,
        assumptions=['Current pilot kernel explicitly binds full input/output frame arrays to BRAM.',
            'Excludes parameters, padding, memory bank fragmentation, FIFOs and shell BRAM: this is only a lower bound.',
            'Does not claim the network or layer is infeasible with tiling, streaming, URAM or another implementation.',
            'Trial P4 timings alone cannot establish implementation coverage or resource fit of every larger structural key.'],
        primitive_count=len(rows), exceeds_bram_lower_bound_count=sum(row['assessment'].startswith('EXCEEDS') for row in rows),
        absent_template_count=sum(not row['operator_template_exists'] for row in rows),
        formal_cost_eligible=False, rows=rows)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', required=True)
    parser.add_argument('--capacity-report', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    result=analyze(args.inventory,args.capacity_report)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:result[key] for key in ('primitive_count','available_bram18','exceeds_bram_lower_bound_count','absent_template_count')}))
