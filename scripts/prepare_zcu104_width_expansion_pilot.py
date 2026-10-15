"""Freeze the infeasible space and prepare a bounded mapping-A hardware pilot.

No training, architecture search, fitted cost extrapolation, or board measurement.
Run from the repository root; outputs are an isolated, portable experiment bundle.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "hls_lut_builder/scripts"), str(ROOT / "scripts")]
from common import build_cases, load_config as load_builder_config, render_template
from hwnas_fpga.runtime import load_config, build_constraints, build_search_space, build_cost_estimator, build_hardware_spec
from probe_zcu104_mobilenetA_proxy_feasibility import enumerate_proxy

SEARCH = "configs/search/zcu104_mobilenetA_v2_rl_entropy_pareto_feasible_budget.yaml"
BASE_BUILDER = "hls_lut_builder/configs/candidate_kernels_zcu104_mobilenetA_full_dmdc_cfem_v1.yaml"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prepare(output: Path, *, enumerate_original: bool = True) -> dict:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Use a new experiment directory: {output}")
    output.mkdir(parents=True)
    cfg = load_config(str(ROOT / SEARCH))
    constraints = build_constraints(cfg)
    space = build_search_space(cfg, image_size=224, input_channels=1, num_classes=8, constraints=constraints)
    estimator = build_cost_estimator(cfg, hardware_spec=build_hardware_spec(cfg), constraints=constraints)
    baseline = space.baseline_architecture()
    from dataclasses import replace
    lower = []
    chosen = []
    for i, stage in enumerate(baseline.stages):
        entries = []
        for block in space.concrete_block_choices_for_stage(stage_index=i,
                in_channels=baseline.stem_channels if i == 0 else baseline.stages[i-1].channels,
                out_channels=stage.channels, stride=stage.stride):
            stages = list(baseline.stages)
            stages[i] = replace(stage, blocks=(block,))
            row = estimator._selected_rows(replace(baseline, stages=tuple(stages)), space)[i+1]
            entries.append((float(row['cost']['bram_tile_post_route']), block, row))
        value, block, row = min(entries, key=lambda x: x[0])
        chosen.append(replace(stage, blocks=(block,)))
        lower.append(dict(stage=i+1, channels=stage.channels, minimum_bram36=value,
            selected_block=block.to_dict(), row_key=row['row_key'],
            choice_count=len(entries)))
    minimum = replace(baseline, stages=tuple(chosen))
    rows = estimator._selected_rows(minimum, space)
    exact_sum = sum(float(row['cost']['bram_tile_post_route']) for row in rows)
    snapshot_paths = [SEARCH, BASE_BUILDER,
        "src/hwnas_fpga/hardware/mobilenetA_operator_cost.py",
        "src/hwnas_fpga/search_space/space.py", "src/hwnas_fpga/models/builder.py",
        "src/hwnas_fpga/runtime.py", "src/hwnas_fpga/hardware/cost.py",
        "scripts/probe_zcu104_mobilenetA_proxy_feasibility.py",
        "scripts/prepare_zcu104_width_expansion_pilot.py",
        cfg['hardware']['operator_cost_lut_path'], cfg['hardware']['operator_cost_lut_coverage_path'],
        cfg['hardware']['operator_cost_lut_preflight_path']]
    portable_files = list((ROOT / "hls_lut_builder/scripts").glob("*.py"))
    portable_files += list((ROOT / "hls_lut_builder/scripts").glob("*.tcl"))
    portable_files += list((ROOT / "hls_lut_builder/include").glob("*.h"))
    templates = ['stem_conv_k3_s2', 'pw_conv', 'mbconv', 'mbconv_e1_noexpand', 'global_avg_pool', 'fc_layer', 'skip']
    portable_files += [ROOT / f"hls_lut_builder/templates/{name}.cpp.tmpl" for name in templates]
    hashes = {}
    for rel in snapshot_paths:
        src = ROOT / rel
        dest = output / 'source_snapshot' / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        hashes[rel] = sha(src)
    for src in portable_files:
        rel = src.relative_to(ROOT)
        dest = output / 'bundle' / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        hashes[str(rel).replace('\\', '/')] = sha(src)
    dump(output / 'source_freeze.json', {'created_at': datetime.now(timezone.utc).isoformat(), 'files': hashes})
    proof = {
        'status': 'EXHAUSTIVELY_PROXY_INFEASIBLE' if enumerate_original else 'ANALYTIC_PROXY_LOWER_BOUND',
        'source_config_sha256': hashes[SEARCH], 'legal_architecture_count': math.prod(x['choice_count'] for x in lower),
        'minimum_bram36_sum': exact_sum, 'minimum_bram36_estimate': math.ceil(exact_sum),
        'max_bram36': constraints.max_bram, 'required_reduction_fraction': (math.ceil(exact_sum)-constraints.max_bram)/math.ceil(exact_sum),
        'last_two_stages_bram36': sum(x['minimum_bram36'] for x in lower[-2:]),
        'last_two_stages_fraction': sum(x['minimum_bram36'] for x in lower[-2:])/exact_sum,
        'per_stage_minimum': lower, 'minimizing_architecture': minimum.to_dict(),
        'proof': 'Independent per-stage options and additive BRAM cost; sum of minima is attainable. Exact enumeration verifies all legal architectures.',
        'claim_boundary': 'No proxy-feasible network in this frozen space under this estimator and constraints. Not a universal physical impossibility theorem.',
        'nas_resume_allowed': False,
    }
    if enumerate_original:
        enumeration = enumerate_proxy(ROOT / SEARCH, topk=1)
        assert enumeration['candidate_count'] == proof['legal_architecture_count']
        assert enumeration['feasible_count'] == 0
        dump(output / 'original_space_enumeration.json', enumeration)
        proof['feasible_count'] = enumeration['feasible_count']
    dump(output / 'original_space_conclusion.json', proof)
    dump(output / 'mapping_contract.json', {
        'mapping_id': 'mapping_A_independent_blocks_packed_int8_pi4_po4_u1_v1',
        'network_instantiation': 'one physical instance per network block; no inter-layer engine sharing',
        'resource_model': 'operator sum plus separately measured interfaces, parameter storage, FIFOs and control overhead',
        'part': 'xczu7ev-ffvc1156-2-e', 'clock_mhz': 200, 'bitwidth': 8,
        'pi': 4, 'po': 4, 'unroll_factor': 1, 'array_partition_factor': 4,
        'fixed': ['stage_count=8', 'stage_strides', 'depth=1', 'stem', 'head_output_channels=1280', 'FC8 interface'],
        'pilot_variables': ['S6/S7/S8 width', 'S7/S8 expansion'],
        'head_input_tracks_S8': True, 'sonar_slot': 'skip for hardware boundary pilot; DMDC and CFEM retained for subsequent admission',
        'latency_scope': 'sum of isolated HLS cycles at 200 MHz, not measured board latency or steady-state throughput',
        'legacy_lutram_2p01_policy': 'risk_only_for_pilot; never treated as measured LUT-as-memory',
        'hard_bram36_cap': 312, 'hard_board_lutram_sites_cap_on_implementation': 101760,
        'e1_semantics': 'no expansion convolution and no expansion ReLU, matching PyTorch MBConvBlock; separate kernel and semantic witness',
        'full_network_route_required_for_feasible_seed': True,
    })
    base = yaml.safe_load((ROOT / BASE_BUILDER).read_text(encoding='utf-8'))
    builder = {key: copy.deepcopy(base[key]) for key in ['toolchain', 'defaults', 'clock_profiles']}
    builder['implementation_profiles'] = {'balanced_pi4_po4_u1': base['implementation_profiles']['balanced_pi4_po4_u1']}
    builder['representative_shapes'] = {}
    builder['operators'] = {}
    builder['workspace'] = {'project_root': '../../work/p', 'synthesis_summary_json': '../../work/synthesis_summary.json'}
    builder['measurement_contract'] = {'mapping_id': 'mapping_A_independent_blocks_packed_int8_pi4_po4_u1_v1', 'search_ready': False}
    pilot_specs = []
    op_by_key = {}
    old_by_spec = {}
    for row in estimator.rows_by_pair.values():
        spec = row['op_spec']
        key = (row['operator'], spec.get('in_channels'), spec.get('out_channels'), tuple(spec.get('input_resolution', [])), spec.get('stride'), spec.get('kernel_size'), spec.get('expand_ratio'))
        old_by_spec[key] = row

    def add_case(kind, cin, cout, resolution, stride=1, kernel=1, expand=1, **extra):
        key = (kind, cin, cout, resolution, stride, kernel, expand)
        if key in op_by_key:
            return op_by_key[key]
        name = f'{kind}_h{resolution}_i{cin}_o{cout}_s{stride}_k{kernel}_e{expand}'
        op_by_key[key] = name
        params = dict(feature_h=resolution, feature_w=resolution, in_channels=cin, out_channels=cout,
                      stride=stride, kernel_size=kernel, padding=kernel//2 if kernel>1 else 0, expand_ratio=expand, **extra)
        template = 'mbconv_e1_noexpand' if kind=='mbconv' and expand==1 else ('stem_conv_k3_s2' if kind=='stem' else kind)
        top = f'mbconv_e{expand}_k{kernel}_kernel' if kind=='mbconv' else f'{template}_kernel'
        builder['representative_shapes'][name] = params
        builder['operators'][name] = dict(enabled=True, template=f'../templates/{template}.cpp.tmpl',
            top_function=top, op_type=f'mbconv_e{expand}_k{kernel}' if kind=='mbconv' else template,
            shape_refs=[name], implementation_refs=['balanced_pi4_po4_u1'], clock_profile_refs=['main_5ns'],
            parameters=params, op_spec_defaults={'op': kind, 'kernel_size': kernel, 'expand_ratio': expand})
        return name

    # Lightest-first queue after the original baseline. Width x expansion is a 4x3 factorial.
    widths = [('baseline', [96,160,320]), ('pilot_c', [80,112,192]), ('pilot_b', [80,128,256]), ('pilot_a', [96,128,256])]
    for label, tail in widths:
        for expansion in ([3,1,2] if label=='baseline' else [1,2,3]):
            arch = copy.deepcopy(minimum.to_dict())
            for i, channel in zip([5,6,7], tail):
                arch['stages'][i]['channels'] = channel
            for i in [6,7]:
                arch['stages'][i]['blocks'][0]['expand_ratio'] = expansion
            names = [add_case('stem', 1,32,224,2,3,apply_relu=1)]
            res, cin = 112, 32
            for i, stage in enumerate(arch['stages']):
                block = stage['blocks'][0]
                kind = 'pw_conv' if block['op']=='conv' else block['op']
                kw = {'apply_relu': 0} if kind=='pw_conv' else {}
                names.append(add_case(kind,cin,stage['channels'],res,stage['stride'],block['kernel_size'],block['expand_ratio'],**kw))
                cin, res = stage['channels'], math.ceil(res/stage['stride'])
            names += [add_case('pw_conv',cin,1280,7,apply_relu=1), add_case('global_avg_pool',1280,1280,7), add_case('fc_layer',1280,8,1)]
            pid = f'{label}_tail_e{expansion}'
            dump(output / 'architectures' / f'{pid}.json', arch)
            pilot_specs.append({'pilot_id':pid, 'channels':[st['channels'] for st in arch['stages']], 'tail_expansion':expansion,
                                'operator_names':names, 'architecture':arch, 'e1_witness_required':expansion==1,
                                'status':'PENDING_HLS_ROUTE', 'proxy_feasible':None, 'implementation_feasible':None})
    config_path = output / 'bundle/hls_lut_builder/configs/width_expansion_pilot.yaml'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(builder,sort_keys=False),encoding='utf-8')
    cases = build_cases(load_builder_config(config_path))
    case_by_op = {case.operator:case for case in cases}
    operator_manifest = []
    for case in cases:
        source = render_template(case.template_path.read_text(encoding='utf-8'), {'top_function':case.top_function, **case.parameters})
        old_kind = case.op_spec['op']
        if old_kind=='stem': old_kind='stem_conv_k3_s2'
        if old_kind=='mbconv': old_kind=f"mbconv_e{case.parameters['expand_ratio']}_k{case.parameters['kernel_size']}"
        old_key=(old_kind,case.parameters['in_channels'],case.parameters['out_channels'],(case.parameters['feature_h'],case.parameters['feature_w']),case.parameters['stride'],case.parameters['kernel_size'],case.parameters['expand_ratio'])
        old=old_by_spec.get(old_key)
        operator_manifest.append({'operator_name':case.operator,'case_name':case.case_name,'top_function':case.top_function,
            'source_sha256_lf':hashlib.sha256(source.replace('\r\n','\n').encode()).hexdigest(),
            'parameters':case.parameters, 'legacy_row':old,
            'reuse_rule':'Require exact generated source, headers, HLS report SHA, and report files; old costs alone are not reusable.'})
    for pilot in pilot_specs:
        pilot['case_names']=[case_by_op[name].case_name for name in pilot['operator_names']]
    dump(output/'pilot_manifest.json', {'schema_version':1,'pilot_count':len(pilot_specs),'unique_operator_count':len(cases),
        'pilots':pilot_specs,'operators':operator_manifest,'nas_resume_allowed':False,
        'new_space_gate': {'sample_count':1000,'seed':42,'zero_sample_feasible_means':'no feasible architecture found; not an exhaustive proof',
            'required':['exact-shape cost coverage','at least one full-route feasible seed','nonempty proxy-feasible sample','predictor calibration held-out check']}})
    dump(output/'status.json', {'stage':'PREPARED','original_space':'PROXY_INFEASIBLE','mapping':'A_FROZEN',
        'pilot_count':len(pilot_specs),'unique_operators':len(cases),'feasible_seed':None,'nas_resume_allowed':False})
    text = f'''# ZCU104 搜索空间不可行结论与 width/e 硬件试点

本实验固定 mapping A（各层独立实例化），保持 8 阶段、stride、depth、INT8、Stem、1280 通道 Head 与 FC8 输出接口。

## 已归档结论

当前冻结配置共 {proof['legal_architecture_count']} 个合法架构。算子成本求和的最小 BRAM36 为 {exact_sum}，估计器取整为 {math.ceil(exact_sum)}，高于 312。可行集合在该估计器与约束下为空，不能归因于 RL 没搜到。

至少需降低 {proof['required_reduction_fraction']:.2%}；最后两阶段合计 229 BRAM36，占最小未取整总和 {proof['last_two_stages_fraction']:.2%}。这是模型条件下的穷举结论，不是对所有 FPGA 映射方式的物理不可行证明。

## 试点设计

4 组后部通道 × S7/S8 的 e1/e2/e3，共 12 点。前部沿用最小 BRAM 见证架构（S3=k5/e6、S5=k3/e6），各点只改变尾部 width/e，S4 固定 Skip。DMDC/CFEM 保留在后续空间的声呐 slot，不作为本轮缩宽首要变量。

所有改变输入通道的相邻算子及 Head 输入均重新建成本键，不缩放旧记录。冻结模板和公共头文件；旧成本仅在源码、头文件、原始报告哈希相符时复用。

e1 必须省略扩展卷积及其 ReLU；通过独立整数参考的数值检查后才允许硬件合成。e2/e3 使用冻结原模板。

## 判定与后续

2.01×总LUT 仅记录风险指标。BRAM36≤312、实际实现 LUTRAM≤101760、LUT/DSP/FF 容量、时序约束仍保留。未知字段不得当作 0。算子求和可行不等于整网可行；第一个 feasible seed 必须通过整网实现。

先完成 12 点算子测量与候选聚合，再对最轻候选做整网 HLS/Route，采集 LUT-as-Logic、LUT-as-Memory、BRAM36、DSP、FF、WNS/TNS。延迟分别保留 HLS 周期、整网仿真/硬件计时，Route 本身不能生成实测 latency。用整网预测误差校准后，再定义新空间并以 seed=42 抽样 1000 点。0/1000 仅代表未发现可行点，停止启动 NAS并继续诊断，不能声称穷举不可行。

NAS 默认禁止恢复，直到成本覆盖、整网 feasible seed、校准和新空间可行率检查均具备证据。RL 控制器/采样脱节问题也须在恢复前单独解决，不混入本轮 width/e 硬件试点。
'''
    (output/'README.md').write_text(text,encoding='utf-8')
    return {'output':str(output),'pilot_count':len(pilot_specs),'unique_operator_count':len(cases),'minimum_bram36':math.ceil(exact_sum)}


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    print(json.dumps(prepare(args.output),ensure_ascii=False))
