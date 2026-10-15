"""Read standalone V0.2 HLS evidence without upgrading its evidence scope."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / 'results/detector_v0_2_execution_20261002/hardware'
REPORTS = ROOT / 'reports/detector_v0_2_execution_20261002'
EXPECTED = ['numeric_boundary', 'pw48', 'prediction21', 'dw48', 'dw48_stride2', 'stem32']


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def xml_metrics(path: Path) -> dict:
    root = ET.parse(path).getroot()
    fields = {
        'estimated_clock_ns': './/SummaryOfTimingAnalysis/EstimatedClockPeriod',
        'latency_cycles_min': './/SummaryOfOverallLatency/Best-caseLatency',
        'latency_cycles_max': './/SummaryOfOverallLatency/Worst-caseLatency',
        'top_interval_min': './/SummaryOfOverallLatency/Interval-min',
        'top_interval_max': './/SummaryOfOverallLatency/Interval-max',
        **{x: './/AreaEstimates/Resources/' + x for x in ['DSP', 'LUT', 'FF', 'BRAM_18K', 'URAM']},
    }
    return {key: root.findtext(xpath) for key, xpath in fields.items()}


def collect(attempt: str) -> dict:
    folder = ARTIFACTS / attempt
    evidence = folder / 'remote_evidence'
    state = json.loads((evidence / 'status.json').read_text())
    uploaded = json.loads((folder / 'source_manifest.json').read_text())
    source_bound = all((folder / p).is_file() and sha(folder / p) == digest
                       for p, digest in uploaded.items())
    source_bound &= sha(folder / 'source_manifest.json') == state['source_manifest_sha256']
    rows, states = [], {r['name']: r for r in state['statuses']}
    for name in EXPECTED:
        case = json.loads((folder / name / 'case.json').read_text())
        status = states.get(name)
        row = dict(name=name, source_bound=bool(source_bound), case=case,
                   csim_status='NOT_COMPLETED', csynth_status='NOT_COMPLETED',
                   route_status='NOT_RUN', board_status='NOT_RUN')
        if status:
            refs = list((folder / name).glob('*reference.bin'))
            pairs = {p['test']: p for p in status['parity']}
            parity_bound = bool(refs) and len(pairs) == len(refs)
            for ref in refs:
                pair = pairs.get(ref.stem, {})
                parity_bound &= (pair.get('binary_exact') is True and
                                  pair.get('reference_sha256') == sha(ref) and
                                  pair.get('actual_sha256') == sha(ref))
            row['run_status'] = status
            # The runner's return code combines C simulation and synthesis;
            # preserve a genuine CSIM pass even if a subsequent synthesis fails.
            log = evidence / name / 'hls.log'
            text = log.read_text(errors='replace') if log.is_file() else ''
            row['csim_status'] = ('PASS' if source_bound and parity_bound and
                                  status['csim_pass'] and 'CSim done with 0 errors.' in text
                                  else 'FAILED_EVIDENCE')
            if name == 'numeric_boundary':
                row['csynth_status'] = 'NOT_REQUESTED'
            else:
                top = evidence / name / 'project/solution1/syn/report' / (case['top'] + '_csynth.xml')
                if source_bound and status['returncode'] == 0 and top.is_file():
                    row['csynth_status'] = 'PASS'
                    row['hls_estimates'] = xml_metrics(top)
                    row['top_report'] = str(top.relative_to(ROOT))
                    row['top_report_sha256'] = sha(top)
                else:
                    row['csynth_status'] = 'FAILED_EVIDENCE'
            row['pipeline_results'] = [
                dict(target_ii=int(a), final_ii=int(b), depth=int(c), loop=d)
                for a, b, c, d in re.findall(
                    r'Target II = (\d+), Final II = (\d+), Depth = (\d+), loop \'([^\']+)\'', text)
            ]
            row['pipeline_log'] = str(log.relative_to(ROOT)) if log.is_file() else None
            row['pipeline_log_sha256'] = sha(log) if log.is_file() else None
            rtl = evidence / name / 'project/solution1/syn/verilog'
            compute_name = 'DW_MAC_STEPS' if name.startswith('dw') else 'DENSE_MAC_STEPS'
            cores = list(rtl.glob('*_Pipeline_' + compute_name + '.v'))
            if len(cores) == 1:
                core = cores[0]
                coretext = core.read_text(errors='replace')
                # Count signed INT8 x INT8 compute multipliers/MACs only;
                # address arithmetic also contains unrelated multipliers.
                instances = re.findall(r'^\s*(\w+_8s_8s_\w+)\s+(?:#\s*\(|\w+\s*\()',
                                       coretext, re.MULTILINE)
                muls = [module for module in instances
                        if '_mul_' in module or '_mac_muladd_' in module]
                row['compute_multiplier_instances'] = len(muls)
                row['compute_multiplier_modules'] = sorted(set(muls))
                row['compute_rtl'] = str(core.relative_to(ROOT))
                row['compute_rtl_sha256'] = sha(core)
            top_rtl = rtl / (case.get('top', '') + '.v')
            if top_rtl.is_file():
                rtltext = top_rtl.read_text(errors='replace')
                row['axi_data_width_parameters'] = dict(re.findall(
                    r'parameter\s+C_M_AXI_(\w+)_DATA_WIDTH\s*=\s*(\d+)', rtltext))
        rows.append(row)
    passed = (source_bound and set(states) == set(EXPECTED) and
              all(r['csim_status'] == 'PASS' and r['csynth_status'] in ['PASS', 'NOT_REQUESTED'] for r in rows))
    mapping = {}
    for row in rows:
        if row['name'] == 'numeric_boundary':
            continue
        expected_lanes = 32 if row['name'].startswith('dw') else (16 if row['name'] == 'stem32' else 128)
        actual_loops = [p for p in row.get('pipeline_results', [])
                        if p['loop'] in ['DENSE_MAC_STEPS', 'DW_MAC_STEPS']]
        widths = row.get('axi_data_width_parameters', {})
        valid = (row.get('compute_multiplier_instances') == expected_lanes and bool(actual_loops) and
                 actual_loops[-1]['final_ii'] == 1 and widths.get('ACTIVATIONS') == '128' and
                 widths.get('PARAMETERS') == '128')
        row['mapping_witness'] = dict(expected_compute_lanes=expected_lanes,
                                     status='PASS' if valid else 'NOT_VERIFIED')
        mapping[row['name']] = valid
    result = dict(attempt=attempt, evidence_scope='standalone synthetic operator CSIM and HLS estimates',
                  source_bound=bool(source_bound), acceptance='PASS' if passed else 'INCOMPLETE_OR_FAILED',
                  mapping_acceptance='PASS' if passed and all(mapping.values()) else 'NOT_VERIFIED',
                  active=state.get('active'), runner_complete=state.get('complete', False), rows=rows,
                  detector_graph_parity='NOT_VERIFIED', detector_board_latency='NOT_MEASURED',
                  limitations=['No whole-graph scheduler', 'No ping-pong/dataflow overlap',
                               'No fused FPN add or expand-DW streaming', 'No route or board measurement',
                               'Five fixed-shape templates, not a single shared programmable engine',
                               'Dynamic image dimensions give unknown top HLS cycle latency',
                               'No RTL co-simulation'])
    (folder / 'operator_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    REPORTS.mkdir(parents=True, exist_ok=True)
    lines = ['# V0.2 独立算子硬件执行记录', '',
             f'- 当前 attempt：`{attempt}`；独立算子验收：`{result["acceptance"]}`。',
             f'- 实际计算通路和 AXI 宽度见证：`{result["mapping_acceptance"]}`。',
             '- 证据范围：真实 Vitis HLS 2022.2 C simulation（C 仿真）和综合估计。',
             '- 未执行 route（布局布线）或上板；没有检测器端到端时延结论。',
             '- MAC 循环改为按 256 个输出像素交错累加，避免连续 reduction 项立即读写同一 BRAM 部分和。',
             '- 请求 II=1 是源码约束；下表只报告实际 HLS 排程结果。', '',
             '| Case | Csim | Csynth | MAC loop 实际 II | 计算核乘法实例 | DSP | LUT | BRAM18K | HLS 估计时钟(ns) |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        name = row['name']; metric = row.get('hls_estimates', {})
        loops = [p for p in row.get('pipeline_results', []) if p['loop'] in ['DENSE_MAC_STEPS', 'DW_MAC_STEPS']]
        ii = loops[-1]['final_ii'] if loops else ''
        lines.append(f'| {name} | {row["csim_status"]} | {row["csynth_status"]} | {ii} | '
                     f'{row.get("compute_multiplier_instances", "")} | {metric.get("DSP", "")} | '
                     f'{metric.get("LUT", "")} | {metric.get("BRAM_18K", "")} | '
                     f'{metric.get("estimated_clock_ns", "")} |')
    lines += ['', '源码、独立 NumPy 参考输出、已上传 source manifest、日志、报告和 RTL 的 SHA-256 绑定见：',
              f'`results/detector_v0_2_execution_20261002/hardware/{attempt}/operator_summary.json`。', '',
              '每个卷积 case 含 160×160、17×19 tile 尾部、1×1、正负 INT32 溢出五组测试；prediction21 同时校验通道 padding 尾部为零。',
              '数值边界含 nearest-even（最接近偶数舍入）、INT32 模 2^32 累加、ReLU/ReLU6 与 FPN 分阶段尺度对齐。', '',
              '当前仅证明算子实现。加载/存储阶段、部分和初始化与回写、通道补齐会额外耗时；MAC loop II 不等于端到端吞吐。',
              '全图 scheduler、ping-pong/dataflow、FPN 融合、expand-DW 流式连接尚未实现。',
              'stem 的输入只有 1 个通道，编译器将 SIMD 补零路径优化掉，因此其真实乘法通路为 16；PW48/prediction21 的真实通路为 128，DW 为 32。',
              '接口宽度从生成顶层 RTL 的 C_M_AXI_ACTIVATIONS/PARAMETERS_DATA_WIDTH 读出；statistics 接口单独为 64 bit。',
              '本次运行仍使用已上传并绑定 SHA-256 的 remote_run.py；后续模板补充了缺失 case/缺失综合报告立即失败的规则，未修改本次冻结源。']
    lines += ['五个卷积均为固定通道/核/步长的独立模板见证；尚未实现覆盖整图所有形状的共享可编程引擎。',
              '图像宽高是运行时参数，顶层 HLS 报告 latency 为未知；本次没有用 MAC/频率估算冒充真实算子周期。',
              '未运行 RTL co-simulation（C/RTL 协同仿真）；数值一致性证据属于厂商 C 模型与 NumPy 独立参考。']
    (REPORTS / 'hardware_progress.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', default='operator_v2')
    args = parser.parse_args()
    print(json.dumps({k: v for k, v in collect(args.attempt).items() if k != 'rows'}, indent=2))
