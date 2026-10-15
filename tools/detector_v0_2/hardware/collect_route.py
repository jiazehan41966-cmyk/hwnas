"""Bind the PW48 OOC route reports and retain clock/board evidence boundaries."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'results/detector_v0_2_execution_20261002/hardware/route_pw48_v1'


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> dict:
    evidence = BASE / 'remote_evidence'
    state = json.loads((evidence / 'route_state.json').read_text())
    manifest = json.loads((BASE / 'route_source_manifest.json').read_text())
    constraints_path = BASE / 'route_constraints.json'
    constraints = json.loads(constraints_path.read_text()) if constraints_path.is_file() else dict(
        clock_period_ns=5.333, setup_uncertainty_ns=1.440, hold_uncertainty_ns=1.440,
        hold_source='v1 unintentionally applied HLS setup margin to hold as well')
    source_bound = all((BASE / p).is_file() and sha(BASE / p) == s for p, s in manifest.items())
    source_bound &= state['source_manifest_sha256'] == sha(BASE / 'route_source_manifest.json')
    hls_rtl = BASE.parent / 'operator_v2/remote_evidence/pw48/project/solution1/syn/verilog'
    rtl_files=list((BASE/'rtl').glob('*.v'))
    hls_bound = bool(rtl_files) and all((hls_rtl / p.name).is_file() and sha(hls_rtl / p.name) == sha(p)
                                      for p in rtl_files)
    metrics_path = evidence / 'route_metrics.txt'
    metrics = dict(line.split('=', 1) for line in metrics_path.read_text().splitlines()) if metrics_path.is_file() else {}
    slack_text = metrics.get('worst_constrained_setup_slack_ns')
    slack = None if slack_text in [None, 'NO_CONSTRAINED_PATH'] else float(slack_text)
    utilization_path = evidence / 'route_utilization.rpt'
    utilization = {}
    if utilization_path.is_file():
        report = utilization_path.read_text(errors='replace')
        for label, key in [('CLB LUTs', 'LUT'), ('CLB Registers', 'FF'),
                           ('DSPs', 'DSP'), ('Block RAM Tile', 'BRAM36_equivalent'), ('URAM', 'URAM')]:
            m = re.search(r'^\|\s*' + re.escape(label) + r'\s*\|\s*([\d.]+)\s*\|', report, re.MULTILINE)
            utilization[key] = float(m.group(1)) if m else None
    timing_path = evidence / 'route_timing_summary.rpt'
    timing_text = timing_path.read_text(errors='replace') if timing_path.is_file() else ''
    # Vivado's Design Timing Summary includes both setup and hold columns.
    # Restrict this parse to its summary instead of unrelated path tables.
    summary = timing_text.split('Design Timing Summary', 1)[-1].split('Clock Summary', 1)[0]
    summary_row = re.search(r'^\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+\d+\s+\d+\s+'
                            r'(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+\d+\s+\d+',
                            summary, re.MULTILINE)
    hold_slack = float(summary_row.group(3)) if summary_row else None
    route_status_path = evidence / 'route_status.rpt'
    route_log_path = evidence / 'route.log'
    route_log = route_log_path.read_text(errors='replace') if route_log_path.is_file() else ''
    intermediate = [dict(setup_slack_ns=float(a), total_setup_slack_ns=float(b),
                         hold_slack_ns=float(c), total_hold_slack_ns=float(d))
                    for a,b,c,d in re.findall(r'WNS=(-?\d+\.\d+)\s*\|\s*TNS=(-?\d+\.\d+)\s*\|\s*WHS=(-?\d+\.\d+)\s*\|\s*THS=(-?\d+\.\d+)', route_log)]
    stop_path = evidence/'stop_record.json'
    route_text = route_status_path.read_text(errors='replace') if route_status_path.is_file() else ''
    # The complete route report is retained, even if this tool version uses a
    # different success text. An unrecognized report never passes the gate.
    fully_routed = bool(re.search(r'(?:#\s+of\s+unrouted\s+nets\s*:?\s*0\b|Number\s+of\s+Unrouted\s+Nets\s*:?\s*0\b|All nets are completely routed)',
                                route_text, re.IGNORECASE))
    routed = bool(source_bound and hls_bound and state.get('returncode') == 0 and
                  state.get('status') == 'ROUTED' and metrics_path.is_file() and timing_path.is_file())
    result = dict(scope='PW48 isolated out-of-context route; intra-clock paths only',
                  attempt=BASE.name, constraints=constraints,
                  source_bound=bool(source_bound), hls_rtl_bound=bool(hls_bound),
                  state=state, route_completed=routed, fully_routed_report=fully_routed,
                  clock_period_ns=5.333, clock_uncertainty_ns=1.440,
                  setup_uncertainty_ns=constraints['setup_uncertainty_ns'],
                  hold_uncertainty_ns=constraints['hold_uncertainty_ns'],
                  worst_constrained_setup_slack_ns=slack,
                  worst_hold_slack_ns=hold_slack,
                  clock_gate='PASS' if routed and fully_routed and slack is not None and slack >= 0
                              and hold_slack is not None and hold_slack >= 0 else 'NOT_PASSED',
                  route_utilization=utilization,
                  intermediate_timing_not_final=intermediate,
                  stop_record=json.loads(stop_path.read_text()) if stop_path.is_file() else None,
                  reports={p.name: dict(path=str(p.relative_to(ROOT)), sha256=sha(p))
                           for p in [metrics_path, utilization_path, timing_path, route_status_path,
                                     route_log_path, stop_path, evidence/'route_state.json', BASE/'route.tcl'] if p.is_file()},
                  limitations=['No IO input/output delays: external AXI paths remain unverified',
                               'No full-graph scheduler, PS interconnect, DDR or board measurement',
                               'No bitstream generated and no board programming'])
    (BASE / 'route_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    report_path = ROOT / 'reports/detector_v0_2_execution_20261002/hardware_progress.md'
    text = report_path.read_text(encoding='utf-8').split('\n## PW48 独立 OOC route')[0]
    text = text.replace('- 未执行 route（布局布线）或上板；没有检测器端到端时延结论。',
                        '- 算子 campaign 不包含 route（布局布线）；PW48 OOC 结果另列。未上板，没有检测器端到端时延结论。')
    text += '\n## PW48 独立 OOC route\n\n'
    for p in sorted(BASE.parent.glob('route_pw48_v*/route_summary.json')):
        r = json.loads(p.read_text()); c = r['constraints']
        text += (f'- `{r["attempt"]}`：完成 `{r["route_completed"]}`；时钟门槛 `{r["clock_gate"]}`；'
                 f'setup slack `{r["worst_constrained_setup_slack_ns"]}` ns；hold slack `{r["worst_hold_slack_ns"]}` ns。\n'
                 f'  clock period `{c["clock_period_ns"]}` ns；setup uncertainty `{c["setup_uncertainty_ns"]}` ns；'
                 f'hold uncertainty `{c["hold_uncertainty_ns"]}` ns。\n'
                 f'  route 资源：`{json.dumps(r["route_utilization"])}`。\n')
    text += ('\n1.440 ns setup uncertainty 来自原 HLS 报告；v1 同时将其施加到 hold，产生人为的高 hold 约束，已保留证据并仅终止该独立进程。\n'
             'v2 分开指定 setup=1.440 ns、hold=0.050 ns；后者为明确的 OOC 假设，尚未绑定板端时钟抖动。\n'
             '仅核验该独立核内部时序；AXI 外部端口未设置 IO delay，全图/PS/DDR 时序尚未核验。\n'
             '未生成 bitstream，未烧写 FPGA，没有检测器板端时延结论。每次独立 attempt 的 route_summary.json 保留原始报告路径和 SHA-256。\n')
    report_path.write_text(text, encoding='utf-8')
    return result


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--attempt',default='route_pw48_v2')
    a=p.parse_args()
    if not re.match(r'^route_pw48_v[1-9][0-9]*$',a.attempt):raise ValueError('invalid route attempt')
    BASE=BASE.parent/a.attempt
    print(json.dumps(main(), indent=2))
