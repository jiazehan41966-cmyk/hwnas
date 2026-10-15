"""Generate preregistered Phase2F multi-seed analyses and candidate freeze."""
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from formal_protocol import ROOT, dump
from phase2f_runner import ARCHITECTURES, MANIFEST, METRICS, REPORT, load_manifest, result_dir


LABEL = {aid: label for label, aid, _ in ARCHITECTURES}


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'No rows for {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def collect():
    manifest, _ = load_manifest(False)
    rows = []
    for row in manifest['architectures']:
        rows.append(dict(label=row['label'], architecture_id=row['architecture_id'], seed=42,
            best_epoch=row['best_epoch'], **{k: row[k] for k in METRICS},
            endpoint_mAP50_95=row['endpoint_mAP50_95'], Params=row['Params'], MACs=row['MACs'],
            training_time_seconds='', best_checkpoint='', latest_checkpoint='',
            best_checkpoint_sha256=row['checkpoint_sha256'], latest_checkpoint_sha256='',
            history_sha256=row['history_sha256'], provenance='phase2e_frozen_seed42', test_evaluated=False))
    for _, aid, _ in ARCHITECTURES:
        for seed in (43, 44):
            path = result_dir(aid, seed) / 'result.json'
            if not path.exists():
                raise FileNotFoundError(f'Missing completed Phase2F result: {path}')
            row = json.loads(path.read_text(encoding='utf-8'))
            rows.append(dict(label=LABEL[aid], **row, provenance='phase2f_independent_from_scratch'))
    rows.sort(key=lambda r: (r['label'], int(r['seed'])))
    if len(rows) != 18 or any(r['test_evaluated'] not in (False, 'False') for r in rows):
        raise ValueError('Expected 18 sealed multi-seed rows')
    return rows


def aggregate(rows):
    result = []
    for _, aid, _ in ARCHITECTURES:
        group = [r for r in rows if r['architecture_id'] == aid]
        values = [float(r['mAP50_95']) for r in group]
        endpoints = [float(r['endpoint_mAP50_95']) for r in group]
        result.append(dict(label=LABEL[aid], architecture_id=aid, n=3,
            mean_mAP50_95=mean(values), std_mAP50_95=stdev(values), min_mAP50_95=min(values),
            max_mAP50_95=max(values), endpoint_mean_mAP50_95=mean(endpoints),
            mean_peak_minus_endpoint=mean(v-e for v, e in zip(values, endpoints)),
            Params=int(group[0]['Params']), MACs=int(group[0]['MACs'])))
    return result


def paired(rows, left, right, path):
    result = []
    for seed in (42, 43, 44):
        a = next(r for r in rows if r['label'] == left and int(r['seed']) == seed)
        b = next(r for r in rows if r['label'] == right and int(r['seed']) == seed)
        result.append(dict(seed=seed, left=left, right=right,
            delta_mAP50_95=float(a['mAP50_95'])-float(b['mAP50_95']),
            delta_AP50=float(a['AP50'])-float(b['AP50']),
            delta_AP75=float(a['AP75'])-float(b['AP75']),
            delta_Recall=float(a['Recall'])-float(b['Recall'])))
    summary = dict(seed='mean', left=left, right=right)
    spread = dict(seed='sample_std', left=left, right=right)
    for metric in ('delta_mAP50_95', 'delta_AP50', 'delta_AP75', 'delta_Recall'):
        vals = [r[metric] for r in result]
        summary[metric] = mean(vals); spread[metric] = stdev(vals)
    write_csv(path, result + [summary, spread])
    return result, summary, spread


def pareto(agg):
    front = []
    for row in agg:
        dominated = any(other['mean_mAP50_95'] >= row['mean_mAP50_95'] and other['MACs'] <= row['MACs']
            and (other['mean_mAP50_95'] > row['mean_mAP50_95'] or other['MACs'] < row['MACs'])
            for other in agg)
        if not dominated:
            front.append(row)
    front.sort(key=lambda r: (-r['mean_mAP50_95'], r['MACs'], r['architecture_id']))
    selected = front[:4]
    if len(selected) < 2:
        for row in sorted(agg, key=lambda r: (-r['mean_mAP50_95'], r['MACs'], r['architecture_id'])):
            if row not in selected:
                selected.append(row)
            if len(selected) == 2:
                break
    return selected


def figures(rows, agg, stage7, p2):
    out = REPORT / 'figures'; out.mkdir(parents=True, exist_ok=True)
    labels = [x[0] for x in ARCHITECTURES]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, label in enumerate(labels):
        group = [r for r in rows if r['label'] == label]
        ax.scatter([i-.15, i, i+.15], [r['mAP50_95'] for r in group], label=label)
    ax.set_xticks(range(len(labels)), labels); ax.set_ylabel('best valid mAP50:95'); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(out/'architecture_seed_map.png', dpi=180); fig.savefig(out/'architecture_seed_map.svg'); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(range(len(agg)), [r['mean_mAP50_95'] for r in agg],
        yerr=[r['std_mAP50_95'] for r in agg], fmt='o', capsize=4)
    ax.set_xticks(range(len(agg)), [r['label'] for r in agg]); ax.set_ylabel('mAP50:95 mean +/- sample std'); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(out/'mean_std.png', dpi=180); fig.savefig(out/'mean_std.svg'); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for r in rows:
        ax.scatter(float(r['endpoint_mAP50_95']), float(r['mAP50_95']), label=f"{r['label']}-{r['seed']}")
    lo = min(min(float(r['endpoint_mAP50_95']), float(r['mAP50_95'])) for r in rows)
    hi = max(max(float(r['endpoint_mAP50_95']), float(r['mAP50_95'])) for r in rows)
    ax.plot([lo, hi], [lo, hi], '--', color='gray'); ax.set_xlabel('epoch80 endpoint'); ax.set_ylabel('best'); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(out/'best_vs_endpoint.png', dpi=180); fig.savefig(out/'best_vs_endpoint.svg'); plt.close(fig)

    for name, paired_rows, ylabel in [('stage7_paired_difference', stage7, 'A1 - A2 mAP50:95'), ('p2_paired_difference', p2, 'B1 - B2 mAP50:95')]:
        fig, ax = plt.subplots(figsize=(5, 4)); vals=[r['delta_mAP50_95'] for r in paired_rows]
        ax.bar([str(r['seed']) for r in paired_rows], vals); ax.axhline(0, color='black', linewidth=.8)
        ax.set_xlabel('seed'); ax.set_ylabel(ylabel); fig.tight_layout(); fig.savefig(out/f'{name}.png', dpi=180); fig.savefig(out/f'{name}.svg'); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in agg:
        ax.scatter(r['MACs']/1e9, r['mean_mAP50_95']); ax.annotate(r['label'], (r['MACs']/1e9, r['mean_mAP50_95']))
    ax.set_xlabel('MACs (G)'); ax.set_ylabel('3-seed mean valid mAP50:95'); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(out/'accuracy_vs_macs_pareto.png', dpi=180); fig.savefig(out/'accuracy_vs_macs_pareto.svg'); plt.close(fig)


def main():
    rows = collect(); write_csv(REPORT/'multiseed_results.csv', rows)
    agg = aggregate(rows)
    stability = [dict(label=r['label'], architecture_id=r['architecture_id'], seed=r['seed'],
        best_epoch=r['best_epoch'], best_mAP50_95=r['mAP50_95'], endpoint_mAP50_95=r['endpoint_mAP50_95'],
        peak_minus_endpoint=float(r['mAP50_95'])-float(r['endpoint_mAP50_95'])) for r in rows]
    write_csv(REPORT/'peak_endpoint_stability.csv', stability)
    s7, s7mean, s7std = paired(rows, 'A1', 'A2', REPORT/'matched_stage7_analysis.csv')
    p2, p2mean, p2std = paired(rows, 'B1', 'B2', REPORT/'matched_p2_analysis.csv')
    s7d=[r['delta_mAP50_95'] for r in s7]; p2d=[r['delta_mAP50_95'] for r in p2]
    stage7_state=('REPLICATED_STAGE7_K5_SIGNAL' if all(x>0 for x in s7d) and mean(s7d)>stdev(s7d)
        else 'UNCERTAIN_STAGE7_SIGNAL' if mean(s7d)>0 else 'STAGE7_K5_NOT_REPLICATED')
    p2_state=('REPLICATED_POSITIVE' if all(x>0 for x in p2d) and mean(p2d)>stdev(p2d)
        else 'CONTEXT_DEPENDENT / UNCERTAIN' if mean(p2d)>0 else 'NOT_SUPPORTED')
    selected=pareto(agg)
    dump(REPORT/'network_candidate_set.json', dict(version='phase2f_network_candidate_set_v1',
        selection='accuracy_compute_pareto_over_three_seed_mean_capped_4_floor_2',
        architectures=selected, test_evaluated=False,
        caveat='selected promoted population; MACs are not FPGA latency'))
    candidate_lines='\n'.join(f"- {r['label']}: `{r['architecture_id']}`; mean={r['mean_mAP50_95']:.6f}, sample std={r['std_mAP50_95']:.6f}, endpoint mean={r['endpoint_mean_mAP50_95']:.6f}, MACs={r['MACs']}" for r in selected)
    (REPORT/'NETWORK_CANDIDATE_SET.md').write_text('# Network Candidate Set\n\n'+candidate_lines+
        '\n\nThis set is the input to future Q(x) and FPGA mapping evaluation. MACs are software compute estimates, not FPGA latency. Test remains sealed.\n', encoding='utf-8')
    gate='YES' if stage7_state=='REPLICATED_STAGE7_K5_SIGNAL' else ('CONDITIONAL' if stage7_state=='UNCERTAIN_STAGE7_SIGNAL' else 'NO')
    (REPORT/'SONAROP_GATE.md').write_text(f'# SonarOp Gate\n\nGate: **{gate}**\n\nStage7 result: `{stage7_state}`.\n\nPaired deltas: '+', '.join(f'{x:.6f}' for x in s7d)+f'; mean={mean(s7d):.6f}, sample std={stdev(s7d):.6f}.\n\nNo SonarOp is implemented in Phase 2F.\n', encoding='utf-8')
    table='\n'.join(f"| {r['label']} | {r['mean_mAP50_95']:.6f} | {r['std_mAP50_95']:.6f} | {r['min_mAP50_95']:.6f} | {r['max_mAP50_95']:.6f} | {r['endpoint_mean_mAP50_95']:.6f} | {r['MACs']} |" for r in agg)
    (REPORT/'MULTISEED_CONFIRMATION.md').write_text(f'''# Phase 2F Multi-seed Confirmation

All six preregistered architectures have seed42/43/44 results. Seeds 43/44 are independent from-scratch runs. Test was not evaluated.

| Label | Mean mAP50:95 | Sample std | Min | Max | Endpoint mean | MACs |
|---|---:|---:|---:|---:|---:|---:|
{table}

## Stage7 matched result

State: `{stage7_state}`. Deltas by seed: {', '.join(f'{x:.6f}' for x in s7d)}; mean {mean(s7d):.6f}, sample std {stdev(s7d):.6f}.

## P2 matched result

State: `{p2_state}`. Deltas by seed: {', '.join(f'{x:.6f}' for x in p2d)}; mean {mean(p2d):.6f}, sample std {stdev(p2d):.6f}. This is evidence only for the preregistered B1/B2 kernel context.

## Limitations

Results concern the selected promoted population, one official split, a 99-image validation set, and n=3 seeds. The test split remains sealed. The Phase 2E FINAL 72 are conditioned on promotion and do not prove a global optimum over all 144 architectures. P234 only failed the frozen 40-epoch promotion rule; it was not proven inferior at 80 epochs.
''', encoding='utf-8')
    figures(rows,agg,s7,p2)


if __name__=='__main__':
    main()
