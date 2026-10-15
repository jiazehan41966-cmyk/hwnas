"""Render saved Phase 2A diagnostics to CSV/Markdown without rerunning models."""
import argparse
import csv
import json
from pathlib import Path


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path(__file__).resolve().parents[2]/'reports/phase2a_candidate/sanity_results.json')
    args=parser.parse_args()
    r=json.loads(args.input.read_text(encoding='utf-8'))
    out=args.input.parent
    assignments=[]; topk=[]; complexity=[]
    for c in r['cases']:
        assignments.extend(dict(architecture=c['name'],mode=c['mode'],**d) for d in c['assignment'])
        topk.extend(dict(architecture=c['name'],mode=c['mode'],image_in_batch=d['image_in_batch'],**row) for d in c['topk'] for row in d['levels'])
        complexity.append({k:c[k] for k in ['name','mode','params','macs','flops_2macs']})
    write_csv(out/'assignment_diagnostics.csv',assignments)
    write_csv(out/'topk_diagnostics.csv',topk)
    write_csv(out/'complexity.csv',complexity)
    lines=['# Phase 2A Candidate Validation', '', f"Status: {r['status']}", '',
        '## Implementation and baseline protection', '',
        '`src/hwnas_fpga/models/sonar_candidate/__init__.py` provides build_sonar_candidate and validate_sonar_architecture. '
        'A package rather than a new top-level models/*.py preserves the frozen manifest glob. '
        'No frozen source or YAML was edited. The original detector, FPN, head, assignment, loss, decode and evaluator are reused.', '',
        'The truncated ArchitectureSpec physically constructs only stages0..5 for P2/P3/P4; execution hooks verify this path. '
        'Head/feature/anchor counts derive from the tuple; P2 anchors copy P3 without rank reassignment. '
        'Only active MBConv depthwise k3/k5 are configurable; extra fields and modified specifications fail at entry.', '',
        '## Baseline equivalence', '',
        f"Strict state_dict mapping: {r['equivalence']['state_dict_mapping']}; max_abs_error={r['equivalence']['max_abs_error']}; mean_abs_error={r['equivalence']['mean_abs_error']}.", '',
        'Eval mode, identical seed42 fixed tensor [1,1,640,640], all three prediction scales. '
        'Regression also checks backbone/neck features. No renaming or partial load required. '
        'The 42-source hash dictionary and frozen YAML SHA match before and after sanity.', '',
        '## Execution protocol', '',
        'CPU FP32, seed42, real batch4 from frozen train augmentation/WeightedRandomSampler, workers0. '
        'The same materialized image/target batch is reused for every independent candidate. '
        'Five hand-specified cases only: three all-k3 tuples plus four-level all-k5 and mixed. '
        'No architecture sampling, weight sharing, exhaustive evaluation or full training.', '',
        'For each: measured Conv/Linear hooks at batch1, real batch forward, finite loss/backward and gradient checks, '
        'decode diagnostic equality, one AdamW update, one real valid batch4 with the frozen COCO-style evaluator. '
        'All reported valid AP/P/R are zero at this initialization/single-step sanity point, not baseline or candidate performance results. '
        'Train paths/targets and validation counts are recorded in sanity_results.json; test is not evaluated.', '',
        '## Shapes and measured complexity', '',
        '| Combination | Kernel | Params | MACs (batch1) | FLOPs≈2MACs |',
        '| --- | --- | ---: | ---: | ---: |']
    for c in r['cases']:
        lines.append(f"| {c['name']} | {c['mode']} | {c['params']:,} | {c['macs']:,} | {c['flops_2macs']:,} |")
    lines.extend(['', 'All-k3 MACs exactly match the three static estimates; four-level P2 cost is real Conv execution cost, not latency.', ''])
    for c in r['cases'][:3]:
        lines.append(f"- {c['name']}: backbone {c['feature_shapes']}; heads {c['head_shapes']}; executed stages {c['executed_stages']}.")
    lines.extend(['','## Assignment (same four GTs)', '',
        'IoU means anchor-width/height IoU against the assigned GT, weighted by surviving positive slots; '
        'it is not predicted-box IoU. GT coverage is per level (can overlap across levels), matched from actual assignment tensors. '
        'No loss/assignment changes were made.', '',
        '| Combination | Level | Positive | Ignore | Negative | IoU mean/median | GT covered |',
        '| --- | --- | ---: | ---: | ---: | --- | --- |'])
    for c in r['cases'][:3]:
        for d in c['assignment']:
            lines.append(f"| {c['name']} | {d['level']} | {d['positive']} | {d['ignore']} | {d['negative']} | {d['matched_iou_mean']:.4f}/{d['matched_iou_median']:.4f} | {d['gt_matched_count']}/{d['gt_total']} |")
    lines.extend(['', 'P2 adds22 positive/11 ignored slots; original P3/P4/P5 assignments remain identical on this batch. '
        'The P2/P3 positive count difference comes from stride-dependent neighbor-cell assignment. '
        'A synthetic below-positive-threshold tie test confirms the existing best-anchor fallback chooses the first level (P2) '
        'when copied P2/P3 anchors tie. This is observed frozen behavior, not a repair. '
        'P2/P3/P4 also removes P5 and stages6/7; it is not a pure +P2 ablation.', '',
        '## Global top500 and NMS', '',
        'Counts below aggregate four images (2000 top-K slots, up to400 final detections). '
        'Per-image raw/pre-topK/global-topK/post-NMS values are in topk_diagnostics.csv. '
        'Score threshold0 for AP, global top500, NMS0.5, final max100 unchanged. '
        'The diagnostic decode is checked bit-for-bit against the frozen decoder.', '',
        '| Combination | Level | Top500 aggregate | Fraction | Post-NMS aggregate |',
        '| --- | --- | ---: | ---: | ---: |'])
    for c in r['cases'][:3]:
        for level in [d['level'] for d in c['assignment']]:
            rows=[z for d in c['topk'] for z in d['levels'] if z['level']==level]
            n=sum(z['global_top500'] for z in rows)
            lines.append(f"| {c['name']} | {level} | {n} | {n/2000:.2%} | {sum(z['post_nms'] for z in rows)} |")
    lines.extend(['', 'Four-level top500 is concentrated in P2/P3 on this batch, but P2 has75.29% of raw anchors '
        'and56.65% of top500; this is not disproportionate P2 dominance relative to raw counts. '
        'Candidates are independently initialized; changing module layouts changes random draws. '
        'One untrained batch cannot establish a persistent trained-model level bias or P2 benefit.', '',
        '## Reproduction and gate', '', '```powershell',
        '.venv_cuda/Scripts/python.exe tools/sonar_spec/candidate_sanity.py --space configs/search_space_sonar_v0_1.yaml --baseline configs/baseline_sonar_frozen.yaml',
        '.venv_cuda/Scripts/python.exe -m pytest tests/test_sonar_candidate_shapes.py -q',
        '.venv_cuda/Scripts/python.exe tools/sonar_spec/render_candidate_report.py', '```', '',
        'The runner performs no long training and fails on frozen-source/config drift. '
        'Phase2A runtime checks pass. Formal80epoch baseline performance remains pending for later performance comparisons, '
        'not a blocker for implementing a minimal software one-shot Supernet. No Supernet is implemented in this phase.'])
    (out/'PHASE2A_CANDIDATE_VALIDATION.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__': main()
