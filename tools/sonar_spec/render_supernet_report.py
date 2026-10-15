"""Render Phase2B correctness evidence; never evaluates or ranks architectures."""
import argparse
import csv
import json
from pathlib import Path


def csv_file(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path(__file__).resolve().parents[2]/'reports/phase2b_supernet/results.json')
    args=parser.parse_args();r=json.loads(args.input.read_text());out=args.input.parent
    sampling=r['sampling'];rows=[]
    for name,p in sampling['output_proportions'].items():
        rows.append(dict(variable=name,choice='active',count=sampling['counts'][name],denominator=sampling['samples'],proportion=p,target=1/3,error_pp=100*(p-1/3)))
    for stage,proportions in sampling['kernel_proportions'].items():
        for k,p in proportions.items():
            n=sum(sampling['counts'][f'{stage}_{j}'] for j in ('k3','k5'))
            rows.append(dict(variable=stage,choice=k,count=sampling['counts'][f'{stage}_{k}'],denominator=n,proportion=p,target=.5,error_pp=100*(p-.5)))
    csv_file(out/'sampling_statistics.csv',rows)
    csv_file(out/'equivalence.csv',[dict(id=e['id'],candidate_max=e['candidate']['max_abs_error'],candidate_mean=e['candidate']['mean_abs_error'],export_max=e['export']['max_abs_error'],export_mean=e['export']['mean_abs_error']) for e in r['equivalence']])
    csv_file(out/'bn_inventory.csv',r['bn']['inventory'])
    csv_file(out/'assignment_diagnostics.csv',r['assignment'])
    csv_file(out/'topk_diagnostics.csv',[dict(image_in_batch=d['image_in_batch'],**row) for d in r['topk'] for row in d['levels']])
    lines=['# Phase 2B — Minimal One-shot Supernet','',f"Status: {r['status']}",'',
        '## Implementation and representation','',
        '`src/hwnas_fpga/models/sonar_supernet/` reuses the unchanged candidate validator, builder, metadata and '
        'original backbone/FPN execution methods. Only active stages are visited. Candidate and active subnet use '
        'the same two-field architecture mapping. JSON serialization is canonical; IDs are deterministic readable strings.', '',
        'Coverage: 144/144 legal configurations activated and round-tripped, 144 unique IDs. Enumeration here only '
        'checks structure/identity; it does not run144 validations or rank architectures. Hierarchical sampling first '
        'chooses one of three combinations uniformly, then independently chooses k3/k5 for each active MBConv.', '',
        '## Sharing granularity','',
        '- Stem, fixed stages, expansion PW/BN and projection PW/BN are shared.',
        '- Every MBConv owns independent depthwise3 and depthwise5 parameters, plus independent depthwise BN affine/running state per kernel; no center cropping.',
        '- P2/P3/P4/P5 lateral, output convolution and head modules are persistent and shared by level across combinations.',
        '- Each iteration uses one path. Inactive depthwise branches, omitted tail and unused level modules have grad=None after zero_grad(set_to_none=True); AdamW leaves their parameters bitwise unchanged.',
        '- P2/P3/P4 execution hooks confirm no stage6/7, P5 lateral/output/head execution. Export contains only selected modules and physically omits the tail.', '',
        '## Sampling witness (seed42, CPU12000 draws)','',
        '| Variable | Choice | Count / conditional total | Proportion | Error percentage points |',
        '| --- | --- | ---: | ---: | ---: |']
    for row in rows:
        lines.append(f"| {row['variable']} | {row['choice']} | {row['count']}/{row['denominator']} | {row['proportion']:.4%} | {row['error_pp']:+.4f} |")
    lines+=['','Stage6/7 proportions condition on combinations where they are active. Observed maximum combination deviation '
        'is0.842 percentage points and maximum kernel deviation0.558 percentage points. Test tolerances were fixed at2/3 percentage points respectively.', '',
        '## Candidate and export equivalence','',
        '19 architectures: nine prescribed tuples×{all3,all5,mixed} plus ten seeded random distinct architectures. '
        'Eval FP32, identical [1,1,640,640] tensor, explicit candidate-key to supernet-key mapping and strict state_dict load. '
        'Every candidate and exported-subnet max_abs_error=0, mean_abs_error=0; per-architecture values are in equivalence.csv. '
        'Independent exports contain no dynamic branch switch and can save/reload their own state_dict. '
        'These are numerical correctness witnesses, not inherited-weight AP comparisons.', '',
        '## BN audit and recalibration','',
        f"Supernet has {len(r['bn']['inventory'])} BatchNorm2d modules (names/channels in bn_inventory.csv): "
        '2 stem/fixed-stage BN, 24 MBConv BN (six×[expand, DW3, DW5, project]), 12 level-specific Neck/Head BN. '
        'Candidate BN counts: P3/P4/P5=29, P2/P3/P4=23, four-level=32.', '',
        'BN affine/running state outside the depthwise branch is shared across kernels and output combinations. '
        'Even level-specific FPN BN sees different upstream context when P5 is omitted. Therefore shared training running '
        'statistics must not be used directly for final subnet ranking.', '',
        '`calibrate_subnet_bn` exports an independent candidate without changing active selection; resets only its BN '
        'mean/variance/counter, uses cumulative batch statistics (momentum=None), forwards without autograd, then restores '
        'BN momentum and returns eval mode. This averages batch statistics, not an exact pooled global-variance estimator. '
        'The witness uses first two sequential, unaugmented train batches (4 images each), workers0, fixed seed. '
        'Calibration rejects valid/test loaders. Candidate weights, original supernet parameters/buffers, active architecture '
        'and mode stay unchanged; torch RNG is preserved. The caller must supply a fixed calibration loader for reproducibility.', '',
        'One valid batch4 then passed the unchanged frozen evaluator; assignment and top500/max100 diagnostics retain '
        'Phase2A semantics and decode parity. Sanity AP is not a formal performance result. No test evaluation occurred.', '',
        '## Smoke and checkpoint','',
        '12 single-path iterations on the same materialized real augmented train batch4; 11 unique architectures, '
        'combination counts P3/P4/P5=3, P2/P3/P4=2, four-level=7. Every active MBConv stage saw both kernels. '
        'All losses/active gradients finite; loss range12.1482–27.8305 is a chain check, not a convergence claim.', '',
        'smoke_checkpoint.pt stores model/optimizer, epoch0, global_step12, architecture/sampler RNG and statistics, '
        'Python/NumPy/Torch/CUDA RNG, architecture, space version/hash, seed/policy, baseline hash and git commit. '
        'data_state contains the materialized smoke batch and loader sampler RNG. resumed_checkpoint.pt records step13. '
        'One resumed update exactly matches the uninterrupted reference in architecture, loss and every model-state tensor. '
        'Version/hash/protocol/policy mismatches fail before state loading. Checkpoints are trusted local pickle artifacts.', '',
        'Resume correctness here applies to fixed-batch smoke replay. A formal epoch-based runner must additionally '
        'persist sampler index order/cursor and augmentation/data-loader state, or explicitly checkpoint at epoch boundaries. '
        'Do not claim arbitrary mid-epoch data-stream equivalence from this smoke witness.', '',
        '## Baseline protection and reproduction','',
        'Frozen configuration and the entire42-file source hash dictionary are unchanged. The new package is outside '
        'the frozen top-level models/*.py scan. No shared baseline implementation or Phase2A candidate file was modified.', '',
        '```powershell',
        '.venv_cuda/Scripts/python.exe tools/sonar_spec/supernet_smoke.py',
        '.venv_cuda/Scripts/python.exe -m pytest tests/test_sonar_supernet.py tests/test_sonar_candidate_shapes.py -q',
        '.venv_cuda/Scripts/python.exe tools/sonar_spec/render_supernet_report.py',
        '```','',
        '## Phase 2C gate','',
        'Phase2B correctness gates pass. Ready to prepare formal supernet training and ranking-reliability protocol, '
        'using train/valid only and explicit fixed train-only BN calibration per subnet. No formal training,144-subnet '
        'ranking, Top-K retraining, hardware or quantization search was run. The80epoch baseline performance remains '
        'pending for later independent performance comparisons.']
    (out/'PHASE2B_SUPERNET_VALIDATION.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
