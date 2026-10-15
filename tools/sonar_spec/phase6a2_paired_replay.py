"""Offline, paired continuation replay. Never imports a dataset or trains a detector.

Reuses Phase6A.1 parsers, encoding and statistics; leaves its results untouched.
Run --prepare to bind the protocol before --run. All labels are validation labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata, spearmanr, kendalltau, pearsonr
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase6a1_surrogate_diagnosis as prior

ROOT = prior.ROOT
DEFAULT_OUT = ROOT / "reports/phase6a2_paired_replay"
METHODS = ["random", "direct40", "direct60"] + [
    f"{family}_{mode}" for mode in ("arch", "mf40", "mf60")
    for family in ("ridge", "et", "pairwise")]
SEEDS = list(range(61000, 61050))
CHECKS = (32, 44, 56)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prior.conv(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.12g")


def inputs():
    paths = [prior.FINAL, prior.STAGE1, prior.STAGE2, prior.PHASE2F, prior.SPEC,
             prior.CURVES, ROOT / "configs/baseline_sonar_frozen.yaml",
             ROOT / "reports/baseline_sonar_frozen/freeze_manifest.json",
             ROOT / "reports/phase2e_multifidelity/MULTIFIDELITY_PROTOCOL.yaml",
             Path(prior.__file__), Path(__file__)]
    return {str(p.relative_to(ROOT)).replace("\\", "/"): prior.sha256(p) for p in paths}


def prepare(out):
    protocol = dict(
        version="phase6a2_paired_continuation_replay_v1", input_sha256=inputs(),
        python=sys.version, numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
        population="72 promoted finalists; 49 P345, 23 P2345; no P234 ground truth",
        label="best valid mAP50:95 within 80 epochs, seed42; endpoint analyzed separately",
        initial_size=20, batch_size=6, checkpoints=list(CHECKS),
        development_seeds=SEEDS[:20], verification_seeds=SEEDS[20:], methods=METHODS,
        initial_policy="macro-stratified, identical initial20 and random continuation within each seed",
        model_parameters=dict(ridge_alpha=1.0, et_trees=64, et_min_leaf=2, et_max_features=1.0,
                              pairwise_C=1.0, pairwise_fit_intercept=False),
        feature_policy="arch bits + Params/MACs; MF40 budget<=40 summaries; MF60 budget<=60 summaries",
        missing_features="no imputed late-epoch slope/loss; full curves available only for original 12 witnesses",
        prediction="ridge/ET MF predict residual above same-budget best; pairwise predicts preference, no calibrated mAP",
        acquisition="accuracy-only descending score; ties architecture_id ascending; no uncertainty or EHVI",
        development_selection="one learned method per arch/mf40/mf60, lowest mean regret across 32/44/56; tie highest mean top5 then method name",
        verification="same 72 labels, new initialization seeds, NOT an independent architecture/dataset validation",
        conditional_gate="selected method mean regret reduction vs paired random has simultaneous bootstrap lower CI>0; mean Top5 difference>=0; MF must also improve over direct same-fidelity sorting",
        bootstrap=5000, familywise_comparisons=5,
        ci="ordinary 95% reported; gate uses Bonferroni 99% intervals for up to 5 selected comparisons; seed is resampling unit",
        noise_policy="Phase2F six selected architectures x three seeds descriptive only, not a hard gain threshold or population noise ceiling",
        cost_policy="arch cost=80*n; MF b cost=72*b+(80-b)*n; additionally disclose historical 144*40+108*20 sunk cost",
        pareto="accuracy-compute diagnostics only; HV reference accuracy=0, MACs=10e9; accuracy scale=0.1",
        gate_scope="conditional on promoted72 seed42; no automatic new training; do not claim full144 superiority",
        test_evaluated=False, detector_training_launched=False)
    path = out / "PROTOCOL.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != protocol:
        raise ValueError("Existing protocol differs; use a new output directory, do not silently overwrite.")
    dump(path, protocol)
    return protocol


def audit_data(data):
    for path, n, budget in [(prior.STAGE1, 144, 40), (prior.STAGE2, 108, 60), (prior.FINAL, 72, 80)]:
        rows = prior.read_csv(path)
        assert len(rows) == n and len({r['architecture_id'] for r in rows}) == n
        for r in rows:
            assert int(r['budget']) == budget and int(r['seed']) == 42
            assert int(r['best_epoch']) <= budget
            assert r['baseline_protocol_hash'] == prior.sha256(ROOT / 'configs/baseline_sonar_frozen.yaml')
            assert float(r['mAP50_95']) + 1e-12 >= float(r['endpoint_mAP50_95'])
    for aid in data['pool_ids']:
        assert float(data['stage1'][aid]['mAP50_95']) <= float(data['stage2'][aid]['mAP50_95']) + 1e-12
        assert float(data['stage2'][aid]['mAP50_95']) <= float(data['final'][aid]['mAP50_95']) + 1e-12
    rows = data['seeds']
    assert len(rows) == 18 and len({(r['architecture_id'], r['seed']) for r in rows}) == 18
    for r in rows:
        assert r['test_evaluated'].lower() == 'false'
        if int(r['seed']) == 42:
            assert abs(float(r['mAP50_95']) - float(data['final'][r['architecture_id']]['mAP50_95'])) < 1e-12
    expected = yaml.safe_load((ROOT / 'reports/phase2e_multifidelity/MULTIFIDELITY_PROTOCOL.yaml').read_text())
    assert prior.sha256(prior.SPEC) == expected['search_space_sha256']
    assert prior.sha256(ROOT / 'configs/baseline_sonar_frozen.yaml') == expected['baseline_sha256']


def make_features(data, mode):
    # Explicit whitelist: no final80 fields, future checkpoint hashes, or future elapsed time.
    X = np.column_stack([data['X_arch'], data['params'], data['macs']])
    names = data['minimal_names'] + ['Params', 'MACs']
    base = np.zeros(len(X))
    for budget in ([40] if mode == 'mf40' else [40, 60] if mode == 'mf60' else []):
        table = data['stage1'] if budget == 40 else data['stage2']
        keys = ['mAP50_95', 'endpoint_mAP50_95', 'AP50', 'AP75', 'Precision', 'Recall', 'best_epoch']
        extra = [[float(table[aid][k]) for k in keys] for aid in data['pool_ids']]
        X = np.column_stack([X, extra])
        names += [f'{k}_budget{budget}' for k in keys]
        base = np.asarray([float(table[a]['mAP50_95']) for a in data['pool_ids']])
    if mode == 'mf60':
        X = np.column_stack([X, (data['map60'] - data['map40']) / 20])
        names += ['best_score_change_40_to_60_per_epoch_not_last_epoch_slope']
    assert np.isfinite(X).all()
    return X, base, names


def predict(family, X, base, train, y_train, remaining, seed):
    # Caller cannot provide hidden labels: only y_train is accepted.
    if len(train) != len(y_train) or set(train) & set(remaining):
        raise ValueError('Architecture train/holdout overlap or target count mismatch')
    if family == 'ridge':
        return base[remaining] + prior.fit_ridge(X[train], y_train - base[train], X[remaining], alpha=1.0)
    if family == 'et':
        model = ExtraTreesRegressor(n_estimators=64, min_samples_leaf=2, max_features=1.0,
                                   random_state=seed, n_jobs=1)
        model.fit(X[train], y_train - base[train])
        return base[remaining] + model.predict(X[remaining])
    if family == 'pairwise':
        mu, sd = prior.standardize_fit(X[train])
        z, zc = (X[train]-mu)/sd, (X[remaining]-mu)/sd
        i, j = np.triu_indices(len(train), 1)
        mask = y_train[i] != y_train[j]
        delta, labels = z[i[mask]]-z[j[mask]], (y_train[i[mask]]>y_train[j[mask]]).astype(int)
        if not len(labels):
            return np.zeros(len(remaining))
        # Both pair orders, zero intercept; avoids dependence on row/id ordering.
        model = LogisticRegression(C=1., fit_intercept=False, max_iter=1000, solver='lbfgs')
        model.fit(np.concatenate([delta, -delta]), np.concatenate([labels, 1-labels]))
        return zc @ model.coef_[0]
    raise ValueError(family)


def pareto_indices(y, costs):
    return np.array([i for i in range(len(y)) if not np.any(
        (y >= y[i]) & (costs <= costs[i]) & ((y > y[i]) | (costs < costs[i])))], dtype=int)


def hypervolume(y, costs):
    # Union of rectangles: lower cost and higher accuracy preferred.
    order = np.argsort(costs, kind='stable')
    best, area = 0., 0.
    for i in order:
        if y[i] > best:
            area += (float(y[i])-best)/0.1 * max(0., 1-float(costs[i])/1e10)
            best = float(y[i])
    return area


def score_metrics(y, score):
    return dict(spearman=prior.safe_spearman(y, score), kendall=prior.safe_kendall(y, score),
                top3_recall=prior.topk_recall(y, score, 3), top5_recall=prior.topk_recall(y, score, 5),
                ndcg10=prior.ndcg(y, score, 10), rank_mae=float(np.abs(rankdata(-y)-rankdata(-score)).mean()))


def replay(data, method, seed, initial, features):
    y, ids, costs = data['y'], np.array(data['pool_ids']), data['macs']
    observed = set(int(i) for i in initial)
    rng = np.random.default_rng(seed + 100000)
    random_score = rng.random(len(y))  # one continuation, paired resampling across seeds
    top = np.argsort(-y, kind='stable')
    front = set(pareto_indices(y, costs))
    full_hv = hypervolume(y, costs)
    trace, selections, benchmark, prediction_rows = [], [], None, []
    fidelity = 40 if method.endswith('40') else 60 if method.endswith('60') else 0
    while True:
        tr = np.array(sorted(observed))
        n = len(tr)
        trace.append(dict(seed=seed, method=method, evaluated=n, regret=float(y.max()-y[tr].max()),
            best_map=float(y[tr].max()), top3_recall=len(set(top[:3]) & observed)/3,
            top5_recall=len(set(top[:5]) & observed)/5, pareto_recall=len(front & observed)/len(front),
            hv_regret=full_hv-hypervolume(y[tr], costs[tr]),
            candidate_epochs_total=72*fidelity+(80-fidelity)*n,
            additional_epochs_after_initial20=(80-fidelity)*(n-20),
            historical_campaign_total_epochs=144*40+108*20+(80-fidelity)*n if fidelity == 60 else
                144*40+40*n if fidelity == 40 else 80*n))
        if n == 56:
            break
        remain = np.array([i for i in range(len(y)) if i not in observed])
        if method == 'random':
            score = random_score[remain]
        elif method.startswith('direct'):
            score = data['map'+method[6:]][remain]
        else:
            family, mode = method.split('_')
            X, base, _ = features[mode]
            score = predict(family, X, base, tr, y[tr].copy(), remain, seed+n)
        assert np.isfinite(score).all()
        if n == 20:
            benchmark = dict(seed=seed, method=method, n_train=20, n_holdout=52, **score_metrics(y[remain], score))
            for i, s in zip(remain, score):
                prediction_rows.append(dict(seed=seed, method=method, architecture_id=ids[i],
                    score=float(s), score_type='preference' if method.startswith('pairwise') else 'random' if method=='random' else 'map',
                    observed_map80=float(y[i])))
        order = np.lexsort((ids[remain], -score))[:6]
        for rank, p in enumerate(order, 1):
            i = int(remain[p])
            selections.append(dict(seed=seed, method=method, observed_before=n, rank_in_batch=rank,
                                   architecture_id=ids[i], score=float(score[p]), revealed_map80=float(y[i])))
            observed.add(i)
    return trace, selections, benchmark, prediction_rows


def mean_ci(values, alpha=.05, seed=424242):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return dict(n=0, mean=None, median=None, low=None, high=None)
    boot = np.random.default_rng(seed).choice(x, size=(5000, len(x))).mean(axis=1)
    lo, hi = np.quantile(boot, [alpha/2, 1-alpha/2])
    return dict(n=len(x), mean=float(x.mean()), median=float(np.median(x)), low=float(lo), high=float(hi))


def signal_audit(data, out):
    rows = []
    yend = np.array([float(data['final'][a]['endpoint_mAP50_95']) for a in data['pool_ids']])
    for semantics, target in [('best_to_best', data['y']), ('endpoint_to_endpoint', yend)]:
        for budget in (40, 60):
            low = data['map'+str(budget)] if semantics == 'best_to_best' else data['endpoint'+str(budget)]
            for group in ['ALL'] + sorted(set(data['combos'])):
                mask = np.ones(len(low), bool) if group == 'ALL' else data['combos'] == group
                a, b = low[mask], target[mask]
                ci = prior.corr_ci(a, b, np.random.default_rng(4242))
                rows.append(dict(semantics=semantics, budget=budget, macro=group, n=len(a),
                    spearman=ci['spearman'], spearman_low=ci['spearman_ci95'][0], spearman_high=ci['spearman_ci95'][1],
                    kendall=ci['kendall'], kendall_low=ci['kendall_ci95'][0], kendall_high=ci['kendall_ci95'][1],
                    pearson=float(pearsonr(a,b).statistic), top3_recall=prior.topk_recall(b,a,3),
                    top5_recall=prior.topk_recall(b,a,5), **prior.concordance(a,b)))
    save_csv(out/'BUDGET_FIDELITY.csv', rows)
    ms = pd.DataFrame(data['seeds'])
    noise, pairs = [], []
    for aid, group in ms.groupby('architecture_id', sort=True):
        assert set(group.seed.astype(int)) == {42,43,44}
        for key in ['mAP50_95', 'endpoint_mAP50_95']:
            v = group[key].astype(float).to_numpy()
            noise.append(dict(architecture_id=aid, label=group.iloc[0]['label'], metric=key, n=3,
                mean=v.mean(), std=v.std(ddof=1), variance=v.var(ddof=1), min=v.min(), max=v.max()))
    for a, b in [('A1','A2'), ('B1','B2')]:
        aa = ms[ms.label==a].set_index('seed'); bb = ms[ms.label==b].set_index('seed')
        d = aa['mAP50_95'].astype(float)-bb['mAP50_95'].astype(float)
        pairs.append(dict(pair=a+'-'+b, n=3, delta42=d['42'],delta43=d['43'],delta44=d['44'],
                          mean=d.mean(), std=d.std(ddof=1), positive_seeds=int((d>0).sum())))
    save_csv(out/'SEED_NOISE.csv',noise); save_csv(out/'PAIRED_SEED_DELTAS.csv',pairs)
    seedrank=[]
    for a,b in [(42,43),(42,44),(43,44)]:
        va=ms[ms.seed==str(a)].sort_values('architecture_id').mAP50_95.astype(float).to_numpy()
        vb=ms[ms.seed==str(b)].sort_values('architecture_id').mAP50_95.astype(float).to_numpy()
        seedrank.append(dict(seed_a=a,seed_b=b,n=6,spearman=prior.safe_spearman(va,vb),kendall=prior.safe_kendall(va,vb)))
    save_csv(out/'CROSS_SEED_RANKING.csv',seedrank)
    return rows, noise


def summarize(trace, bench, out):
    trace = pd.DataFrame(trace); bench = pd.DataFrame(bench)
    trace['partition'] = np.where(trace.seed<61020,'development','verification')
    bench['partition'] = np.where(bench.seed<61020,'development','verification')
    save_csv(out/'REPLAY_TRAJECTORIES.csv',trace)
    save_csv(out/'PREDICTOR_BENCHMARK_RAW.csv',bench)
    summary=[]
    for (part,method,n),g in trace[trace.evaluated.isin(CHECKS)].groupby(['partition','method','evaluated']):
        r=dict(partition=part,method=method,evaluated=n)
        for key in ['regret','top3_recall','top5_recall','pareto_recall','hv_regret']:
            r.update({key+'_'+k:v for k,v in mean_ci(g[key]).items()})
        r['candidate_epochs_total']=int(g.candidate_epochs_total.iloc[0])
        summary.append(r)
    save_csv(out/'REPLAY_SUMMARY.csv',summary)
    bs=[]
    for (part,method),g in bench.groupby(['partition','method']):
        r=dict(partition=part,method=method)
        for key in ['spearman','kendall','top3_recall','top5_recall','ndcg10','rank_mae']:
            r.update({key+'_'+k:v for k,v in mean_ci(g[key]).items()})
        bs.append(r)
    save_csv(out/'PREDICTOR_BENCHMARK.csv',bs)
    # Select only on development; fixed model parameters, no verification feedback.
    checkpoints=trace[trace.evaluated.isin(CHECKS)]
    development=checkpoints[checkpoints.partition=='development'].groupby('method').agg(
        regret=('regret','mean'), top5=('top5_recall','mean'))
    selected={}
    for mode in ['arch','mf40','mf60']:
        candidates=[m for m in METHODS if m.endswith('_'+mode)]
        selected[mode]=min(candidates,key=lambda m:(development.loc[m,'regret'],-development.loc[m,'top5'],m))
    dump(out/'DEVELOPMENT_SELECTION.json',selected)
    comparisons=[]
    for mode,method in selected.items():
        for control in ['random'] + ([] if mode=='arch' else ['direct'+mode[2:]]):
            p=checkpoints[(checkpoints.partition=='verification')&(checkpoints.method==method)]
            c=checkpoints[(checkpoints.partition=='verification')&(checkpoints.method==control)]
            joint=p.merge(c,on=['seed','evaluated'],suffixes=('_p','_c'),validate='one_to_one')
            for budget in ['AUC_CHECKPOINT_MEAN']+list(CHECKS):
                j=joint if isinstance(budget,str) else joint[joint.evaluated==budget]
                # Aggregate within initialization before bootstrap: not 90 independent rows.
                improvements=(j.regret_c-j.regret_p).groupby(j.seed).mean()
                recall=(j.top5_recall_p-j.top5_recall_c).groupby(j.seed).mean()
                ci=mean_ci(improvements); gate_ci=mean_ci(improvements,alpha=.01)
                comparisons.append(dict(mode=mode,method=method,control=control,budget=budget,
                    **{'regret_reduction_'+k:v for k,v in ci.items()},
                    simultaneous_low=gate_ci['low'],simultaneous_high=gate_ci['high'],
                    top5_difference_mean=recall.mean(),strict_win_fraction=float((improvements>0).mean()),
                    **{'top5_difference_'+k:v for k,v in mean_ci(recall).items() if k in ['low','high']}))
    save_csv(out/'PAIRED_VERIFICATION.csv',comparisons)
    eligible=[]
    for mode,method in selected.items():
        tests=[r for r in comparisons if r['method']==method and r['budget']=='AUC_CHECKPOINT_MEAN']
        if all(r['simultaneous_low']>0 and r['top5_difference_mean']>=0 for r in tests): eligible.append(method)
    return trace, pd.DataFrame(bs), selected, comparisons, eligible


def figures(out, data, trace):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    target=np.array(data['y'])
    points=pd.DataFrame({'architecture_id':data['pool_ids'],'map40':data['map40'],'map60':data['map60'],'map80':target})
    save_csv(out/'figures/fidelity_scatter.csv',points)
    fig,ax=plt.subplots(1,2,figsize=(9,4))
    for a,b in zip(ax,[40,60]):
        a.scatter(points['map'+str(b)],target,s=15); a.set(xlabel=f'Best valid mAP at {b}e',ylabel='Best valid mAP at 80e')
    fig.tight_layout()
    for ext in ['png','svg']: fig.savefig(out/f'figures/fidelity_scatter.{ext}',dpi=180)
    plt.close(fig)
    chosen=['random','direct40','direct60','ridge_arch','et_mf40','et_mf60']
    g=trace[(trace.partition=='verification')&trace.method.isin(chosen)].groupby(['method','evaluated']).agg(
        mean_regret=('regret','mean'),median_regret=('regret','median'),top5_recall=('top5_recall','mean')).reset_index()
    save_csv(out/'figures/replay_curves.csv',g)
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for method,r in g.groupby('method'):
        ax[0].plot(r.evaluated,r.mean_regret,label=method); ax[1].plot(r.evaluated,r.top5_recall,label=method)
    ax[0].set(xlabel='80e labels observed',ylabel='Mean simple regret');ax[1].set(xlabel='80e labels observed',ylabel='Top-5 recall')
    ax[1].legend(fontsize=7);fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(out/f'figures/replay_curves.{ext}',dpi=180)
    plt.close(fig)


def report(out, data, fidelity, noise, bench, selected, comparisons, eligible, elapsed):
    lines=['# Phase6A.2 — Signal/noise and paired offline continuation replay',
        '\nNo detector training. No test data access. Existing Phase6A/6A.1 outputs preserved.',
        '\n## Scope and integrity',
        '- 144@40 / 108@60 / 72@80 matched by architecture ID, seed42 and baseline hash.',
        '- 80e replay population is promotion-conditioned: 49 P345, 23 P2345, zero P234.',
        '- All primary metrics are best-valid within budget. Endpoint-to-endpoint is separate.',
        '- 50 shared initial20 sets: 20 development, 30 verification. Verification seeds were fixed before execution.',
        '- Verification reuses the same 72 architecture labels; it tests initialization robustness, not independent population generalization.',
        '\n## Fidelity', '| Target | Budget | n | Spearman [95% CI] | Kendall | Top3 | Top5 |','|---|---:|---:|---|---:|---:|---:|']
    for r in fidelity:
        if r['macro']=='ALL': lines.append(f"| {r['semantics']} | {r['budget']} | {r['n']} | {r['spearman']:.4f} [{r['spearman_low']:.4f}, {r['spearman_high']:.4f}] | {r['kendall']:.4f} | {r['top3_recall']:.3f} | {r['top5_recall']:.3f} |")
    stds=[r['std'] for r in noise if r['metric']=='mAP50_95']
    lines += ['\n## Training variability',f'- Six selected architectures, three seeds each: median sample SD={np.median(stds):.6f}, range={min(stds):.6f}..{max(stds):.6f}.',
        '- This variability exceeds approximately 0.003 regret for several architectures. It is not proof that noise alone caused surrogate failure.',
        '- n=3 and selection on seed42 prevent treating this SD as a universal noise floor or hard admission threshold.',
        '\n## Predictor comparison (verification initial20 / unseen52)',
        '| Method | Spearman | Kendall | Top5 | NDCG@10 |','|---|---:|---:|---:|---:|']
    for r in bench[bench.partition=='verification'].to_dict('records'):
        lines.append(f"| {r['method']} | {r['spearman_mean']:.4f} | {r['kendall_mean']:.4f} | {r['top5_recall_mean']:.3f} | {r['ndcg10_mean']:.3f} |")
    lines += ['\n## Selected-method paired verification',f'Development-only selection: `{selected}`.',
        '| Method | Control | Mean regret reduction [95% CI] | Simultaneous 99% CI | Top5 difference |',
        '|---|---|---|---|---:|']
    for r in comparisons:
        if r['budget']=='AUC_CHECKPOINT_MEAN':
            lines.append(f"| {r['method']} | {r['control']} | {r['regret_reduction_mean']:.6f} [{r['regret_reduction_low']:.6f}, {r['regret_reduction_high']:.6f}] | [{r['simultaneous_low']:.6f}, {r['simultaneous_high']:.6f}] | {r['top5_difference_mean']:.3f} |")
    lines += ['\n## Interpretation and gate',
        f'- Conditional promoted72 gate: {"PASS_WITH_LIMITATIONS" if eligible else "NOT_PASSED"}; eligible methods: {eligible}.',
        '- No Phase6B jobs launched. A passed gate would still not authorize claims about all144 or multi-seed expected performance.',
        '- Direct60 is a continuation baseline, not an architecture-only predictor. Its stronger global correlation does not guarantee recovering the best architecture early.',
        '- Pairwise models output preference scores, not calibrated mAP or uncertainty. No EHVI or hardware-aware selection is claimed.',
        '\n## Cost and replay corrections',
        '- Primary paired statistic: average regret at 32/44/56, with bootstrap over initialization seeds. 72/72 is excluded.',
        '- Random shares initial20 and has one random continuation per seed. Mean regret avoids zero-median saturation being an impossible strict-win gate.',
        '- No requirement that gains exceed an SD estimated from six selected architectures; noise is reported separately.',
        '- Total candidate-epochs: architecture-only=80*n; MF40=72*40+40*n; MF60=72*60+20*n. Costs are not equal at equal query counts.',
        '- For the actual historical campaign, 144*40+108*20=7920 candidate-epochs were already spent before final continuation; not new savings.',
        '- No GPU-hours claimed: historical concurrency makes candidate-epochs different from wall time.',
        '\n## Missing features and limitations',
        '- Full local curves exist for 12 original witnesses only; no slope/loss feature was invented for the other finalists.',
        '- Budget40/60 summary features use only information available at the stated budget. 40-to-60 best-score change is not a last-epoch slope.',
        '- Best-by-budget scores share prefixes, so best60-to-best80 correlation partly reflects that cumulative definition.',
        '- Bootstrap intervals condition on the fixed72 labels and shared official validation set; they do not include training-seed or dataset uncertainty.',
        '- Accuracy-Compute Pareto only. No FPGA latency inference, no test images/metrics, no changes to frozen architectures.',
        f'\nElapsed CPU analysis wall time: {elapsed:.1f}s. `test_evaluated=false`.']
    (out/'PHASE6A2_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=DEFAULT_OUT)
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--run',action='store_true')
    args=p.parse_args();out=args.output.resolve()
    if args.prepare:
        prepare(out);print('Protocol frozen:',out/'PROTOCOL.json',flush=True);return
    protocol=json.loads((out/'PROTOCOL.json').read_text(encoding='utf-8'))
    assert inputs()==protocol['input_sha256'], 'Input or source hashes changed since protocol freeze'
    if (out/'EXECUTION_MANIFEST.json').exists():raise ValueError('Completed output exists; preserve it and choose a new output directory.')
    start=time.perf_counter();data=prior.load_pool();audit_data(data)
    features={m:make_features(data,m) for m in ['arch','mf40','mf60']}
    dump(out/'FEATURE_SCHEMA.json',{m:fs[2] for m,fs in features.items()})
    encoding=prior.encoding_audit(data);assert not encoding['encoding_insufficient']
    dump(out/'ENCODING_AUDIT.json',encoding)
    dump(out/'CURVE_COVERAGE.json',prior.curve_coverage(prior.CURVES,set(data['pool_ids'])))
    fidelity,noise=signal_audit(data,out)
    print('Data, fidelity, noise and encoding audit complete',flush=True)
    initial={str(s):prior.stratified_indices(data['combos'],20,s).tolist() for s in SEEDS}
    dump(out/'INITIAL_SETS.json',{s:[data['pool_ids'][i] for i in ix] for s,ix in initial.items()})
    trace,selected,bench,predictions=[],[],[],[]
    for seed in SEEDS:
        for method in METHODS:
            t,s,b,pr=replay(data,method,seed,initial[str(seed)],features)
            trace+=t;selected+=s;bench.append(b);predictions+=pr
        print(f'Paired replay {seed-SEEDS[0]+1}/50 complete ({time.perf_counter()-start:.1f}s)',flush=True)
    save_csv(out/'REPLAY_SELECTIONS.csv',selected);save_csv(out/'HOLDOUT_PREDICTIONS.csv',predictions)
    traces,bs,choice,comparisons,eligible=summarize(trace,bench,out)
    figures(out,data,traces)
    elapsed=time.perf_counter()-start
    report(out,data,fidelity,noise,bs,choice,comparisons,eligible,elapsed)
    assert inputs()==protocol['input_sha256'], 'Source/input changed during execution'
    manifest=dict(status='COMPLETE',conditional_gate='PASS_WITH_LIMITATIONS' if eligible else 'NOT_PASSED',
        eligible_methods=eligible,phase6b_started=False,new_detector_training=0,test_evaluated=False,
        matched_hashes_unchanged=True,replay_seeds=50,methods=len(METHODS),initial20_unique=True,
        protocol_sha256=prior.sha256(out/'PROTOCOL.json'),elapsed_seconds=elapsed,
        files={str(f.relative_to(out)):prior.sha256(f) for f in out.rglob('*') if f.is_file()})
    dump(out/'EXECUTION_MANIFEST.json',manifest)
    print(json.dumps({k:manifest[k] for k in ['status','conditional_gate','eligible_methods','elapsed_seconds']},indent=2))


if __name__=='__main__':main()
