"""Preregistered nine-run recipe check on the frozen C01 YOLOv9-T quality cell.

R1 adds author-style EMA, R2 adds a precise step-based warmup/cosine schedule,
and R3 adds the registered mosaic/affine adaptation. The frozen C01 model,
loss, optimizer construction, train loop, validation and metric code are reused.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.sonar_head_factorial import train as old
from tools.sonar_recipe_check import augmentation

p, torch, yaml = old.p, old.torch, old.yaml
read_json = old.read_json
CONFIG = ROOT / 'configs/sonar_recipe_check_20261013.yaml'
VERSION = 'sonar_recipe_check_20261013_01'
RECIPES, SEEDS = ('R1', 'R2', 'R3'), (42, 43, 44)
BACKBONE, HEAD = 'YOLOv9_T', 'quality_aware'
EXPECTED_RECIPES = {
    key: dict(ema=dict(decay=.9999, tau=2000),
              scheduler='none' if key == 'R1' else 'linear3_cosine',
              augmentation='mosaic_affine_additive' if key == 'R3' else 'frozen')
    for key in RECIPES}
EXPECTED_LR_SCHEDULE = dict(warmup_epochs=3, final_lr_ratio=.01)
STABLE_EPOCHS = old.STABLE_EPOCHS


class ModelEMA:
    """Exact floating-state update formula from pinned author ModelEMA.

    Integer buffers stay at their initial values, matching the author. No RNG
    is consumed and the training model is neither mutated nor evaluated here.
    """
    def __init__(self, model, decay=.9999, tau=2000):
        self.ema = deepcopy(model).eval()
        self.decay, self.tau, self.updates = float(decay), int(tau), 0
        for parameter in self.ema.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = self.decay * (1 - math.exp(-self.updates/self.tau))
        raw = model.state_dict()
        for key, value in self.ema.state_dict().items():
            if value.dtype.is_floating_point:
                value *= decay
                value += (1-decay) * raw[key].detach()

    def state_dict(self):
        return dict(model_state_dict=self.ema.state_dict(), updates=self.updates,
                    decay=self.decay, tau=self.tau)

    def load_state_dict(self, state):
        if (state['decay'] != self.decay or state['tau'] != self.tau
                or type(state['updates']) is not int or state['updates'] < 0):
            raise ValueError('EMA configuration/update-count drift')
        self.ema.load_state_dict(state['model_state_dict'], strict=True)
        self.updates = state['updates']


class RecipeSchedule:
    """LR for the next optimizer step; index 0 and final index are explicit."""
    def __init__(self, optimizer, kind, *, batches_per_epoch=83, epochs=80,
                 warmup_epochs=3, final_lr_ratio=.01, base_lr=.001):
        if kind not in ('none', 'linear3_cosine'):
            raise ValueError('Unregistered scheduler')
        self.optimizer = optimizer
        self.spec = dict(kind=kind, batches_per_epoch=batches_per_epoch, epochs=epochs,
                         warmup_epochs=warmup_epochs, final_lr_ratio=final_lr_ratio,
                         base_lr=base_lr)
        self.next_step = 0

    def lr_at(self, step):
        spec = self.spec
        total = spec['epochs'] * spec['batches_per_epoch']
        if not 0 <= step < total:
            raise ValueError('Schedule step outside registered training budget')
        if spec['kind'] == 'none':
            return spec['base_lr']
        warmup = spec['warmup_epochs'] * spec['batches_per_epoch']
        if step < warmup:
            factor = step/(warmup-1)
        else:
            progress = (step-(warmup-1))/(total-warmup)
            factor = spec['final_lr_ratio'] + (1-spec['final_lr_ratio'])*(1+math.cos(math.pi*progress))/2
        return spec['base_lr'] * factor

    def before_step(self):
        value = self.lr_at(self.next_step)
        for group in self.optimizer.param_groups:
            group['lr'] = value
        return value

    def after_step(self):
        self.next_step += 1

    def state_dict(self):
        return dict(spec=self.spec, next_step=self.next_step)

    def load_state_dict(self, state):
        if (state['spec'] != self.spec or type(state['next_step']) is not int
                or not 0 <= state['next_step'] <= self.spec['epochs']*self.spec['batches_per_epoch']):
            raise ValueError('Scheduler configuration/step drift')
        self.next_step = state['next_step']


class ObservedOptimizer:
    """Call the unchanged C01 train_epoch with only registered step hooks."""
    def __init__(self, optimizer, model, ema, scheduler):
        self.optimizer, self.model = optimizer, model
        self.ema, self.scheduler = ema, scheduler
        self.observed_lrs = []

    def __getattr__(self, name):
        return getattr(self.optimizer, name)

    def step(self, *args, **kwargs):
        value = self.scheduler.before_step()
        output = self.optimizer.step(*args, **kwargs)
        self.scheduler.after_step()
        self.ema.update(self.model)
        self.observed_lrs.append(value)
        return output


def train_epoch(model, criterion, optimizer, loader, device, ema, scheduler, max_batches=None):
    observer = ObservedOptimizer(optimizer, model, ema, scheduler)
    result = old.train_epoch(model, criterion, observer, loader, device, max_batches)
    result.update(lr_first=observer.observed_lrs[0], lr_min=min(observer.observed_lrs),
                  lr_max=max(observer.observed_lrs), ema_updates=ema.updates,
                  scheduler_next_step=scheduler.next_step)
    return result


def load_config(config=CONFIG):
    cfg = yaml.safe_load(Path(config).read_text(encoding='utf-8'))
    previous, baseline = old.load_config()
    expected_training = dict(epochs=80, batch_size=4, input=640, optimizer='AdamW',
                             lr=.001, weight_decay=.0001, amp=False, workers=0,
                             accumulation=1, save_prediction_epochs=list(STABLE_EPOCHS),
                             stable_prediction_image_count=99, missing_stable_epoch_policy='fail_closed')
    if (cfg.get('version') != VERSION or cfg.get('result_root') != f'results/{VERSION}/remote'
            or cfg.get('recipes') != EXPECTED_RECIPES or cfg.get('seeds') != list(SEEDS)
            or cfg.get('total_formal_runs') != 9 or cfg.get('lr_schedule') != EXPECTED_LR_SCHEDULE
            or cfg.get('augmentation') != augmentation.PARAMETERS
            or any(cfg.get('training', {}).get(k) != v for k, v in expected_training.items())
            or any(cfg.get(k) != previous[k] for k in ('author', 'baseline', 'environment_spec',
                                                       'subset_contract', 'subset_contract_sha256', 'evaluation'))):
        raise ValueError('Unregistered recipe training/evaluation/configuration drift')
    if (cfg['queue'].get('retry') != 'none' or cfg['queue'].get('adaptive_seed_extension') is not False
            or cfg['decision'].get('primary_checkpoint') != 'mean_epochs_76_80'
            or cfg['decision'].get('metric') != 'mAP50_95'
            or cfg['decision'].get('threshold_absolute') != .01):
        raise ValueError('Unregistered recipe decision/retry policy')
    old.load_subset_contract(cfg)
    return cfg, baseline


def source_paths(cfg, config=CONFIG):
    previous, _ = old.load_config()
    paths = set(old.source_paths(previous))
    paths.update((ROOT / 'tools/sonar_recipe_check').glob('*.py'))
    paths.update((ROOT / 'tests').glob('test_sonar_recipe_check*.py'))
    paths.add(Path(config))
    for name in ('r0_reuse_gate', 'preregistration'):
        paths.add(old._inside_root(cfg[name]))
    return sorted(paths)


def source_hashes(cfg, config=CONFIG):
    return {path.relative_to(ROOT).as_posix(): p.sha(path) for path in source_paths(cfg, config)}


def read_gate(cfg):
    path = old._inside_root(cfg['r0_reuse_gate'])
    if p.sha(path) != cfg['r0_reuse_gate_sha256']:
        raise ValueError('R0 reuse gate hash drift')
    gate = read_json(path)
    if gate.get('status') != 'PASS':
        raise ValueError('R0 reuse evidence not eligible')
    for relative, expected in gate['source_sha256'].items():
        source = old._inside_root(relative)
        if not source.exists() and relative in gate.get('staging_overrides', {}):
            source = old._inside_root(gate['staging_overrides'][relative])
        if p.sha(source) != expected:
            raise ValueError(f'R0 source changed: {relative}')
    if (set(gate['initializations']) != {str(seed) for seed in SEEDS}
            or set(gate['training_witnesses']) != {str(seed) for seed in SEEDS}):
        raise ValueError('R0 must supply all three paired seeds')
    for seed in SEEDS:
        rows = gate['training_witnesses'][str(seed)]
        if len(rows) != 80 or [row['epoch'] for row in rows] != list(range(1, 81)):
            raise ValueError('R0 epoch witness incomplete')
    from tools.sonar_recipe_check.evidence import verify_r0_artifacts
    verify_r0_artifacts(cfg, gate)
    return gate


def prepare(root, config=CONFIG):
    cfg, baseline = load_config(config)
    gate = read_gate(cfg)
    if p.sha(old._inside_root(cfg['preregistration'])) != cfg['preregistration_sha256']:
        raise ValueError('Preregistration hash drift')
    inputs = p.verify_inputs(baseline)
    subset = old.load_subset_contract(cfg, inputs)
    targets = old.frozen_validation_targets(baseline)
    if [row['image'] for row in targets] != subset['full_images']:
        raise ValueError('Frozen full99 GT image membership differs')
    author = ROOT / cfg['author']['source_root']
    if (author / '.git').exists():
        commit = subprocess.check_output(['git', '-C', str(author), 'rev-parse', 'HEAD'], text=True).strip()
    else:
        commit = read_json(author / 'source_receipt.json')['commit']
    if commit != cfg['author']['commit']:
        raise ValueError('Pinned author commit drift')
    manifest = dict(version=VERSION, config=cfg, config_sha256=p.sha(config), author_commit=commit,
                    source_sha256=source_hashes(cfg, config), inputs=inputs,
                    r0_reuse_gate_sha256=cfg['r0_reuse_gate_sha256'],
                    r0_environment_fingerprint=gate['r0_environment_fingerprint'],
                    subset_contract_sha256=cfg['subset_contract_sha256'],
                    validation_targets=targets, validation_target_sha256=p.canonical_sha(targets),
                    subset_id=subset['subset_id'], test_evaluated=False)
    path = Path(root) / 'execution_manifest.json'
    if path.exists() and read_json(path) != manifest:
        raise ValueError('Existing recipe freeze differs; use a new campaign root')
    if not path.exists():
        p.dump(path, manifest)
    return manifest


def verify_manifest(root, config=CONFIG):
    cfg, baseline = load_config(config)
    manifest = read_json(Path(root) / 'execution_manifest.json')
    if (manifest['config'] != cfg or manifest['config_sha256'] != p.sha(config)
            or manifest['source_sha256'] != source_hashes(cfg, config)
            or manifest['inputs'] != p.verify_inputs(baseline)
            or manifest['validation_target_sha256'] != p.canonical_sha(manifest['validation_targets'])):
        raise ValueError('Frozen recipe source/config/data identity drift')
    read_gate(cfg)
    subset = old.load_subset_contract(cfg, manifest['inputs'])
    if [row['image'] for row in manifest['validation_targets']] != subset['full_images']:
        raise ValueError('Frozen validation target IDs differ')
    return cfg, baseline


def result_dir(root, recipe, seed, mode='formal'):
    return Path(root) / mode / recipe / f'seed_{seed}'


def binding(root, recipe, seed, mode):
    manifest = read_json(Path(root) / 'execution_manifest.json')
    return dict(execution_manifest_sha256=p.sha(Path(root) / 'execution_manifest.json'),
                environment_fingerprint=read_json(Path(root) / 'environment_lock.json')['fingerprint'],
                subset_contract_sha256=manifest['subset_contract_sha256'],
                validation_target_sha256=manifest['validation_target_sha256'],
                backbone=BACKBONE, head=HEAD, recipe=recipe, seed=seed, mode=mode,
                test_evaluated=False)


def make_run(root, recipe, seed, device, mode, config=CONFIG):
    from tools.sonar_head_factorial import model as adapter
    cfg, baseline = verify_manifest(root, config)
    if recipe not in RECIPES or seed not in SEEDS:
        raise ValueError('Unregistered recipe/seed')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no fallback')
    p.seed_all(seed)
    lock = old.bind_environment(root, device, cfg)
    gate = read_gate(cfg)
    if lock['fingerprint'] != gate['r0_environment_fingerprint']:
        raise ValueError('Recipe environment differs from the R0 evidence environment')
    model = adapter.build_model(BACKBONE, HEAD, seed,
                                source_root=ROOT / cfg['author']['source_root']).to(device)
    criterion = adapter.build_criterion(model, baseline)
    identity, metadata = model.initialization_identity(), model.metadata()
    metadata['criterion'] = adapter.criterion_metadata(model, criterion)
    metadata = json.loads(json.dumps(metadata, allow_nan=False))
    initial = p.state_hash(model)
    expected = gate['initializations'][str(seed)]
    if identity != expected['initialization'] or initial != expected['initial_state_sha256']:
        raise ValueError('Recipe model initialization differs from paired C01 R0')
    # Exactly the C01 reset/loader/optimizer sequence. EMA deepcopy has no RNG.
    p.seed_all(seed)
    rng_identity = p.training_rng_identity(seed)
    data = ROOT / baseline['dataset']['data_dir']
    train = p.frozen.make_loader(baseline, data, 'train', seed, 0, train=True)
    valid = p.frozen.make_loader(baseline, data, 'valid', seed, 0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    if recipe == 'R3':
        train = augmentation.wrap_loader(train, ROOT / cfg['author']['source_root'], cfg['augmentation'])
    ema = ModelEMA(model, **cfg['recipes'][recipe]['ema'])
    scheduler = RecipeSchedule(optimizer, cfg['recipes'][recipe]['scheduler'])
    folder = result_dir(root, recipe, seed, mode)
    folder.mkdir(parents=True, exist_ok=True)
    for name, value in [('initialization_identity.json', identity), ('model_info.json', metadata),
                        ('training_rng_identity.json', rng_identity)]:
        target = folder / name
        if target.exists() and read_json(target) != value:
            raise ValueError(f'Existing run identity differs: {name}')
        if not target.exists():
            p.dump(target, value)
    return dict(folder=folder, cfg=cfg, baseline=baseline, model=model, criterion=criterion,
                optimizer=optimizer, train=train, valid=valid, initial=initial,
                identity=identity, lock=lock, ema=ema, scheduler=scheduler, r0=gate)


def check_epoch_pairing(record, recipe, seed, gate):
    witness = gate['training_witnesses'][str(seed)][record['epoch']-1]
    keys = ['image_order_sha256', 'valid_batches_sha256']
    if recipe != 'R3':
        keys.append('augmented_batches_sha256')
    if any(record[key] != witness[key] for key in keys):
        raise ValueError(f'R0 paired data/sampler witness drift at {recipe}/{seed}/{record["epoch"]}')


def smoke_one(root, recipe, seed, device, config=CONFIG):
    folder = result_dir(root, recipe, seed, 'smoke')
    if folder.exists():
        raise FileExistsError('Smoke invocation already exists; preserve evidence')
    run = make_run(root, recipe, seed, device, 'smoke', config)
    tick = time.perf_counter()
    try:
        record = train_epoch(run['model'], run['criterion'], run['optimizer'], run['train'],
                             device, run['ema'], run['scheduler'], max_batches=2)
        metrics, records, valid_sha = old.evaluate(run['ema'].ema, run['criterion'],
                                                  run['valid'], run['baseline'], device, 1)
        final = p.state_hash(run['model'])
        if (record['batches'] != 2 or record['images'] != 8 or metrics['num_images'] != 4
                or final == run['initial'] or run['ema'].updates != 2
                or run['scheduler'].next_step != 2):
            raise ValueError('Smoke failed to exercise registered training/EMA/schedule')
        report = dict(status='PASS', binding=binding(root, recipe, seed, 'smoke'),
                      training=record, valid=metrics, valid_batches_sha256=valid_sha,
                      initialization=run['identity'], initial_state_sha256=run['initial'],
                      final_state_sha256=final, ema_state_sha256=p.state_hash(run['ema'].ema),
                      seconds=time.perf_counter()-tick, formal_eligible=False, test_evaluated=False)
        p.dump(folder / 'valid_records.json', records)
        p.dump(folder / 'smoke_report.json', report)
        return report
    except Exception as error:
        p.dump(folder / 'smoke_failure.json', dict(status='FAILED', error=str(error),
                error_type=type(error).__name__, formal_eligible=False, test_evaluated=False))
        raise
    finally:
        del run
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()


def smoke(root, device, config=CONFIG):
    rows = [smoke_one(root, recipe, 42, device, config) for recipe in RECIPES]
    if (len({row['initial_state_sha256'] for row in rows}) != 1
            or len({row['valid_batches_sha256'] for row in rows}) != 1
            or len({row['training']['image_order_sha256'] for row in rows}) != 1
            or rows[0]['training']['augmented_batches_sha256'] != rows[1]['training']['augmented_batches_sha256']):
        raise ValueError('Recipe smoke initialization/sampler/augmentation pairing failed')
    gate = dict(status='PASS', execution_manifest_sha256=p.sha(Path(root) / 'execution_manifest.json'),
                environment_fingerprint=read_json(Path(root) / 'environment_lock.json')['fingerprint'],
                recipes=list(RECIPES), report_sha256={recipe:p.sha(result_dir(root, recipe, 42, 'smoke') / 'smoke_report.json')
                                                    for recipe in RECIPES},
                formal_eligible=False, test_evaluated=False)
    p.dump(Path(root) / 'smoke_gate.json', gate)
    return gate


def require_smoke(root):
    gate = read_json(Path(root) / 'smoke_gate.json')
    if (gate['status'] != 'PASS' or gate['recipes'] != list(RECIPES)
            or gate['execution_manifest_sha256'] != p.sha(Path(root) / 'execution_manifest.json')
            or gate['environment_fingerprint'] != read_json(Path(root) / 'environment_lock.json')['fingerprint']):
        raise ValueError('Recipe smoke gate missing or source/environment drift')
    for recipe in RECIPES:
        path = result_dir(root, recipe, 42, 'smoke') / 'smoke_report.json'
        row = read_json(path)
        if (gate['report_sha256'][recipe] != p.sha(path) or row['status'] != 'PASS'
                or row['binding'] != binding(root, recipe, 42, 'smoke')):
            raise ValueError('Recipe smoke report hash/binding drift')


def validate_training_state(state, recipe, epoch):
    steps = epoch*83
    expected_spec = dict(kind=EXPECTED_RECIPES[recipe]['scheduler'], batches_per_epoch=83,
                         epochs=80, warmup_epochs=3, final_lr_ratio=.01, base_lr=.001)
    if (state['ema_state_dict']['updates'] != steps
            or state['scheduler_state_dict']['next_step'] != steps
            or state['ema_state_dict']['decay'] != .9999 or state['ema_state_dict']['tau'] != 2000
            or state['validation_weights'] != 'ema'
            or state['scheduler_state_dict']['spec'] != expected_spec):
        raise ValueError('EMA/scheduler checkpoint differs from epoch boundary')


def run_result(root, recipe, seed, *, write=False, config=CONFIG):
    cfg, baseline = verify_manifest(root, config)
    if recipe not in RECIPES or seed not in SEEDS:
        raise ValueError('Unregistered recipe/seed result')
    folder = result_dir(root, recipe, seed)
    expected = binding(root, recipe, seed, 'formal')
    latest = torch.load(folder / 'latest.pt', map_location='cpu', weights_only=False)
    history = latest['history']
    old.validate_history(history)
    best = max(history, key=lambda row: row['valid']['mAP50_95'])
    checkpoint = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
    validate_training_state(latest, recipe, 80)
    validate_training_state(checkpoint, recipe, best['epoch'])
    if (latest['binding'] != expected or latest['epoch'] != 80 or checkpoint['binding'] != expected
            or checkpoint['epoch'] != best['epoch'] or checkpoint['history'] != history[:best['epoch']]
            or read_json(folder / 'epoch_log.json') != history
            or read_json(folder / 'best_valid_metrics.json') != best['valid']
            or latest['initialization'] != read_json(folder / 'initialization_identity.json')):
        raise ValueError('Recipe checkpoint/binding/history mismatch')
    gate = read_gate(cfg)
    initial = gate['initializations'][str(seed)]
    if latest['initial_state_sha256'] != initial['initial_state_sha256'] or latest['initialization'] != initial['initialization']:
        raise ValueError('Result does not match frozen R0 initialization')
    for row in history:
        check_epoch_pairing(row, recipe, seed, gate)
    subset = old.load_subset_contract(cfg)
    best_records = read_json(folder / 'best_valid_records.json')
    if old.enrich_metrics(p.metrics_from_records(best_records, baseline)) != best['valid']:
        raise ValueError('Best EMA predictions do not reproduce logged metrics')
    terminal = old.enrich_metrics(p.metrics_from_records(read_json(folder / 'terminal_valid_records.json'), baseline))
    if terminal != history[-1]['valid'] or terminal != read_json(folder / 'terminal_valid_metrics.json'):
        raise ValueError('Terminal EMA predictions do not reproduce logged metrics')
    stable = old.verify_stable_predictions(folder, history, expected, baseline, subset['full_images'])
    if set(stable['paths']) != {str(epoch) for epoch in STABLE_EPOCHS}:
        raise ValueError('All five full99 prediction archives required')
    metrics = old.stable_metrics_from_history(history)
    report = dict(recipe=recipe, binding=expected, epochs=80, best_epoch=best['epoch'],
                  primary_checkpoint='mean_epochs_76_80', primary_metric='mAP50_95',
                  metrics=metrics, stable_metrics=metrics, stable_prediction_paths=stable['paths'],
                  stable_prediction_sha256=stable['sha256'], stable_epoch_metrics=stable['metrics'],
                  secondary_terminal_metrics=terminal, secondary_best_metrics=best['valid'],
                  initialization=latest['initialization'], initial_state_sha256=latest['initial_state_sha256'],
                  model_info=read_json(folder / 'model_info.json'), history=history,
                  validation_weights='ema', ema_updates=latest['ema_state_dict']['updates'],
                  hashes={name:p.sha(folder / name) for name in ('latest.pt', 'best.pt', 'epoch_log.json',
                    'best_valid_records.json', 'terminal_valid_records.json', 'terminal_valid_metrics.json')},
                  formal_eligible=True, test_evaluated=False)
    if write:
        p.dump(folder / 'result.json', report)
    return report


def run_training(root, recipe, seed, device, resume=False, config=CONFIG):
    require_smoke(root)
    folder = result_dir(root, recipe, seed)
    if (folder / 'run_binding.json').exists() and not resume:
        raise FileExistsError('Formal run already invoked; explicit epoch-boundary resume required')
    run = make_run(root, recipe, seed, device, 'formal', config)
    model, optimizer, ema, scheduler = (run[key] for key in ('model', 'optimizer', 'ema', 'scheduler'))
    expected = binding(root, recipe, seed, 'formal')
    images = old.load_subset_contract(run['cfg'])['full_images']
    history, start = [], 0
    if resume:
        state = torch.load(folder / 'latest.pt', map_location='cpu', weights_only=False)
        start, history = old.validate_resume_state(folder, state, expected, run['initial'],
                         run['identity'], run['baseline'], images)
        validate_training_state(state, recipe, start)
        model.load_state_dict(state['model_state_dict'], strict=True)
        optimizer.load_state_dict(state['optimizer_state_dict'])
        ema.load_state_dict(state['ema_state_dict'])
        scheduler.load_state_dict(state['scheduler_state_dict'])
        p.restore_rng(state['rng'], run['train'])
        if run['valid'].generator is not None:
            run['valid'].generator.set_state(state['valid_loader_generator'])
    else:
        p.dump(folder / 'run_binding.json', dict(binding=expected, initialization=run['identity'],
            initial_state_sha256=run['initial'], started_at=p.now(), validation_weights='ema',
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
    try:
        for epoch in range(start, 80):
            tick = time.perf_counter()
            previous_best = max((row['valid']['mAP50_95'] for row in history), default=-1.)
            record = train_epoch(model, run['criterion'], optimizer, run['train'], device, ema, scheduler)
            metrics, records, valid_sha = old.evaluate(ema.ema, run['criterion'], run['valid'], run['baseline'], device)
            record.update(epoch=epoch+1, valid=metrics, valid_batches_sha256=valid_sha,
                          wall_seconds=time.perf_counter()-tick)
            check_epoch_pairing(record, recipe, seed, run['r0'])
            if epoch+1 in STABLE_EPOCHS:
                record['prediction_archive'] = old.save_stable_predictions(folder, epoch+1, records, expected, images)
            history.append(record)
            old.validate_history(history, epoch+1)
            state = dict(binding=expected, initial_state_sha256=run['initial'], initialization=run['identity'],
                         model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                         ema_state_dict=ema.state_dict(), scheduler_state_dict=scheduler.state_dict(),
                         epoch=epoch+1, rng=p.rng_state(run['train']), history=history,
                         valid_loader_generator=run['valid'].generator.get_state() if run['valid'].generator is not None else None,
                         validation_weights='ema', test_evaluated=False)
            validate_training_state(state, recipe, epoch+1)
            if metrics['mAP50_95'] > previous_best:
                p.atomic_save(folder / 'best.pt', state)
                p.dump(folder / 'best_valid_metrics.json', metrics)
                p.dump(folder / 'best_valid_records.json', records)
            if epoch+1 == 80:
                p.dump(folder / 'terminal_valid_metrics.json', metrics)
                p.dump(folder / 'terminal_valid_records.json', records)
            p.atomic_save(folder / 'latest.pt', state)
            p.dump(folder / 'epoch_log.json', history)
            p.dump(folder / 'status.json', dict(status='RUNNING', epoch=epoch+1, binding=expected,
                   latest_valid={key:metrics[key] for key in old.METRICS}, updated_at=p.now()))
            print(f'{recipe} seed={seed} epoch={epoch+1}/80 valid_mAP50_95={metrics["mAP50_95"]:.8f} '
                  f'EMA_updates={ema.updates} lr={optimizer.param_groups[0]["lr"]:.8g} '
                  f'seconds={record["wall_seconds"]:.2f}', flush=True)
        report = run_result(root, recipe, seed, write=True, config=config)
        p.dump(folder / 'status.json', dict(status='COMPLETE', epochs=80, binding=expected,
               best_epoch=report['best_epoch'], primary_checkpoint='mean_epochs_76_80',
               stable_mAP50_95=report['stable_metrics']['mAP50_95'],
               stable_prediction_epochs=list(STABLE_EPOCHS), updated_at=p.now(), test_evaluated=False))
        return report
    except Exception as error:
        p.dump(folder / 'status.json', dict(status='FAILED_NO_AUTORETRY', completed_epochs=len(history),
               error_type=type(error).__name__, error=str(error), binding=expected, updated_at=p.now()))
        raise


def queue_jobs():
    return [(recipe, seed) for seed in SEEDS for recipe in RECIPES]


def queue(root, device, config=CONFIG):
    cfg, _ = verify_manifest(root, config)
    require_smoke(root)
    status_path = Path(root) / 'queue_status.json'
    if status_path.exists():
        raise FileExistsError('Queue already invoked; no automatic retry')
    status = dict(status='RUNNING', pid=os.getpid(), jobs=queue_jobs(), completed=[],
                  started_at=p.now(), execution_manifest_sha256=p.sha(Path(root) / 'execution_manifest.json'),
                  test_evaluated=False)
    p.dump(status_path, status)
    try:
        for recipe, seed in queue_jobs():
            folder = result_dir(root, recipe, seed)
            if folder.exists():
                raise FileExistsError(f'Existing formal folder cannot be queued: {folder}')
            folder.mkdir(parents=True)
            command = [sys.executable, str(Path(__file__).resolve()), 'train', '--config', str(config),
                       '--output', str(root), '--recipe', recipe, '--seed', str(seed), '--device', device]
            with (folder / 'stdout.log').open('x', encoding='utf-8') as handle:
                child = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
                status.update(active=[recipe, seed], child_pid=child.pid, updated_at=p.now())
                p.dump(status_path, status)
                try:
                    code = child.wait(timeout=float(cfg['queue']['maximum_hours_per_run'])*3600)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=10)
                    raise RuntimeError('Run exceeded registered timeout; no retry')
            if code:
                raise RuntimeError(f'{recipe}/{seed} failed with exit {code}; no retry')
            run_result(root, recipe, seed, config=config)
            status['completed'].append([recipe, seed])
            status.update(updated_at=p.now(), active=None, child_pid=None)
            p.dump(status_path, status)
        status.update(status='COMPLETE', updated_at=p.now())
        p.dump(status_path, status)
        return status
    except Exception as error:
        status.update(status='FAILED_NO_AUTORETRY', error=str(error), error_type=type(error).__name__, updated_at=p.now())
        p.dump(status_path, status)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'verify', 'smoke', 'train', 'queue', 'result'))
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--output', '--result-root', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--recipe', choices=RECIPES, default='R1')
    parser.add_argument('--seed', choices=SEEDS, type=int, default=42)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    cfg, _ = load_config(args.config)
    root = (args.output or ROOT / cfg['result_root']).resolve()
    spec = yaml.safe_load((ROOT / cfg['environment_spec']).read_text(encoding='utf-8'))
    torch.set_num_threads(spec['threads'])
    if args.mode == 'prepare':
        prepare(root, args.config)
        output = dict(status='PREPARED', execution_manifest_sha256=p.sha(root / 'execution_manifest.json'))
    elif args.mode == 'verify':
        verify_manifest(root, args.config)
        output = dict(status='PASS')
    elif args.mode == 'smoke':
        output = smoke(root, args.device, args.config)
    elif args.mode == 'train':
        output = run_training(root, args.recipe, args.seed, args.device, args.resume, args.config)
        output = dict(status='COMPLETE', binding=output['binding'], metrics=output['metrics'])
    elif args.mode == 'queue':
        output = queue(root, args.device, args.config)
    else:
        output = run_result(root, args.recipe, args.seed, write=True, config=args.config)
        output = dict(status='COMPLETE', binding=output['binding'], metrics=output['metrics'])
    print(json.dumps(output, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
