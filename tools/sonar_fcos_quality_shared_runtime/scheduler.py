"""Resource-only wrapper around the frozen v2 finite supervisor and workers."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

from tools.sonar_fcos_pilot.common import ROOT, read_json, resolve_path, sha256_file, utc_now, write_json
from tools.sonar_fcos_quality_v2 import scheduler as original
from .resource import LaunchResourceBlockedError, SharedResources, validate_policy

MODULE = 'tools.sonar_fcos_quality_shared_runtime.scheduler'
RUNTIME_FILES = ('tools/sonar_fcos_quality_shared_runtime/__init__.py',
                 'tools/sonar_fcos_quality_shared_runtime/resource.py',
                 'tools/sonar_fcos_quality_shared_runtime/scheduler.py')


def verify_freeze(config_path, policy_path, freeze_path, review_path):
    """Verify the independently reviewed resource wrapper; no metric reads."""
    from tools.sonar_fcos_quality_v2.campaign import load_config, source_binding
    policy = validate_policy(read_json(policy_path))
    manifest, review = read_json(freeze_path), read_json(review_path)
    if manifest.get('schema_version') != 1:
        raise ValueError('Invalid shared runtime freeze schema')
    if manifest['policy_sha256'] != sha256_file(policy_path) or manifest['config_sha256'] != sha256_file(config_path):
        raise ValueError('Shared policy or original experiment config changed')
    paths = manifest['runtime_sources']
    if not set(RUNTIME_FILES) <= set(paths):
        raise ValueError('Shared runtime freeze is incomplete')
    for relative, digest in paths.items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT.resolve()) or sha256_file(path) != digest:
            raise ValueError('Shared runtime source changed: ' + relative)
    source_set_sha = original.canonical_sha(paths)
    if (review.get('status') != 'PASS' or review.get('policy_sha256') != manifest['policy_sha256']
            or review.get('runtime_source_set_sha256') != source_set_sha
            or review.get('freeze_manifest_sha256') != sha256_file(freeze_path)):
        raise ValueError('Independent review does not match the frozen shared runtime')
    config = load_config(config_path)
    original.run_order(config)
    source = source_binding(config)
    if original.canonical_sha(source) != manifest['source_binding_sha256']:
        raise ValueError('Original experimental source binding changed')
    output = resolve_path(config['output_dir'])
    if read_json(output / 'execution_manifest.json')['source_binding'] != source:
        raise ValueError('Original execution manifest source binding differs')
    return policy, config, source


def prepare_claims(output, source, policy, inherited_affinity):
    """Fill only unassigned pairs under the unchanged v2 claims lock."""
    original_initialize = _ORIGINAL_INITIALIZE
    original_initialize(output, source, inherited_affinity)
    path = Path(output) / 'scheduler/seed_pair_claims.json'
    with original.blocking_lock(Path(output) / 'scheduler/claims.lock'):
        claims = read_json(path)
        if claims['source_binding_sha256'] != original.canonical_sha(source):
            raise ValueError('Seed claims source binding changed')
        if set(claims['pairs']) != {'42', '43', '44'}:
            raise ValueError('Seed claims differ from the fixed six-run budget')
        for seed, gpu in policy['pair_gpu_mapping'].items():
            current = claims['pairs'][seed]['gpu']
            if current not in (None, gpu):
                raise ValueError('Existing seed-pair GPU affinity cannot be moved')
        for seed, gpu in policy['pair_gpu_mapping'].items():
            if claims['pairs'][seed]['gpu'] is None:
                claims['pairs'][seed].update(gpu=gpu, assigned_at=utc_now(),
                                            resource_policy='shared_gpu_v3')
        write_json(path, claims)
    return claims


_ORIGINAL_INITIALIZE = original.initialize_claims
_ORIGINAL_SELECT = original.select_idle_gpu
_ORIGINAL_FOREIGN = original.foreign_pids
_ORIGINAL_ENVIRONMENT = original.child_environment
_ORIGINAL_SUBPROCESS = original.subprocess


class SubprocessAdapter:
    def __init__(self, policy_path, freeze_path, review_path, subprocess_module=None):
        self.paths = tuple(str(Path(value).resolve()) for value in (policy_path, freeze_path, review_path))
        self.base = subprocess_module or _ORIGINAL_SUBPROCESS

    def __getattr__(self, name):
        return getattr(self.base, name)

    def Popen(self, command, *args, **kwargs):
        command = list(command)
        prefix = [sys.executable, '-m', 'tools.sonar_fcos_quality_v2.scheduler', 'worker']
        if command[:4] == prefix:
            gpu = int(command[command.index('--gpu') + 1])
            if gpu not in (0, 2, 3):
                raise ValueError('GPU 1 cannot start a shared-runtime worker')
            command[2] = MODULE
            for flag, path in zip(('--policy', '--freeze-manifest', '--review-gate'), self.paths):
                command.extend((flag, path))
        return self.base.Popen(command, *args, **kwargs)


def install_hooks(policy, output, policy_path, freeze_path, review_path):
    def record(value):
        write_json(Path(output) / 'scheduler' / f"shared_resource_gpu{value['physical_gpu']}.json",
                   {**value, 'updated_at': utc_now(), 'policy_sha256': sha256_file(policy_path)})
    resources = SharedResources(policy, record=record)
    original.select_idle_gpu = lambda cards, scheduling=None: resources.select_gpu(cards, scheduling, _ORIGINAL_SELECT)
    original.foreign_pids = lambda cards, gpu, own: resources.foreign_pids(cards, gpu, own, _ORIGINAL_FOREIGN)
    original.child_environment = lambda card, pause: resources.child_environment(card, pause, _ORIGINAL_ENVIRONMENT)
    original.initialize_claims = lambda out, source, affinity=None: prepare_claims(out, source, policy, affinity or {})
    original.subprocess = SubprocessAdapter(policy_path, freeze_path, review_path)
    return resources


def run_worker(config_path, gpu, output):
    """Retry only a resource veto immediately before a train process is created.

    In this specific path the original worker has already released its singleton
    lock. Runtime resource failures are caught inside the original live-child
    monitor and use its cooperative pause protocol instead.
    """
    while True:
        try:
            return original.worker(config_path, gpu)
        except LaunchResourceBlockedError as exc:
            if gpu not in (2, 3):
                raise
            write_json(Path(output) / 'scheduler' / f'worker_gpu{gpu}.json',
                       {'status': 'WAITING_SHARED_RESOURCE', 'updated_at': utc_now(),
                        'pid': os.getpid(), 'physical_gpu': gpu,
                        'mainline_reserved_gpu': 1, 'reason': str(exc)})
            time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('supervise', 'worker'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--gpu', type=int, choices=(0, 2, 3))
    parser.add_argument('--policy', required=True)
    parser.add_argument('--freeze-manifest', required=True)
    parser.add_argument('--review-gate', required=True)
    args = parser.parse_args()
    if args.mode == 'worker' and args.gpu is None:
        parser.error('worker requires --gpu')
    policy, config, source = verify_freeze(args.config, args.policy, args.freeze_manifest, args.review_gate)
    if args.mode == 'worker' and args.gpu in (2, 3):
        # The original train command inherits this lower CPU scheduling priority.
        current = os.getpriority(os.PRIO_PROCESS, 0)
        if current < policy['guard']['shared_worker_nice']:
            os.nice(policy['guard']['shared_worker_nice'] - current)
    install_hooks(policy, resolve_path(config['output_dir']), args.policy, args.freeze_manifest, args.review_gate)
    value = run_worker(args.config, args.gpu, resolve_path(config['output_dir'])) if args.mode == 'worker' else original.supervise(args.config)
    print(json.dumps(value, ensure_ascii=False))


if __name__ == '__main__':
    main()
