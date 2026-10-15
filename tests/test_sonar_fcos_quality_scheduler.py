"""Resource ownership and crash-resume checks for the bounded side queue."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from tools.sonar_fcos_quality import remote, scheduler


def card(index=0, **changes):
    return {'index': index, 'uuid': f'gpu-{index}', 'name': 'NVIDIA GeForce RTX 5090',
            'memory_used_mib': 29, 'memory_total_mib': 32607,
            'utilization_percent': 0, 'compute_pids': [], **changes}


def test_idle_selector_never_takes_mainline_gpu():
    assert scheduler.select_idle_gpu([card(1)], {'gpu_indices': [1], 'reserved_gpu_indices': []}) is None
    assert scheduler.select_idle_gpu([card(1), card(0)])['index'] == 0


@pytest.mark.parametrize('changes', [
    {'compute_pids': [123]}, {'memory_used_mib': 501}, {'utilization_percent': 6},
    {'memory_total_mib': 8000}, {'name': 'NVIDIA RTX A6000'}])
def test_busy_or_incompatible_gpu_is_ineligible(changes):
    assert scheduler.select_idle_gpu([card(**changes)]) is None


def test_default_does_not_expand_to_other_cards():
    assert scheduler.select_idle_gpu([card(2), card(3)]) is None


def test_foreign_ownership_excludes_only_own_family():
    assert scheduler.foreign_pids([card(compute_pids=[100, 101, 500])], 0, {100, 101}) == [500]
    with pytest.raises(RuntimeError, match='disappeared'):
        scheduler.foreign_pids([card(1)], 0, {100})


def test_six_runs_paired_by_seed_and_budget_fail_closed():
    config = {'variants': ['AAA', 'QFL_DFL'], 'seeds': [42, 43, 44], 'epochs': 30, 'max_formal_runs': 6}
    assert scheduler.run_order(config) == [('AAA', 42), ('QFL_DFL', 42), ('AAA', 43),
                                           ('QFL_DFL', 43), ('AAA', 44), ('QFL_DFL', 44)]
    with pytest.raises(ValueError):
        scheduler.run_order({**config, 'max_formal_runs': 7})


def test_checkpoint_commit_is_authoritative_after_crash(tmp_path):
    import torch
    run = tmp_path / 'output/runs/AAA_seed42'
    run.mkdir(parents=True)
    (run / 'status.json').write_text(json.dumps({'status': 'TRAINING', 'completed_epochs': 1}))
    cp = tmp_path / 'checkpoints/AAA_seed42'
    cp.mkdir(parents=True)
    torch.save({'epoch': 1, 'history': [{}, {}],
                'binding': {'variant': 'AAA', 'seed': 42, 'source_binding': {'frozen': True}}}, cp / 'last.pt')
    # Crash after atomically committing epoch 2, before status/history sidecars update.
    progress = scheduler.run_progress(tmp_path / 'output', 'AAA', 42, tmp_path / 'checkpoints', {'frozen': True})
    assert progress['completed_epochs'] == 2
    with pytest.raises(ValueError, match='source binding'):
        scheduler.run_progress(tmp_path / 'output', 'AAA', 42, tmp_path / 'checkpoints', {'frozen': False})


def test_progress_rejects_uncommitted_status(tmp_path):
    run = tmp_path / 'output/runs/AAA_seed42'
    run.mkdir(parents=True)
    (run / 'status.json').write_text(json.dumps({'status': 'PARTIAL', 'completed_epochs': 2}))
    with pytest.raises(ValueError, match='without a committed checkpoint'):
        scheduler.run_progress(tmp_path / 'output', 'AAA', 42, tmp_path / 'checkpoints')


def test_progress_rejects_early_complete(tmp_path):
    run = tmp_path / 'runs/AAA_seed42'
    run.mkdir(parents=True)
    (run / 'status.json').write_text(json.dumps({'status': 'COMPLETE', 'completed_epochs': 29}))
    with pytest.raises(ValueError, match='thirty'):
        scheduler.run_progress(tmp_path, 'AAA', 42)


def test_remote_preflight_script_and_nested_supervisor_compile(monkeypatch, tmp_path):
    captured = []
    monkeypatch.setattr(remote, 'LOCAL', tmp_path)
    monkeypatch.setattr(remote, 'execute_json', lambda code, timeout=60: captured.append(code) or {'status': 'MOCK'})
    remote.preflight()
    script = captured[0]
    compile(script, '<remote-preflight>', 'exec')
    tree = ast.parse(script)
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'supervisor' for t in node.targets))
    expression = ast.Expression(assignment.value)
    scope = {'root': Path('/isolated'), 'result': Path('/isolated/results/test'),
             'commands': [['python', '-m', 'pytest', '-q']]}
    supervisor = eval(compile(ast.fix_missing_locations(expression), '<supervisor-builder>', 'eval'), scope)
    compile(supervisor, '<supervisor>', 'exec')
    assert '/isolated/results/test' not in supervisor  # RESULT is deliberately root-relative.
    assert 'subprocess.run(command,check=True,timeout=1800)' in supervisor


def test_remote_launch_script_compiles_and_has_all_gates(monkeypatch, tmp_path):
    captured = []
    monkeypatch.setattr(remote, 'LOCAL', tmp_path)
    monkeypatch.setattr(remote, 'execute_json', lambda code, timeout=60: captured.append(code) or {'status': 'MOCK'})
    remote.launch()
    compile(captured[0], '<remote-launch>', 'exec')
    for name in ('preflight_complete.json', 'smoke_gate.json', 'independent_review.json'):
        assert name in captured[0]
    assert '30*1024**3' in captured[0]


def test_singleton_on_linux(tmp_path):
    pytest.importorskip('fcntl')
    lock = tmp_path / 'queue.lock'
    with scheduler.singleton(lock):
        with pytest.raises(RuntimeError, match='already owns'):
            with scheduler.singleton(lock):
                pass
    with scheduler.singleton(lock):
        pass
