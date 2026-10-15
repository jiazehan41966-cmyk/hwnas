"""Static Phase2F protocol and queue tests; no dataset or GPU required."""
import importlib.util
from pathlib import Path
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT/'tools/sonar_spec'))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixed_architecture_set_and_pairs():
    runner = load_module('phase2f_runner_test', ROOT/'tools/sonar_spec/phase2f_runner.py')
    items = {label: aid for label, aid, _ in runner.ARCHITECTURES}
    assert list(items) == ['A1', 'A2', 'B1', 'B2', 'C', 'D']
    assert len(set(items.values())) == 6
    assert items['A1'].replace('s7k5', 's7k3') == items['A2']
    assert items['B1'].replace('P2_P3_P4_P5', 'P3_P4_P5') == items['B2']


def test_queue_runs_twelve_unique_jobs_with_two_slots_per_gpu(monkeypatch):
    pipeline = load_module('phase2f_pipeline_test', ROOT/'tools/sonar_spec/phase2f_pipeline.py')
    jobs = [(f'a{i}', seed) for i in range(6) for seed in (43, 44)]
    barrier = threading.Barrier(4); active={}; peaks={0:0,1:0}; finished=[]
    def start(key,gpu,aid,seed):
        active[key]=(aid,seed); peaks[gpu]=max(peaks[gpu],sum(k.startswith(f'{gpu}:') for k in active))
    def execute(aid,seed,gpu):
        barrier.wait(timeout=5); time.sleep(.01); return aid,seed,gpu
    def finish(key,gpu,aid,seed,row):
        assert row==(aid,seed,gpu); active.pop(key); finished.append((aid,seed))
    pipeline.run_queue(jobs,[0,1],2,execute,start,finish)
    assert sorted(finished)==sorted(jobs) and peaks=={0:2,1:2}


def test_duplicate_job_and_bad_parallelism_rejected():
    pipeline = load_module('phase2f_pipeline_reject_test', ROOT/'tools/sonar_spec/phase2f_pipeline.py')
    for jobs,gpus,workers in [([('a',43),('a',43)],[0,1],2),([],[],2),([],[0,0],2),([],[0,1],3)]:
        try:
            pipeline.run_queue(jobs,gpus,workers,None,None,None)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid Phase2F queue accepted')
