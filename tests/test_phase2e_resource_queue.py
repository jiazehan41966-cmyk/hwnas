"""Resource scheduling tests; no model, GPU, or dataset access."""
import importlib.util
from pathlib import Path
import threading
import time
import unittest

spec = importlib.util.spec_from_file_location('resource_queue',
    Path(__file__).resolve().parents[1] / 'tools/sonar_spec/phase2e_resource_queue.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ResourceQueueTests(unittest.TestCase):
    def test_four_slots_exactly_once_and_correct_gpu(self):
        barrier = threading.Barrier(4)
        active, seen, finished = {}, [], []
        peaks = {0: 0, 1: 0}

        def start(key, gpu, aid):
            self.assertNotIn(aid, active.values())
            active[key] = aid
            seen.append(aid)
            count = sum(k.startswith(f'{gpu}:') for k in active)
            peaks[gpu] = max(peaks[gpu], count)
            self.assertLessEqual(count, 2)

        def execute(aid, gpu):
            barrier.wait(timeout=5)
            time.sleep(.01)
            return aid, gpu

        def finish(key, gpu, aid, row):
            self.assertEqual(row, (aid, gpu))
            self.assertEqual(active.pop(key), aid)
            finished.append(aid)

        ids = [f'architecture_{i}' for i in range(12)]
        module.run_queue(ids, [0, 1], 2, execute, start, finish)
        self.assertCountEqual(seen, ids)
        self.assertCountEqual(finished, ids)
        self.assertEqual(peaks, {0: 2, 1: 2})
        self.assertEqual(active, {})

    def test_duplicate_architecture_rejected_before_execution(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate architecture'):
            module.run_queue(['a', 'a'], [0, 1], 2, None, None, None)

    def test_unsupported_parallelism_rejected(self):
        for gpus, workers in [([0, 0], 2), ([], 2), ([0, 1], 3)]:
            with self.assertRaises(ValueError):
                module.run_queue([], gpus, workers, None, None, None)

    def test_failure_stops_pending_work(self):
        started = []

        def fail(aid, gpu):
            raise RuntimeError('training failed')

        with self.assertRaisesRegex(RuntimeError, 'training failed'):
            module.run_queue(list(range(100)), [0, 1], 2, fail,
                lambda key, gpu, aid: started.append(aid), lambda *args: None)
        self.assertLessEqual(len(started), 4)


if __name__ == '__main__':
    unittest.main()
