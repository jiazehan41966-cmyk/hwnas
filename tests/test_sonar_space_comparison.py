"""Protocol regression witnesses executed only in the pinned remote CUDA runtime."""
import json
import random
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import torch
from tools.sonar_space_comparison import campaign as c


class ComparisonProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        c.require_remote_cuda()
        cls.manifest=c.manifest()

    def test_all_seeds_common_architecture_tensor_parity(self):
        for seed in (42,43,44):
            for kernel in (3,5):
                a=dict(pan=False,td_kernels=[kernel]*3,pan_kernels=[])
                b=dict(backbone=dict(stage1=dict(kernel=3,downsample='strided_dw'),stage2=dict(kernel=5)),
                    neck=dict(td_ops={p:f'mbconv_k{kernel}' for p in ('P2','P3','P4')},pan=False))
                left=c.build(c.SPACES[0],a,seed).state_dict();right=c.build(c.SPACES[1],b,seed).state_dict()
                self.assertEqual(set(left),set(right))
                for key in left:
                    self.assertTrue(torch.equal(left[key],right[key]),f'{seed}/{kernel}/{key}')

    def test_all_architectures_and_feasible_costs(self):
        for space,count in zip(c.SPACES,(72,192)):
            inventory=self.manifest['inventory'][space]
            self.assertEqual(len(inventory),count)
            self.assertEqual(len({r['architecture_sha256'] for r in inventory}),count)
            for seed in (101,202,303):
                rows=self.manifest['random_orders'][f'{space}/search_{seed}']
                self.assertEqual(len(rows),self.manifest['eligible_counts'][space])
                self.assertTrue(all(r['MACs']<=self.manifest['MAC_upper_bound'] for r in rows))
                self.assertEqual(len(rows),len({r['architecture_sha256'] for r in rows}))

    def test_random_orders_reconstruct_without_training_RNG(self):
        for space in c.SPACES:
            base=[r for r in self.manifest['inventory'][space] if r['MACs']<=self.manifest['MAC_upper_bound']]
            for seed in (101,202,303):
                rows=base.copy();before=random.getstate();random.Random(seed).shuffle(rows)
                self.assertEqual(random.getstate(),before)
                self.assertEqual(rows,self.manifest['random_orders'][f'{space}/search_{seed}'])

    def test_blur_cost_not_omitted(self):
        for row in self.manifest['inventory'][c.SPACES[1]]:
            expected=11059200 if row['architecture']['backbone']['stage1']['downsample']=='blur3_after_dense_dw' else 0
            self.assertEqual(row['fixed_blur_MACs'],expected)
            self.assertEqual(row['MACs'],row['conv_module_MACs']+expected)

    def test_matched_missing_and_deterministic_ties(self):
        lo,hi=self.manifest['matched_MAC_band']
        rows=[dict(mAP50_95=.5,MACs=hi+10,architecture_id='b'),dict(mAP50_95=.5,MACs=hi+1,architecture_id='a')]
        selected=c.select(rows,self.manifest)
        self.assertIsNone(selected['matched']);self.assertEqual(selected['overall']['architecture_id'],'a')
        rows.append(dict(mAP50_95=.3,MACs=(lo+hi)/2,architecture_id='c'))
        self.assertEqual(c.select(rows,self.manifest)['matched']['architecture_id'],'c')

    def test_deadline_excludes_unfinished_batch(self):
        with self.assertRaises(c.BudgetExpired):next(iter(c.DeadlineLoader([1,2],-float('inf'))))

    def test_latency_path_and_reference_cost(self):
        row=next(r for r in self.manifest['inventory'][c.SPACES[0]] if r['architecture']==dict(pan=False,td_kernels=[3]*3,pan_kernels=[]))
        model=c.build(c.SPACES[0],row['architecture'],42).to('cuda')
        cost=c.measure_cost(model);self.assertEqual(cost['MACs'],row['MACs'])
        baseline=self.manifest['baseline'];loader=c.frozen.make_loader(baseline,ROOT/baseline['dataset']['data_dir'],'valid',42,0)
        measured=c.latency(model,loader,baseline)
        self.assertEqual(set(measured['measurements']),{'model_forward','model_plus_decode_NMS'})
        self.assertTrue(all(v['median_ms']>0 and v['repetitions']==100 for v in measured['measurements'].values()))


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ComparisonProtocolTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    c.dump(c.RESULT/'protocol_tests.json',dict(status='PASS' if result.wasSuccessful() else 'FAIL',
        tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        execution_manifest_sha256=c.sha(c.RESULT/'execution_manifest.json'),
        test_evaluated=False,updated_at=c.now()))
    sys.exit(0 if result.wasSuccessful() else 1)
