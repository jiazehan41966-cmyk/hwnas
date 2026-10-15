"""Numerical and contract tests; synthetic detections are not experiment evidence."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.sonar_head_factorial import stable_analysis as analysis


def record(name, cls, detections):
    return dict(image=name,target=dict(boxes=[[0.,0.,10.,10.]],labels=[cls]),
        prediction=dict(boxes=[d[0] for d in detections],scores=[d[1] for d in detections],
                        labels=[d[2] for d in detections]))


def example_records():
    # Deliberately include score ties, a TP/FP tie, cross-class predictions,
    # duplicate matching candidates, and an exact IoU=0.5 boundary.
    return [record('a.jpg',0,[([20.,20.,30.,30.],.8,0),([0.,0.,10.,10.],.8,0),
                              ([0.,0.,10.,10.],.8,1),([0.,0.,10.,5.],.4,0)]),
            record('b.jpg',1,[([0.,0.,10.,10.],.8,1),([20.,20.,30.,30.],.8,1)]),
            record('c.jpg',0,[([0.,0.,10.,10.],.7,0)]),
            record('d.jpg',1,[])]


def synthetic_rows():
    rows=[]
    for backbone,seeds in analysis.SEEDS.items():
        for head in analysis.HEADS:
            for seed in seeds:
                rows.append(dict(binding=dict(backbone=backbone,head=head,seed=seed,
                    environment_fingerprint='common',execution_manifest_sha256='frozen'),
                    epochs=80,formal_eligible=True,primary_checkpoint='mean_epochs_76_80',
                    stable_prediction_paths={str(e):f'{backbone}_{head}_{seed}_{e}.json' for e in analysis.EPOCHS},
                    stable_prediction_sha256={str(e):'unused' for e in analysis.EPOCHS},
                    history=[dict(epoch=e,valid=dict(mAP50_95=.2)) for e in range(1,81)]))
    return rows


def contract():
    names=[f'image_{i:03d}.jpg' for i in range(99)]
    return dict(full_images=names,kept_images=names[10:],excluded_images=names[:10],
                selection_uses_model_predictions=False,selection_uses_performance=False)


class ExactAPTests(unittest.TestCase):
    def test_original_evaluator_parity_with_ties_and_repeated_images(self):
        records=example_records()
        prepared=analysis.PreparedAP(records)
        self.assertTrue(any(c['has_score_ties'] for c in prepared.classes))
        checks=prepared.validate(dict(full=np.arange(4),repeat=[3,0,0,1],
                                      reverse=[3,2,1,0],many_copies=[0,0,0,0,0,1,1]))
        self.assertTrue(all(c['passed'] for c in checks))
        self.assertTrue(any(c['duplicate_images'] for c in checks))

    def test_absent_class_does_not_get_counted_as_zero(self):
        records=[record('a.jpg',0,[([0.,0.,10.,10.],.8,0)])]
        value=analysis.PreparedAP(records).evaluate_one([0,0,0])
        self.assertAlmostEqual(value[0],1.)
        self.assertTrue(np.isnan(value[-3:]).all())
        np.testing.assert_allclose(value,analysis.reference_metrics(records,[0,0,0]),
                                   atol=2e-6,rtol=0,equal_nan=True)

    def test_false_positives_and_empty_predictions_match_public_evaluator(self):
        records=[record('a.jpg',0,[]),record('b.jpg',1,[([20.,20.,30.,30.],.9,1)])]
        value=analysis.PreparedAP(records).evaluate_one([0,1,1])
        self.assertEqual(value[0],0.)
        np.testing.assert_allclose(value,analysis.reference_metrics(records,[0,1,1]),atol=2e-6,rtol=0)

    def test_parity_failure_is_closed(self):
        prepared=analysis.PreparedAP(example_records())
        with patch.object(prepared,'evaluate_one',return_value=np.zeros(len(analysis.METRICS))):
            with self.assertRaisesRegex(analysis.EvaluatorParityError,'parity error'):
                prepared.validate({'full':[0,1,2,3]})


class PairedBootstrapTests(unittest.TestCase):
    def test_seed_resampling_is_present_reproducible_and_not_image_only(self):
        plan=analysis.bootstrap_plan([1,3,7],(42,43,44),replicates=100,random_seed=17)
        again=analysis.bootstrap_plan([1,3,7],(42,43,44),replicates=100,random_seed=17)
        np.testing.assert_array_equal(plan['image_draws'],again['image_draws'])
        np.testing.assert_array_equal(plan['seed_draws'],again['seed_draws'])
        self.assertTrue(any(len(set(r))<3 for r in plan['seed_draws']))
        # No image uncertainty; between-training-seed heterogeneity must remain.
        values=np.broadcast_to(np.array([-1.,0.,2.])[None,:,None,None],(100,3,5,1)).copy()
        hierarchical,image_only=analysis.reduce_paired_bootstrap(values,plan['seed_draws'])
        self.assertGreater(float(hierarchical.std()),.1)
        self.assertAlmostEqual(float(image_only.std()),0.)
        for i in (0,1,2):
            self.assertAlmostEqual(hierarchical[i,0],np.array([-1.,0.,2.])[plan['seed_draws'][i]].mean())

    def test_five_epoch_then_seed_reduction(self):
        values=np.arange(2*3*5*1,dtype=float).reshape(2,3,5,1)
        seeds=np.array([[2,2,0],[0,1,1]])
        hierarchical,image_only=analysis.reduce_paired_bootstrap(values,seeds)
        for rep in range(2):
            expected=np.mean([values[rep,seed,:,0].mean() for seed in seeds[rep]])
            self.assertEqual(hierarchical[rep,0],expected)
            self.assertEqual(image_only[rep,0],values[rep,:,:,0].mean())

    def test_positive_observed_deltas_do_not_override_crossing_ci(self):
        # All observed seed effects can be positive while paired image uncertainty crosses zero.
        observed=np.array([.031,.032,.033,.034,.035])
        ci=analysis.interval(np.linspace(-.02,.08,2000))
        result=analysis.d17_decision(observed.mean(),ci)
        self.assertTrue(np.all(observed>0))
        self.assertTrue(result['threshold_met'])
        self.assertFalse(result['final_pass'])
        self.assertEqual(ci['conclusion'],'INCONCLUSIVE_NOT_EQUIVALENCE')

    def test_d17_requires_both_conditions_and_strict_ci_lower(self):
        self.assertTrue(analysis.d17_decision(.03,dict(low=.001,high=.07))['final_pass'])
        self.assertFalse(analysis.d17_decision(.029,dict(low=.001,high=.07))['final_pass'])
        self.assertFalse(analysis.d17_decision(.05,dict(low=0.,high=.09))['final_pass'])
        self.assertFalse(analysis.d17_decision(.05,dict(low=-.01,high=.09))['final_pass'])

    def test_structure_effect_and_interaction_use_only_common_paired_seeds(self):
        # Extra E0 seeds are extreme; accidentally pooling them changes every contrast.
        cells={}
        for backbone,seeds in analysis.SEEDS.items():
            for head in analysis.HEADS:
                for seed in seeds:
                    base=.2 if backbone=='E0' else .4
                    if seed in (45,46): base=1000.
                    if head=='quality_aware':base+=.03 if backbone=='E0' else .07
                    cells[backbone,head,seed]=np.full((3,5,len(analysis.METRICS)),base)
        result=analysis.common_seed_contrasts(cells,np.array([[2,2,0],[1,0,1]]))
        structure=result['structure_effects']
        self.assertAlmostEqual(structure['obj_softmax']['mAP50_95']['mean'],.2)
        self.assertAlmostEqual(structure['quality_aware']['mAP50_95']['mean'],.24)
        self.assertAlmostEqual(structure['equal_head_mean']['mAP50_95']['mean'],.22)
        self.assertAlmostEqual(result['interaction']['mAP50_95']['mean'],.04)
        self.assertEqual(result['interaction']['mAP50_95']['seeds'],[42,43,44])


class InputContractTests(unittest.TestCase):
    def test_required_sixteen_cells(self):
        self.assertEqual(len(analysis.validate_rows(synthetic_rows())),16)
        for rows in (synthetic_rows()[:-1],synthetic_rows()+[synthetic_rows()[0]]):
            with self.assertRaisesRegex(ValueError,'sixteen'):
                analysis.validate_rows(rows)

    def test_missing_epoch_and_missing_prediction_fail(self):
        for field in ('stable_prediction_paths','stable_prediction_sha256'):
            rows=synthetic_rows();del rows[0][field]['78']
            with self.assertRaisesRegex(ValueError,'epoch'):
                analysis.validate_rows(rows)
        rows=synthetic_rows();rows[0]['history'].pop(77)
        with self.assertRaisesRegex(ValueError,'history'):
            analysis.validate_rows(rows)

    def test_wrong_seed_cannot_replace_required_seed(self):
        rows=synthetic_rows();rows[0]['binding']['seed']=99
        with self.assertRaisesRegex(ValueError,'seed'):
            analysis.validate_rows(rows)

    def test_wrong_image_and_gt_are_rejected(self):
        names=contract()['full_images']
        records=[record(name,0 if i<70 else 1,[]) for i,name in enumerate(names)]
        targets={r['image']:r['target'] for r in records}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'records.json';path.write_text(json.dumps(records),encoding='utf-8')
            self.assertEqual(len(analysis.load_records(path,names,targets)),99)
            wrong=deepcopy(records);wrong[0]['image']='unexpected.jpg'
            path.write_text(json.dumps(wrong),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'image set'):
                analysis.load_records(path,names)
            wrong=deepcopy(records);wrong[0]['target']['boxes'][0][0]=.5
            path.write_text(json.dumps(wrong),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'GT differs'):
                analysis.load_records(path,names,targets)

    def test_missing_or_outcome_selected_subset_contract(self):
        self.assertEqual(len(analysis.validate_contract(contract())['kept_images']),89)
        for mutate in ('missing','overlap','performance'):
            d=contract()
            if mutate=='missing': del d['kept_images']
            if mutate=='overlap':d['kept_images']=d['full_images'][:89]
            if mutate=='performance':d['selection_uses_performance']=True
            with self.assertRaises(ValueError):
                analysis.validate_contract(d)

    def test_source_hash_drift_rejected_before_statistics(self):
        rows=synthetic_rows()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'records.json';path.write_text('[]',encoding='utf-8')
            rows[0]['stable_prediction_paths']['76']=str(path)
            with self.assertRaisesRegex(ValueError,'source hash drift'):
                analysis.factorial_summary(rows,dict(metric='mAP50_95',threshold_absolute=.03,
                    bootstrap=dict(replicates=2)),subset_contract=contract())


if __name__=='__main__':
    unittest.main(verbosity=2)
