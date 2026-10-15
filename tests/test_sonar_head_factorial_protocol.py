"""Fail-closed training/archive gates; synthetic data make no performance claim."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.sonar_head_factorial import train as runner
import torch
import yaml
TMP=ROOT/'results/sonar_head_factorial_20261012_02_stable/protocol_test_tmp'
IMAGES=[f'image_{i:03d}.png' for i in range(99)]


def raw_metrics(value=.2):
    ap={f'{q/100:.2f}':value for q in range(50,96,5)}
    return dict(mAP50_95=value,AP50=value,AP75=value,Precision=value,Recall=value,num_images=99,
                coco_details=dict(per_class_ap=dict(all={label:deepcopy(ap) for label in runner.p.CLASS_NAMES})))


def metrics(value=.2):
    return runner.enrich_metrics(raw_metrics(value))


def history(seed=42,length=80,value=.2):
    return [dict(epoch=e,valid=metrics(value),batches=83,images=330,
                 augmented_batches_sha256=f'pixels-{seed}-{e}',image_order_sha256=f'order-{seed}-{e}',
                 valid_batches_sha256='valid') for e in range(1,length+1)]


def records():
    return [dict(image='valid/images/'+image,prediction=dict(boxes=[],labels=[],scores=[]),
                 target=dict(boxes=[[1.,2.,3.,4.]],labels=[int(i>=70)])) for i,image in enumerate(IMAGES)]


def binding(backbone='E0',head='obj_softmax',seed=42):
    return dict(backbone=backbone,head=head,seed=seed,mode='formal',execution_manifest_sha256='source',
                environment_fingerprint='environment',subset_contract_sha256='subset',
                validation_target_sha256=runner.p.canonical_sha([dict(image=r['image'].split('/')[-1],target=r['target']) for r in records()]),test_evaluated=False)


def synthetic_rows():
    rows=[]
    for b,h,s in runner.queue_jobs():
        hist=history(seed=s);stable=runner.stable_metrics_from_history(hist)
        rows.append(dict(binding=binding(b,h,s),epochs=80,primary_checkpoint='mean_epochs_76_80',
            formal_eligible=True,initialization=dict(backbone_sha256=f'b-{b}-{s}',neck_sha256=f'n-{b}-{s}'),
            history=hist,metrics=stable,stable_metrics=deepcopy(stable),secondary_best_metrics=metrics(),
            stable_prediction_paths={str(e):f'epoch{e}.json' for e in runner.STABLE_EPOCHS},
            stable_prediction_sha256={str(e):f'sha{e}' for e in runner.STABLE_EPOCHS}))
    return rows


class FactorialProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        TMP.mkdir(parents=True,exist_ok=True)

    def temporary(self):
        return tempfile.TemporaryDirectory(dir=TMP)

    def archive(self,folder,length=80):
        hist=history(length=length)
        for row in hist:
            if row['epoch'] in runner.STABLE_EPOCHS:
                row['prediction_archive']=runner.save_stable_predictions(folder,row['epoch'],records(),binding(),IMAGES)
        return hist

    def test_exact_fixed_sixteen_cells_and_shards(self):
        jobs=runner.queue_jobs()
        expected={(b,h,s) for b in runner.BACKBONES for h in runner.HEADS for s in ((42,43,44,45,46) if b=='E0' else (42,43,44))}
        self.assertEqual(len(jobs),16);self.assertEqual(set(jobs),expected)
        first,second=runner.queue_jobs(0,2),runner.queue_jobs(1,2)
        self.assertEqual(set(first)|set(second),expected);self.assertFalse(set(first)&set(second))
        for shard in (first,second,jobs):
            for b,h,s in shard:
                self.assertEqual({x[1] for x in shard if x[0]==b and x[2]==s},set(runner.HEADS))
        with self.assertRaises(ValueError):runner.queue_jobs(2,2)
        self.assertFalse(hasattr(runner,'extension_required'))

    def test_registered_config_and_source_closure(self):
        cfg,_=runner.load_config();files=runner.source_paths(cfg)
        self.assertTrue(all(p.is_file() for p in files))
        for rel in (cfg['subset_contract'],'tools/sonar_head_factorial/stable_analysis.py','tests/test_sonar_head_factorial_stable_analysis.py'):
            self.assertIn(ROOT/rel,files)
        self.assertEqual(cfg['total_formal_runs'],16)
        self.assertNotIn('extension',cfg['decision'])
        self.assertEqual(cfg['decision']['primary_subset'],'full99')
        self.assertFalse(cfg['decision']['automatic_search']);self.assertFalse(cfg['decision']['automatic_promotion'])
        self.assertEqual(cfg['decision']['D01_evidence_source'],'experiment4_only')

    def test_old_metric_terminal_or_adaptive_protocol_rejected(self):
        original=yaml.safe_load(runner.CONFIG.read_text(encoding='utf-8'))
        cases=[('metric','map'),('primary_checkpoint','terminal_epoch_80'),('positivity','all_seeds_positive'),('extension',{'seeds':[45,46]})]
        with self.temporary() as directory:
            path=Path(directory)/'config.yaml'
            for key,value in cases:
                changed=deepcopy(original);changed['decision'][key]=value
                path.write_text(yaml.safe_dump(changed),encoding='utf-8')
                with self.subTest(key=key),self.assertRaisesRegex(ValueError,'Decision metric'):runner.load_config(path)
            changed=deepcopy(original);changed['seeds_by_backbone']['YOLOv9_T'] += [45,46]
            path.write_text(yaml.safe_dump(changed),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Unregistered factorial'):runner.load_config(path)

    def test_wrong_subset_content_or_membership_rejected(self):
        cfg,_=runner.load_config();contract=runner.load_subset_contract(cfg)
        with self.temporary() as directory:
            path=Path(directory)/'subset.json';runner.p.dump(path,contract)
            altered={**cfg,'subset_contract':str(path.relative_to(ROOT)),'subset_contract_sha256':runner.p.sha(path)}
            self.assertEqual(runner.load_subset_contract(altered)['full_images'],contract['full_images'])
            bad=deepcopy(contract);bad['kept_images'][0]=bad['kept_images'][1];runner.p.dump(path,bad)
            with self.assertRaisesRegex(ValueError,'contract hash drift'):runner.load_subset_contract(altered)
            altered['subset_contract_sha256']=runner.p.sha(path)
            with self.assertRaisesRegex(ValueError,'image membership'):runner.load_subset_contract(altered)
        inputs=dict(datasets=dict(valid=dict(entries=[dict(image='valid/images/unrelated.png')])) )
        with self.assertRaisesRegex(ValueError,'differs from the frozen validation'):runner.load_subset_contract(cfg,inputs)

    def test_stable_average_exact_five_late_epochs_not_best_or_terminal(self):
        hist=history(value=.9)
        for e,value in zip(runner.STABLE_EPOCHS,[.1,.2,.3,.4,.5]):hist[e-1]['valid']=metrics(value)
        stable=runner.stable_metrics_from_history(hist)
        self.assertAlmostEqual(stable['mAP50_95'],.3)
        self.assertAlmostEqual(stable['per_class_AP']['cylider']['mAP50_95'],.3)
        self.assertNotEqual(stable['mAP50_95'],hist[-1]['valid']['mAP50_95'])
        self.assertNotEqual(stable['mAP50_95'],max(r['valid']['mAP50_95'] for r in hist))
        with self.assertRaisesRegex(ValueError,'history'):runner.stable_metrics_from_history(hist[:-1])

    def test_missing_duplicate_or_extra_seed_rejected(self):
        rows=synthetic_rows();runner.validate_formal_rows(rows)
        for bad in (rows[:-1],rows[:-1]+[rows[0]],rows+[deepcopy(rows[-1])]):
            with self.assertRaisesRegex(ValueError,'sixteen'):runner.validate_formal_rows(bad)
        rows=synthetic_rows();rows[-1]['binding']['backbone']='YOLOv9_T'
        with self.assertRaisesRegex(ValueError,'sixteen'):runner.validate_formal_rows(rows)

    def test_rows_fail_on_late_omission_wrong_primary_or_changed_inputs(self):
        for mutation,message in [('missing','Every stable epoch'),('terminal','five stable'),('mean','arithmetic mean'),('source','identity differs')]:
            rows=synthetic_rows()
            if mutation=='missing':del rows[0]['stable_prediction_paths']['78']
            elif mutation=='terminal':rows[0]['primary_checkpoint']='terminal_epoch_80'
            elif mutation=='mean':
                rows[0]['metrics']=deepcopy(rows[0]['metrics']);rows[0]['metrics']['mAP50_95'] += .1
            else:rows[0]['binding']['subset_contract_sha256']='different'
            with self.subTest(mutation=mutation),self.assertRaisesRegex(ValueError,message):runner.validate_formal_rows(rows)

    def test_pairing_four_common_cells_and_two_extra_E0_cells(self):
        rows=synthetic_rows();runner.verify_pairing(rows)
        for key in ('backbone_sha256','neck_sha256'):
            bad=deepcopy(rows);bad[-1]['initialization'][key]='different'
            with self.assertRaisesRegex(ValueError,'between heads'):runner.verify_pairing(bad)
        rows[-1]['history'][7]['augmented_batches_sha256']='changed'
        with self.assertRaisesRegex(ValueError,'pixels/labels/order'):runner.verify_pairing(rows)

    def test_records_require_exact_99_unique_validation_images(self):
        values=records();runner.validate_prediction_records(values,IMAGES,binding()['validation_target_sha256'])
        for bad in (values[:-1],values[:-1]+[deepcopy(values[0])]):
            with self.assertRaisesRegex(ValueError,'99'):runner.validate_prediction_records(bad,IMAGES,binding()['validation_target_sha256'])
        bad=deepcopy(values);bad[0]['image']=bad[0]['image'].replace('valid/','train/')
        with self.assertRaisesRegex(ValueError,'validation split'):runner.validate_prediction_records(bad,IMAGES,binding()['validation_target_sha256'])

    def test_saved_ground_truth_must_match_source_reference(self):
        for change in ('missing','shifted','class'):
            bad=records()
            if change=='missing':bad[0]['target']=dict(boxes=[],labels=[])
            elif change=='shifted':bad[0]['target']['boxes'][0][0] += .5
            else:bad[0]['target']['labels'][0]=1
            with self.subTest(change=change),self.assertRaisesRegex(ValueError,'frozen source-label'):
                runner.validate_prediction_records(bad,IMAGES,binding()['validation_target_sha256'])

    def test_archive_missing_round_or_byte_tamper_fails_closed(self):
        with self.temporary() as directory,patch.object(runner.p,'metrics_from_records',return_value=raw_metrics()):
            folder=Path(directory);hist=self.archive(folder)
            verified=runner.verify_stable_predictions(folder,hist,binding(),{},IMAGES)
            self.assertEqual(set(verified['paths']),{'76','77','78','79','80'})
            archive=hist[77]['prediction_archive'];path=folder/archive['records'];original=path.read_bytes()
            path.write_bytes(original+b' ')
            with self.assertRaisesRegex(ValueError,'hash drift'):runner.verify_stable_predictions(folder,hist,binding(),{},IMAGES)
            path.write_bytes(original);del hist[77]['prediction_archive']
            with self.assertRaisesRegex(ValueError,'epoch 78'):runner.verify_stable_predictions(folder,hist,binding(),{},IMAGES)

    def test_resume_verifies_prior_archives_and_repairs_only_exact_log_prefix(self):
        with self.temporary() as directory,patch.object(runner.p,'metrics_from_records',return_value=raw_metrics()):
            folder=Path(directory);hist=self.archive(folder,78)
            before={p.name:runner.p.sha(p) for p in (folder/'epoch_predictions').glob('*.json')}
            runner.p.dump(folder/'epoch_log.json',hist[:77])
            state=dict(binding=binding(),epoch=78,history=hist,initial_state_sha256='initial',initialization={'seed':42})
            start,recovered=runner.validate_resume_state(folder,state,binding(),'initial',{'seed':42},{},IMAGES)
            self.assertEqual(start,78);self.assertEqual(recovered,hist);self.assertEqual(runner.read_json(folder/'epoch_log.json'),hist)
            self.assertEqual(before,{p.name:runner.p.sha(p) for p in (folder/'epoch_predictions').glob('*.json')})
            runner.save_stable_predictions(folder,79,records(),binding(),IMAGES)
            self.assertTrue(all(runner.p.sha(folder/'epoch_predictions'/name)==value for name,value in before.items()))
            bad=deepcopy(hist);bad[0]['valid']['mAP50_95'] += .1;runner.p.dump(folder/'epoch_log.json',bad)
            with self.assertRaisesRegex(ValueError,'contradicts'):runner.validate_resume_state(folder,state,binding(),'initial',{'seed':42},{},IMAGES)

    def test_resume_rejects_missing_committed_predictions_and_changed_binding(self):
        with self.temporary() as directory,patch.object(runner.p,'metrics_from_records',return_value=raw_metrics()):
            folder=Path(directory);hist=self.archive(folder,76)
            state=dict(binding=binding(),epoch=76,history=hist,initial_state_sha256='initial',initialization={'seed':42})
            changed=deepcopy(binding());changed['subset_contract_sha256']='changed'
            with self.assertRaisesRegex(ValueError,'identity drift'):runner.validate_resume_state(folder,state,changed,'initial',{'seed':42},{},IMAGES)
            (folder/hist[-1]['prediction_archive']['records']).unlink()
            with self.assertRaisesRegex(ValueError,'missing file'):runner.validate_resume_state(folder,state,binding(),'initial',{'seed':42},{},IMAGES)

    def test_existing_uncommitted_epoch_is_never_overwritten(self):
        with self.temporary() as directory:
            folder=Path(directory);archive=runner.save_stable_predictions(folder,76,records(),binding(),IMAGES)
            original=runner.p.sha(folder/archive['records'])
            self.assertEqual(archive,runner.save_stable_predictions(folder,76,records(),binding(),IMAGES))
            different=records();different[0]['prediction']=dict(boxes=[[1.,2.,3.,4.]],labels=[0],scores=[.9])
            with self.assertRaisesRegex(ValueError,'preserve uncommitted'):runner.save_stable_predictions(folder,76,different,binding(),IMAGES)
            self.assertEqual(original,runner.p.sha(folder/archive['records']))

    def test_batch_digest_binds_pixels_labels_and_order(self):
        images=torch.zeros((2,1,4,4))
        targets=[dict(path=f'/stage/v6_yolov8/train/images/{i}.png',boxes=torch.tensor([[0.,0.,2.,2.]]),labels=torch.tensor([i])) for i in range(2)]
        original=runner.batch_digest(images,targets);moved=deepcopy(targets)
        for target in moved:target['path']=target['path'].replace('/stage/','/another/')
        self.assertEqual(original,runner.batch_digest(images,moved))
        changed=images.clone();changed[0,0,0,0]=1.
        self.assertNotEqual(original,runner.batch_digest(changed,targets))
        moved[0]['labels'][0]=1
        self.assertNotEqual(original,runner.batch_digest(images,moved))
        self.assertNotEqual(original,runner.batch_digest(images,targets[::-1]))


if __name__=='__main__':
    unittest.main(verbosity=2)
