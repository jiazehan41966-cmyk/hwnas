"""Independent numerical witnesses for contrast, spectrum, and frozen metrics."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).parent))
from complete_analysis import contrast_region,nearby_background,frequency_energy
from baseline import precision_recall,metrics_from_records,load_config,ROOT


def test_ring_clips_and_excludes_other_gt():
    array=np.ones((10,10))*.2
    array[:4,:4]=.8
    array[:4,4:6]=1.
    row=contrast_region(array,(0,0,4,4),[(0,0,4,4),(4,0,6,4)],2.,1e-6)
    assert row['target_pixels']==16
    assert row['background_pixels']==12
    assert row['target_mean']==pytest.approx(.8)
    assert row['background_mean']==pytest.approx(.2)
    assert row['contrast']==pytest.approx(600000)


def test_no_background_is_explicit():
    row=contrast_region(np.ones((8,8)),(0,0,8,8),[(0,0,8,8)])
    assert row['status']=='insufficient_background' and row['contrast'] is None
    crop,bounds=nearby_background(np.ones((8,8)),(0,0,8,8),[(0,0,8,8)])
    assert crop is None and bounds is None


def test_fft_constant_and_sinusoid():
    assert frequency_energy(np.ones((32,32)))['status']=='zero_ac_energy'
    x=np.arange(64)
    low=frequency_energy(np.tile(np.sin(2*np.pi*x/32),(64,1)))
    high=frequency_energy((-1.)**np.indices((64,64)).sum(axis=0))
    assert low['low']>.99 and high['high']>.99
    assert low['low']+low['middle']+low['high']==pytest.approx(1.)


def test_precision_counts_absent_class_false_positive():
    result=precision_recall([{'boxes':[[0,0,10,10]],'scores':[.9],'labels':[1]}],
                            [{'boxes':[[0,0,10,10]],'labels':[0]}],.5)
    assert result['false_positives']==1 and result['Recall']==0


def fixture_records():
    return [
        {'prediction':{'boxes':[[0,0,10,10],[0,0,10,10],[20,20,30,30]],'labels':[0,0,1],'scores':[.95,.8,.9]},
         'target':{'boxes':[[0,0,10,10],[20,20,30,30]],'labels':[0,1]}},
        {'prediction':{'boxes':[[1,0,11,10],[40,40,50,50]],'labels':[0,1],'scores':[.75,.7]},
         'target':{'boxes':[[0,0,10,10]],'labels':[0]}},
        {'prediction':{'boxes':[],'labels':[],'scores':[]},'target':{'boxes':[[0,0,8,8]],'labels':[1]}}
    ]


def official_metrics(records):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt=COCO();annotations=[];predictions=[]
    for i,r in enumerate(records):
        for b,c in zip(r['target']['boxes'],r['target']['labels']):
            x,y,x2,y2=b
            annotations.append(dict(id=len(annotations)+1,image_id=i,category_id=c,bbox=[x,y,x2-x,y2-y],area=(x2-x)*(y2-y),iscrowd=0))
        for b,c,s in zip(r['prediction']['boxes'],r['prediction']['labels'],r['prediction']['scores']):
            x,y,x2,y2=b
            predictions.append(dict(image_id=i,category_id=c,bbox=[x,y,x2-x,y2-y],score=s))
    gt.dataset={'info':{},'images':[{'id':i} for i in range(len(records))],
                'annotations':annotations,'categories':[{'id':i,'name':str(i)} for i in range(2)]}
    gt.createIndex();dt=gt.loadRes(predictions)
    ev=COCOeval(gt,dt,'bbox');ev.params.maxDets=[1,10,100]
    ev.evaluate();ev.accumulate();ev.summarize()
    return dict(mAP50_95=float(ev.stats[0]),AP50=float(ev.stats[1]),AP75=float(ev.stats[2]))


def test_metric_parity_with_pycocotools():
    cfg=load_config(str(ROOT/'configs/baseline_sonar_frozen.yaml'))
    records=fixture_records()
    actual=metrics_from_records(records,cfg);expected=official_metrics(records)
    for key in expected:
        assert actual[key]==pytest.approx(expected[key],abs=1e-6)
    assert actual['mAP50_95']!=actual['AP50']
    assert cfg['evaluation_protocol']['development_splits']==['train','valid']


def test_test_loader_rejected_before_io():
    from baseline import make_loader
    with pytest.raises(ValueError,match='test is forbidden'):
        make_loader({},'/nonexistent','test',42,0)
