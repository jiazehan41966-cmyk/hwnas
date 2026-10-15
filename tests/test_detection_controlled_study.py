from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from detection_controlled_study import classify_image, make_loader, seed_all


def test_disjoint_errors_and_one_to_one_matching():
    r={"prediction":{"boxes":[[0,0,10,10],[0,0,10,10],[20,20,30,30],[40,40,50,50]],
                     "labels":[0,0,0,1],"scores":[.9,.8,.7,.6]},
       "target":{"boxes":[[0,0,10,10],[20,20,30,30]],"labels":[0,1]}}
    pred,gt=classify_image(r,.5)
    assert [p["kind"] for p in pred]==["true_positive","duplicate","classification","background"]
    assert sum(t["detected"] for t in gt)==1
    assert len(pred)==4


def test_localization_vs_complete_miss_and_score_filter():
    r={"prediction":{"boxes":[[5,0,15,10]],"labels":[0],"scores":[.4]},
       "target":{"boxes":[[0,0,10,10]],"labels":[0]}}
    pred,gt=classify_image(r,.3)
    assert pred[0]["kind"]=="localization"
    assert gt[0]["reason"]=="localization"
    pred,gt=classify_image(r,.5)
    assert pred==[]
    assert gt[0]["reason"]=="miss_or_wrong_class_or_duplicate_conflict"


def test_empty_ground_truth_is_all_background():
    r={"prediction":{"boxes":[[0,0,10,10]],"labels":[0],"scores":[.9]},
       "target":{"boxes":[],"labels":[]}}
    pred,gt=classify_image(r,.3)
    assert pred[0]["kind"]=="background" and not gt


def test_test_split_is_rejected_before_filesystem_access():
    with pytest.raises(ValueError,match="test is forbidden"):
        make_loader({},"missing", "test",42,0)


def test_seed_controls_initialization_and_numpy():
    import numpy as np
    import torch
    seed_all(43)
    a=torch.nn.Linear(3,2).weight.detach().clone()
    x=np.random.rand(3)
    seed_all(43)
    assert torch.equal(a,torch.nn.Linear(3,2).weight)
    assert np.array_equal(x,np.random.rand(3))
