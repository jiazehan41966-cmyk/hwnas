"""Standalone migration checks. No legacy checkout, dataset or training job needed."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import pytest
import torch
import yaml
from PIL import Image
from models import build_model
from space import Genotype
from train import DenseAnchorDetectionLoss,loss_config_for_model,metrics_from_records
from train.data import YoloDetectionDataset,make_loader,bind_dataset,prepare_calibration
from train.training import _checkpoint,load_bound_checkpoint

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=json.loads((ROOT/"models/MIGRATION.json").read_text(encoding="utf-8"))
torch.set_num_threads(2)


def hash_tensors(tensors):
    digest=hashlib.sha256()
    for name,tensor in sorted(tensors,key=lambda item:item[0]):
        value=tensor.detach().cpu().contiguous()
        header=json.dumps([name,str(value.dtype),list(value.shape)],separators=(",",":")).encode()
        raw=value.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(header).to_bytes(8,"little"));digest.update(header)
        digest.update(len(raw).to_bytes(8,"little"));digest.update(raw)
    return digest.hexdigest()


@pytest.mark.parametrize("row",MANIFEST["golden_states"],ids=lambda row:f"{row['genotype']['pan']}_{row['genotype']['td_kernels']}_{row['genotype']['pan_kernels']}_s{row['seed']}")
def test_all_72_architectures_three_seed_legacy_state_hashes(row):
    model=build_model(Genotype.from_dict(row["genotype"]),row["seed"])
    assert len(model.state_dict())==row["tensor_count"]
    assert hash_tensors(model.state_dict().items())==row["state_sha256"]


@pytest.mark.parametrize("row",MANIFEST["golden_outputs"])
def test_four_scale_outputs_and_class_generalization(row):
    genotype=Genotype.from_dict(row["genotype"])
    torch.random.default_generator.manual_seed(row["input_seed"])
    images=torch.rand(1,1,640,640)
    model=build_model(genotype,row["seed"]).eval()
    with torch.inference_mode():output=model(images)
    assert [list(t.shape) for t in output["predictions"]]==row["prediction_shapes"]
    assert [list(t.shape[1:]) for t in output["backbone_features"]]==[[24,160,160],[32,80,80],[96,40,40],[320,20,20]]
    assert all(torch.isfinite(t).all() for values in output.values() for t in values)
    assert len(model.neck.bottom_up_blocks)==(3 if genotype.pan else 0)
    assert all("branches" not in key for key in model.state_dict())
    with pytest.raises(ValueError):model(torch.zeros(1,3,640,640))


def test_build_preserves_cpu_rng_and_registered_seed_policy():
    before=torch.get_rng_state().clone()
    build_model(Genotype(True,(3,5,3),(5,3,5)),43)
    assert torch.equal(before,torch.get_rng_state())
    with pytest.raises(ValueError):build_model(Genotype(False,(3,3,3)),1)
    with pytest.raises(TypeError):build_model(dict(pan=False,td_kernels=[3,3,3]))


def baseline():
    return yaml.safe_load((ROOT/"configs/baseline_sonar_frozen.yaml").read_text(encoding="utf-8"))


def test_frozen_loss_and_decode_protocol():
    from train.loss import decode_detection_predictions
    cfg=baseline();model=build_model(Genotype(False,(3,3,3)))
    loss_cfg=loss_config_for_model(model,cfg)
    assert loss_cfg.strides==(4,8,16,32)
    assert loss_cfg.anchor_sizes[0]==loss_cfg.anchor_sizes[1]
    assert loss_cfg.no_object_weight==.1 and loss_cfg.class_weight==2.
    criterion=DenseAnchorDetectionLoss(loss_cfg)
    predictions=[torch.zeros(1,21,size,size,requires_grad=True) for size in (160,80,40,20)]
    target=[dict(boxes=torch.tensor([[10.,10.,22.,26.]]),labels=torch.tensor([1]))]
    loss,parts=criterion(predictions,target)
    assert torch.isfinite(loss) and parts["positive_anchors"]>0
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in predictions)
    empty=[torch.full((1,21,size,size),-100.) for size in (160,80,40,20)]
    decoded=decode_detection_predictions(empty,config=loss_cfg,score_threshold=.1)
    assert len(decoded)==1 and decoded[0]["boxes"].shape==(0,4)


def test_frozen_map_and_absent_class_false_positives():
    config=baseline()
    record=dict(prediction=dict(boxes=[[10.,10.,20.,20.],[30.,30.,40.,40.]],scores=[.9,.8],labels=[0,1]),
                target=dict(boxes=[[10.,10.,20.,20.]],labels=[0]))
    metrics=metrics_from_records([record],config)
    assert metrics["mAP50_95"]==pytest.approx(1.)
    assert metrics["Precision"]==.5 and metrics["Recall"]==1.
    assert metrics["false_positives"]==1


def write_dataset(root,count=2):
    (root/"data.yaml").write_text("names: [cylider, manta]\n",encoding="utf-8")
    for split in ("train","valid"):
        (root/split/"images").mkdir(parents=True);(root/split/"labels").mkdir()
        for index in range(count):
            Image.new("L",(12,10),color=index).save(root/split/"images"/f"{index:03d}.png")
            (root/split/"labels"/f"{index:03d}.txt").write_text("0 0.5 0.5 0.2 0.2\n",encoding="utf-8")


def test_dataset_manifest_content_binding_and_sealed_test(tmp_path):
    write_dataset(tmp_path)
    config=baseline();loader=make_loader(config,tmp_path,"train",42,0)
    first=bind_dataset(loader)
    assert len(first["entries"])==2
    assert set(first["entries"][0])=={"image","annotation","image_sha256","annotation_sha256"}
    (tmp_path/"train/labels/000.txt").write_text("1 0.5 0.5 0.2 0.2\n",encoding="utf-8")
    assert bind_dataset(loader)["sha256"]!=first["sha256"]
    (tmp_path/"train/labels/000.txt").write_text("0 0.5 0.5 -1 0.2\n",encoding="utf-8")
    with pytest.raises(ValueError):bind_dataset(loader)
    with pytest.raises(ValueError):YoloDetectionDataset(tmp_path/"nonexistent",split="test")


def test_first64_calibration_is_ordered_and_has_no_augmentation(tmp_path):
    write_dataset(tmp_path,64)
    loader,manifest=prepare_calibration(baseline(),tmp_path)
    assert len(loader.dataset)==64
    assert loader.dataset.augment_config=={}
    assert manifest["entries"][0]["image"]=="train/images/000.png"
    assert manifest["entries"][-1]["image"]=="train/images/063.png"
    images,targets=next(iter(loader))
    assert list(images.shape)==[1,1,640,640]
    assert targets[0]["boxes"].shape==(1,4)


def test_checkpoint_identity_digest_checked_before_load_and_budget_increase(tmp_path):
    checkpoint=tmp_path/"latest.pt"
    binding=dict(arch_hash="a"*64,seed=42,budget=40,config_sha256="b"*64,dataset_sha256="c"*64)
    _checkpoint(checkpoint,dict(binding=binding,epoch=40),binding)
    increased=dict(binding,budget=60)
    assert load_bound_checkpoint(checkpoint,increased,allow_budget_increase=True)["epoch"]==40
    with patch("train.training.torch.load",side_effect=AssertionError("must not load")):
        with pytest.raises(ValueError):load_bound_checkpoint(checkpoint,dict(increased,seed=43),allow_budget_increase=True)
        with pytest.raises(ValueError):load_bound_checkpoint(checkpoint,dict(binding,budget=20),allow_budget_increase=True)
        checkpoint.write_bytes(checkpoint.read_bytes()+b"tampered")
        with pytest.raises(ValueError):load_bound_checkpoint(checkpoint,binding)
