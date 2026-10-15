"""Explicit independent training executor; importing this file never starts work.

Protocol migrated from joint_v2_elastic.train_candidate and formal_runner's
single-candidate train_epoch. No supernet, automatic job queue or backend gates
are imported. A training result establishes software evidence only.
"""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from models import build_model
from space import Genotype, canonical_hash
from .data import make_loader,bind_dataset
from .loss import DenseAnchorDetectionLoss
from .protocol import loss_config_for_model,validate_protocol
from .evaluation import evaluate,finite_outputs


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)


def rng_state(loader):
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        sampler_generator=loader.sampler.generator.get_state() if getattr(loader.sampler,'generator',None) is not None else None,
        loader_generator=loader.generator.get_state() if loader.generator is not None else None)


def restore_rng(state,loader):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda'] and torch.cuda.is_available():torch.cuda.set_rng_state_all(state['cuda'])
    if state['sampler_generator'] is not None:loader.sampler.generator.set_state(state['sampler_generator'])
    if state['loader_generator'] is not None:loader.generator.set_state(state['loader_generator'])


def train_epoch(model,optimizer,loader,config,device):
    model.train(); sums=Counter(); count=0; start=time.perf_counter()
    criterion=DenseAnchorDetectionLoss(loss_config_for_model(model,config)).to(device)
    for images,targets in loader:
        optimizer.zero_grad(set_to_none=True)
        output=model(images.to(device));finite_outputs(output)
        loss,parts=criterion(output,targets)
        if not torch.isfinite(loss) or not all(math.isfinite(v) for v in parts.values()):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        grads=[p.grad for p in model.parameters() if p.grad is not None]
        if not grads or any(not torch.isfinite(g).all() for g in grads):
            raise ValueError("Nonfinite/missing active gradients")
        optimizer.step();count+=1
        for key in ('loss','box_loss','object_loss','class_loss'):sums[key]+=parts[key]
    if not count:raise ValueError("Empty train loader")
    return dict(losses={k:v/count for k,v in sums.items()},batches=count,
        gradient_finite=True,lr=optimizer.param_groups[0]['lr'],elapsed_seconds=time.perf_counter()-start)


def _save_json(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8");temp.replace(path)


def _checkpoint(path,state,binding):
    path=Path(path);temp=path.with_suffix(".tmp")
    torch.save(state,temp);temp.replace(path)
    _save_json(path.with_suffix(".binding.json"),dict(binding=binding,
        checkpoint_sha256=sha256(path.read_bytes()).hexdigest()))


def load_bound_checkpoint(path, expected_binding, *, allow_budget_increase=False):
    """Check JSON identity and file digest before loading a locally produced PT."""
    path=Path(path)
    meta=json.loads(path.with_suffix(".binding.json").read_text(encoding="utf-8"))
    saved_binding=meta["binding"]
    compare_binding=deepcopy(saved_binding)
    if allow_budget_increase:
        if type(saved_binding.get("budget")) is not int or saved_binding["budget"]>expected_binding["budget"]:
            raise ValueError("Resume budget may only increase")
        compare_binding["budget"]=expected_binding["budget"]
    if compare_binding != expected_binding:
        raise ValueError("Checkpoint genotype/config/data/seed/budget binding differs")
    if meta["checkpoint_sha256"] != sha256(path.read_bytes()).hexdigest():
        raise ValueError("Checkpoint content digest differs")
    state=torch.load(path,map_location="cpu",weights_only=False)
    if state.get("binding") != saved_binding: raise ValueError("Checkpoint internal binding differs")
    return state


def train_candidate(genotype, config, output, *, seed=42, epochs=None,device="cpu",resume=False):
    if not isinstance(genotype,Genotype): raise TypeError("Genotype required")
    validate_protocol(config,genotype)
    if type(seed) is not int or seed not in (42,43,44): raise ValueError("Registered seed required")
    budget=config["training"]["epochs"] if epochs is None else epochs
    if type(budget) is not int or budget < 1: raise ValueError("Positive integer epoch budget required")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no implicit CPU fallback")
    folder=Path(output);folder.mkdir(parents=True,exist_ok=True)
    latest=folder/"latest.pt"
    if latest.exists() and not resume: raise FileExistsError("Use explicit resume for existing training")
    seed_all(seed);model=build_model(genotype,seed).to(device);seed_all(seed)
    data=Path(config["dataset"]["data_dir"]).expanduser()
    if not data.is_absolute(): raise ValueError("dataset.data_dir must be explicitly resolved to an absolute path")
    train=make_loader(config,data,"train",seed,0,train=True)
    valid=make_loader(config,data,"valid",seed,0)
    if train.dataset.class_names != config["dataset"]["class_names"] or valid.dataset.class_names != config["dataset"]["class_names"]:
        raise ValueError("data.yaml class names differ from bound config")
    datasets={"train":bind_dataset(train),"valid":bind_dataset(valid)}
    binding=dict(genotype=genotype.to_dict(),arch_hash=genotype.arch_hash,seed=seed,budget=budget,
        config_sha256=canonical_hash(config),dataset_sha256=canonical_hash(datasets),test_evaluated=False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    criterion=DenseAnchorDetectionLoss(loss_config_for_model(model,config)).to(device)
    history=[];best=None;best_epoch=None;best_score=-1.
    if resume:
        if not latest.is_file(): raise FileNotFoundError(latest)
        state=load_bound_checkpoint(latest,binding,allow_budget_increase=True)
        history=state["history"]
        if [r["epoch"] for r in history] != list(range(1,len(history)+1)) or state["epoch"] != len(history) or len(history)>budget:
            raise ValueError("Resume epoch/history mismatch")
        scores=[r["valid"]["mAP50_95"] for r in history]
        if not scores or any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in scores):
            raise ValueError("Invalid resumed validation history")
        best_score=max(scores);best_epoch=scores.index(best_score)+1
        if state["best_score"] != best_score or state["best_epoch"] != best_epoch: raise ValueError("Resume best epoch differs")
        best=state["best_state"]
        current=state["model_state_dict"]
        if set(best) != set(current) or any(best[k].shape!=current[k].shape or best[k].dtype!=current[k].dtype or not torch.isfinite(best[k]).all() for k in best):
            raise ValueError("Resume best-state tensor set/shape/dtype/finite mismatch")
        model.load_state_dict(current,strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"]);restore_rng(state["rng"],train)
    # Write manifests only after the existing checkpoint identity/history was validated.
    _save_json(folder/"dataset_binding.json",datasets)
    _save_json(folder/"training_binding.json",binding)
    for epoch in range(len(history)+1,budget+1):
        row=train_epoch(model,optimizer,train,config,device)
        metrics,_=evaluate(model,valid,criterion,config,device)
        score=metrics["mAP50_95"]
        if not math.isfinite(score) or not 0<=score<=1:raise ValueError("Invalid validation mAP")
        row.update(epoch=epoch,valid=metrics);history.append(row)
        if score>best_score:
            best_score,best_epoch=score,epoch
            best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        _checkpoint(latest,dict(binding=binding,epoch=epoch,history=history,
            model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
            rng=rng_state(train),best_state=best,best_score=best_score,best_epoch=best_epoch),binding)
        _save_json(folder/"history.json",history)
    if best is None:raise ValueError("No completed epoch")
    best_path=folder/f"best_{budget}.pt"
    _checkpoint(best_path,dict(binding=binding,epoch=best_epoch,
        model_state_dict=best,valid=history[best_epoch-1]["valid"]),binding)
    receipt=dict(binding=binding,best_epoch=best_epoch,mAP50_95=best_score,
        checkpoint=best_path.name,checkpoint_sha256=sha256(best_path.read_bytes()).hexdigest(),
        status="COMPLETE_SOFTWARE_TRAIN_VALID",test_evaluated=False,
        hardware_latency="NEEDS_MEASUREMENT",power="NEEDS_MEASUREMENT")
    _save_json(folder/f"training_result_{budget}.json",receipt)
    return receipt
