"""Version-bound checkpoints for single-path training, including RNG state."""
import hashlib
import random
import subprocess
from pathlib import Path
import numpy as np
import torch
from . import SPEC_SHA256, BalancedArchitectureSampler


def save_checkpoint(path,model,optimizer,sampler,baseline_path,*,epoch,global_step,data_state=None):
    try:
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(baseline_path).resolve().parent.parent,text=True).strip()
    except subprocess.CalledProcessError:
        commit='unavailable'
    state=dict(model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),epoch=epoch,
        global_step=global_step,search_space_version=model.spec['name'],search_space_sha256=SPEC_SHA256,
        sampling_policy=sampler.policy,seed=sampler.seed,architecture_sampling_statistics=dict(sampler.statistics),
        baseline_protocol_hash=hashlib.sha256(Path(baseline_path).read_bytes()).hexdigest(),git_commit=commit,
        architecture_sampler_state=sampler.state_dict(),active_architecture=model.get_active_subnet(),
        python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],data_state=data_state)
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    torch.save(state,path)


def load_checkpoint(path,model,optimizer,sampler,baseline_path):
    # Only load trusted, locally generated checkpoints (optimizer/RNG Python objects).
    state=torch.load(path,map_location='cpu',weights_only=False)
    expected={'search_space_version':model.spec['name'],'search_space_sha256':SPEC_SHA256,
        'sampling_policy':BalancedArchitectureSampler.policy,
        'baseline_protocol_hash':hashlib.sha256(Path(baseline_path).read_bytes()).hexdigest()}
    for k,v in expected.items():
        if state.get(k)!=v: raise ValueError(f'Checkpoint {k} mismatch')
    model.set_active_subnet(state['active_architecture'])
    model.load_state_dict(state['model_state_dict'],strict=True)
    optimizer.load_state_dict(state['optimizer_state_dict'])
    sampler.load_state_dict(state['architecture_sampler_state'])
    random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng'] and torch.cuda.is_available(): torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state
