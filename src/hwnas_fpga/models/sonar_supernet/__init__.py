"""Single-path V0.1 supernet reusing the frozen candidate's execution modules."""
from copy import deepcopy
from dataclasses import replace
import itertools
import json
import random
from collections import Counter
from types import SimpleNamespace

import torch
from torch import nn

from hwnas_fpga.models.sonar_candidate import (SPEC_PATH, SPEC_SHA256, load_spec,
    validate_sonar_architecture, build_sonar_candidate)
from hwnas_fpga.models.detection import HWNASFeatureBackbone, FixedFPNNeck
from hwnas_fpga.training.detection_trainer import DetectionLossConfig


def serialize_architecture(architecture, spec=SPEC_PATH):
    return json.dumps(validate_sonar_architecture(architecture, spec), sort_keys=True, separators=(',', ':'))


def deserialize_architecture(value, spec=SPEC_PATH):
    return validate_sonar_architecture(json.loads(value), spec)


def architecture_id(architecture, spec=SPEC_PATH):
    a = validate_sonar_architecture(architecture, spec)
    return a['output_combination'] + '_' + '_'.join(f's{s[5:]}k{k}' for s,k in sorted(a['kernels'].items()))


def enumerate_architectures(spec=SPEC_PATH):
    x = load_spec(spec)['architecture_space']
    for combo in x['output_combinations']:
        stages = [s['index'] for s in x['stages'] if s['op']=='mbconv' and s['index']<=combo['last_active_stage']]
        for kernels in itertools.product((3,5), repeat=len(stages)):
            yield dict(output_combination=combo['id'], kernels={f'stage{s}':k for s,k in zip(stages,kernels)})


class BalancedArchitectureSampler:
    policy = 'uniform_output_combination_then_independent_bernoulli_kernel_v1'

    def __init__(self, seed=42, spec=SPEC_PATH):
        self.spec = load_spec(spec)
        self.seed = seed
        self.rng = random.Random(seed)
        self.statistics = Counter()

    def sample(self):
        x=self.spec['architecture_space']
        combo=self.rng.choice(x['output_combinations'])
        kernels={f"stage{s['index']}":self.rng.choice((3,5)) for s in x['stages']
                 if s['op']=='mbconv' and s['index']<=combo['last_active_stage']}
        self.statistics[combo['id']]+=1
        for stage,k in kernels.items(): self.statistics[f'{stage}_k{k}']+=1
        return dict(output_combination=combo['id'], kernels=kernels)

    def state_dict(self):
        return dict(seed=self.seed, rng=self.rng.getstate(), statistics=dict(self.statistics),policy=self.policy)

    def load_state_dict(self, state):
        if state['policy']!=self.policy: raise ValueError('Sampling policy mismatch')
        self.seed=state['seed']; self.rng.setstate(state['rng']); self.statistics=Counter(state['statistics'])


class _KernelBranch(nn.Module):
    def __init__(self, k3, k5):
        super().__init__()
        self.branches=nn.ModuleDict({'k3':k3,'k5':k5})
        self.active=3

    def forward(self,x):
        return self.branches[f'k{self.active}'](x)


class SonarSupernet(nn.Module):
    def __init__(self, spec=SPEC_PATH):
        super().__init__()
        self.spec=load_spec(spec)
        arches=list(enumerate_architectures(self.spec))
        full=[a for a in arches if a['output_combination']=='P2_P3_P4_P5']
        k3=build_sonar_candidate(self.spec,full[0])
        k5=build_sonar_candidate(self.spec,full[-1])
        self.backbone=k3.backbone
        for s in full[0]['kernels']:
            idx=int(s[5:]); a=self.backbone.stages[idx][0]; b=k5.backbone.stages[idx][0]
            a.dw_conv=_KernelBranch(a.dw_conv,b.dw_conv)
            a.dw_bn=_KernelBranch(a.dw_bn,b.dw_bn)
        self.laterals=nn.ModuleDict(zip(k3.levels,k3.neck.lateral_convs))
        self.outputs=nn.ModuleDict(zip(k3.levels,k3.neck.output_convs))
        self.heads=nn.ModuleDict(zip(k3.levels,k3.head.towers))
        self.set_active_subnet(full[0])

    def set_active_subnet(self, architecture):
        a=validate_sonar_architecture(architecture,self.spec)
        x=self.spec['architecture_space']
        combo=next(c for c in x['output_combinations'] if c['id']==a['output_combination'])
        self._active=deepcopy(a)
        self.levels=tuple(combo['levels']); self.last_stage=combo['last_active_stage']
        self.feature_indices=tuple(x['features'][p]['stage'] for p in self.levels)
        self.feature_strides=tuple(x['features'][p]['stride'] for p in self.levels)
        self.anchor_sizes=tuple(tuple(map(tuple,x['anchors'][p])) for p in self.levels)
        for s,k in a['kernels'].items():
            block=self.backbone.stages[int(s[5:])][0]
            block.dw_conv.active=k; block.dw_bn.active=k

    def get_active_subnet(self):
        return deepcopy(self._active)

    def forward(self,x):
        if x.ndim!=4 or tuple(x.shape[1:])!=(1,640,640): raise ValueError('V0.1 requires [B,1,640,640]')
        # Reuse candidate execution methods with views of persistent active modules.
        bv=SimpleNamespace(stem=self.backbone.stem, post_stem_downsample=self.backbone.post_stem_downsample,
            stages=self.backbone.stages[:self.last_stage+1],feature_indices=self.feature_indices)
        features=HWNASFeatureBackbone.forward(bv,x)
        nv=SimpleNamespace(in_channels=self.feature_indices,lateral_convs=[self.laterals[p] for p in self.levels],
                           output_convs=[self.outputs[p] for p in self.levels])
        neck=FixedFPNNeck.forward(nv,features)
        return dict(backbone_features=features,neck_features=neck,
                    predictions=tuple(self.heads[p](f) for p,f in zip(self.levels,neck)))

    def loss_config(self, baseline_config):
        return replace(DetectionLossConfig.from_config(baseline_config),strides=self.feature_strides,anchor_sizes=self.anchor_sizes)

    def candidate_state_mapping(self,architecture=None):
        a=validate_sonar_architecture(architecture or self._active,self.spec)
        candidate=build_sonar_candidate(self.spec,a)
        mapping={}
        for key in candidate.state_dict():
            target=key
            parts=key.split('.')
            if key.startswith('backbone.stages.') and parts[4] in ('dw_conv','dw_bn'):
                k=a['kernels'][f'stage{parts[2]}']
                target='.'.join(parts[:5]+['branches',f'k{k}']+parts[5:])
            elif key.startswith('neck.'):
                group={'lateral_convs':'laterals','output_convs':'outputs'}[parts[1]]
                target='.'.join([group,candidate.levels[int(parts[2])]]+parts[3:])
            elif key.startswith('head.towers.'):
                target='.'.join(['heads',candidate.levels[int(parts[2])]]+parts[3:])
            mapping[key]=target
        return candidate,mapping

    def export_active_subnet(self,architecture=None):
        candidate,mapping=self.candidate_state_mapping(architecture)
        state=self.state_dict()
        candidate.to(next(self.parameters()).device)
        candidate.load_state_dict({k:state[v] for k,v in mapping.items()},strict=True)
        candidate.train(self.training)
        return candidate


def build_sonar_supernet(spec=SPEC_PATH):
    return SonarSupernet(spec)


@torch.no_grad()
def calibrate_subnet_bn(supernet,architecture,train_loader,num_batches=2):
    """Return an independent recalibrated candidate; never mutate the supernet."""
    if num_batches<1: raise ValueError('num_batches must be positive')
    if getattr(train_loader.dataset,'split',None)!='train': raise ValueError('BN calibration requires train split')
    # Export construction initializes modules before loading weights: preserve caller RNG too.
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        candidate=supernet.export_active_subnet(architecture)
        candidate.eval()
        bns=[m for m in candidate.modules() if isinstance(m,nn.BatchNorm2d)]
        momenta=[m.momentum for m in bns]
        for m in bns: m.reset_running_stats(); m.momentum=None; m.train()
        count=0
        for images,_ in train_loader:
            candidate(images.to(next(candidate.parameters()).device))
            count+=1
            if count>=num_batches: break
        if count<num_batches: raise ValueError(f'Only {count} calibration batches available, requested {num_batches}')
        for m,momentum in zip(bns,momenta): m.momentum=momentum
        candidate.eval()
        candidate.bn_calibration_batches=count
        return candidate
