"""Independent, strictly V0.1 candidate adapter. No shared weights or search."""
from copy import deepcopy
from dataclasses import replace
import hashlib
from pathlib import Path

import yaml

from hwnas_fpga.models.detection import HWNASMultiScaleDetector
from hwnas_fpga.search_space import ArchitectureSpec
from hwnas_fpga.training.detection_trainer import DetectionLossConfig

ROOT = Path(__file__).resolve().parents[4]
SPEC_PATH = ROOT / 'configs/search_space_sonar_v0_1.yaml'
SPEC_SHA256 = 'ebb1708b394af20258c920eb29204d5cc9504f5908fdfdae62a62d78042eb7c4'


def load_spec(spec):
    """Accept a path or parsed mapping, but never silently widen frozen V0.1."""
    raw = SPEC_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SPEC_SHA256:
        raise ValueError('Frozen V0.1 specification SHA256 changed')
    canonical = yaml.safe_load(raw)
    value = yaml.safe_load(Path(spec).read_text(encoding='utf-8')) if isinstance(spec, (str, Path)) else deepcopy(spec)
    if value != canonical:
        raise ValueError('Specification differs from frozen V0.1 (levels/head/anchors/fixed variables)')
    return value


def validate_sonar_architecture(arch, spec):
    space = load_spec(spec)['architecture_space']
    if not isinstance(arch, dict):
        raise ValueError('architecture must be a mapping')
    extra = set(arch) - {'output_combination', 'kernels'}
    if extra:
        raise ValueError(f'Unsupported architecture fields {sorted(extra)}; levels/head/anchors/expansion/dilation are derived or frozen')
    combos = {c['id']: c for c in space['output_combinations']}
    name = arch.get('output_combination')
    if not isinstance(name, str) or name not in combos:
        raise ValueError(f'Invalid output_combination: {name!r}')
    combo = combos[name]
    expected = {f"stage{s['index']}" for s in space['stages'] if s['op'] == 'mbconv' and s['index'] <= combo['last_active_stage']}
    kernels = arch.get('kernels')
    if not isinstance(kernels, dict):
        raise ValueError('kernels must explicitly map every active MBConv stage to 3 or 5')
    missing, extra = expected-set(kernels), set(kernels)-expected
    if missing or extra:
        raise ValueError(f'Invalid kernel stages: missing active={sorted(missing)}, inactive/illegal={sorted(extra)}')
    for stage, k in kernels.items():
        if type(k) is not int or k not in (3, 5):
            raise ValueError(f'{stage}: kernel must be integer 3 or 5; got {k!r}')
    return deepcopy(arch)


class SonarCandidateDetector(HWNASMultiScaleDetector):
    """Reuse frozen modules with a physically truncated ArchitectureSpec."""
    def __init__(self, space, arch, *, num_classes=2, image_size=640):
        self.input_channels = 1
        self.image_size = int(image_size)
        if self.image_size <= 0:
            raise ValueError('image_size must be positive')
        x = space['architecture_space']
        combo = next(c for c in x['output_combinations'] if c['id'] == arch['output_combination'])
        stages = []
        for s in x['stages'][:combo['last_active_stage']+1]:
            block = dict(op=s['op'], kernel_size=arch['kernels'].get(f"stage{s['index']}", s.get('kernel', 1)),
                         expand_ratio=s.get('expansion', 1), stride=s['stride'])
            stages.append(dict(channels=s['channels'], depth=1, stride=s['stride'], blocks=[block]))
        architecture = ArchitectureSpec.from_dict(dict(input_channels=1, stem_channels=32, stem_stride=2,
            post_stem_downsample_stride=1, stages=stages))
        levels = tuple(combo['levels'])
        super().__init__(architecture, num_classes=int(num_classes), feature_indices=[x['features'][p]['stage'] for p in levels],
                         neck_channels=96, anchors_per_location=3)
        self.candidate_architecture = deepcopy(arch)
        self.levels = levels
        self.feature_strides = tuple(x['features'][p]['stride'] for p in levels)
        self.anchor_sizes = tuple(tuple(tuple(pair) for pair in x['anchors'][p]) for p in levels)

    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != (self.input_channels, self.image_size, self.image_size):
            raise ValueError(
                f'V0.1 transfer requires [B,{self.input_channels},{self.image_size},{self.image_size}] input'
            )
        return super().forward(x)

    def loss_config(self, baseline_config):
        base = DetectionLossConfig.from_config(baseline_config)
        return replace(
            base,
            num_classes=int(self.head.num_classes),
            image_size=self.image_size,
            strides=self.feature_strides,
            anchor_sizes=self.anchor_sizes,
            class_loss_weights=tuple(1.0 for _ in range(int(self.head.num_classes))),
        )


def build_sonar_candidate(search_space_config, architecture, *, num_classes=2, image_size=640):
    space = load_spec(search_space_config)
    arch = validate_sonar_architecture(architecture, space)
    return SonarCandidateDetector(space, arch, num_classes=num_classes, image_size=image_size)
