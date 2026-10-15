"""Seeded synthetic export / BN witnesses. No dataset, training, NAS or Q search."""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from hwnas_fpga.models.sonar_candidate import build_sonar_candidate, SPEC_PATH, load_spec
from export_contract import export_candidate_predictions, fold_candidate_bn

OUT = ROOT / 'reports/fpga_v0_1_closure'
ART = ROOT / 'artifacts/fpga_v0_1_closure/20260914_v1'


def metrics(a, b):
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return dict(max_abs_error=float(d.max()), mean_abs_error=float(d.mean()), exact_match_ratio=float((a == b).mean()))


def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    OUT.mkdir(parents=True, exist_ok=True)
    export_dir = ART / 'export'
    export_dir.mkdir(parents=True, exist_ok=True)
    x = torch.randn(1, 1, 640, 640)
    np.save(export_dir / 'input.npy', x.numpy())
    space = load_spec(SPEC_PATH)['architecture_space']
    rows, manifests, layer_rows = [], [], []
    for combo in space['output_combinations']:
        for kernel in [3, 5]:
            arch = dict(output_combination=combo['id'], kernels={f"stage{s['index']}": kernel for s in space['stages'] if s['op'] == 'mbconv' and s['index'] <= combo['last_active_stage']})
            model = build_sonar_candidate(SPEC_PATH, arch).eval()
            # Nontrivial frozen BN statistics avoid a vacuous identity-BN witness.
            with torch.no_grad():
                for m in model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.running_mean.uniform_(-.2, .2)
                        m.running_var.uniform_(.5, 1.5)
                        m.weight.uniform_(.7, 1.3)
                        m.bias.uniform_(-.1, .1)
            tag = combo['id'] + f'_k{kernel}'
            folded, pairs = fold_candidate_bn(model)
            layer_outputs = {}
            handles = []
            for pair in pairs:
                name = pair['bn']
                def capture(module, inputs, output, name=name):
                    layer_outputs[name] = output.detach().clone()
                handles.append(model.get_submodule(name).register_forward_hook(capture))
            with torch.no_grad():
                ref = model(x)['predictions']
            for h in handles:
                h.remove()
            for pair in pairs:
                name = pair['bn']
                def compare(module, inputs, output, name=name):
                    a, b = layer_outputs[name].numpy(), output.detach().numpy()
                    result = metrics(a, b)
                    passed = bool(np.allclose(a, b, rtol=1e-4, atol=2e-5))
                    layer_rows.append(dict(case=tag, layer=name, dtype='FP32', passed=passed, **result))
                    assert passed, (tag, name, result)
                handles.append(folded.get_submodule(name).register_forward_hook(compare))
            with torch.no_grad():
                got = folded(x)['predictions']
            for h in handles:
                h.remove()
            for p, a, b in zip(model.levels, ref, got):
                a, b = a.numpy(), b.numpy()
                np.save(export_dir / f'{tag}_{p}_reference.npy', a)
                np.save(export_dir / f'{tag}_{p}_bn_folded.npy', b)
                assert np.allclose(a, b, rtol=1e-4, atol=2e-5)
                rows.append(dict(case=tag, operator='detector_'+p, domain='FP32_BN_FOLD', status='NUMERICAL_PARITY_VERIFIED', **metrics(a, b)))
            meta = export_candidate_predictions(model, export_dir / (tag + '.onnx'), x)
            torch.save(model.state_dict(), export_dir / (tag + '_synthetic_state.pt'))
            manifests.append(dict(case=tag, bn_pairs=pairs, **meta))
            print(tag, 'export/BN PASS', len(pairs), flush=True)
    for filename, records in [('software_parity.csv', rows), ('bn_layer_parity.csv', layer_rows)]:
        with (OUT / filename).open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    (OUT / 'export_manifest.json').write_text(json.dumps(manifests, indent=2), encoding='utf-8')
    (OUT / 'software_environment.json').write_text(json.dumps(dict(seed=42, torch=torch.__version__, numpy=np.__version__, python=sys.version, input='synthetic standard normal; no dataset', spec_sha256=hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest()), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
