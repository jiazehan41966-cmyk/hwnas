"""Bind reused C01 artifacts to the pre-candidate audit receipt."""
import hashlib,json
from pathlib import Path

def verify_r0_artifacts(cfg,gate):
    root=Path(cfg['r0_root']).resolve(strict=True)
    if root!=Path(gate['r0_root']).resolve(strict=True):raise ValueError('R0 root differs from audited root')
    expected={root/'execution_manifest.json':gate['r0_manifest_sha256']}
    for seed,files in gate['result_artifact_hashes'].items():
        for name,digest in files.items():
            path=root/'formal/YOLOv9_T__quality_aware'/f'seed_{seed}'/name
            if not path.resolve().is_relative_to(root):raise ValueError('R0 artifact path escape')
            expected[path]=digest
    for path,digest in expected.items():
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Audited R0 artifact changed: '+str(path))
    return {str(path):digest for path,digest in expected.items()}
