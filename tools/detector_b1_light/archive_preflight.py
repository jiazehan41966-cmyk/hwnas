"""Preserve the source bundle of the failed pre-manifest import attempt."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.detector_b1_light import remote

code = '''from pathlib import Path
import shutil,json,hashlib,time
b=Path(REMOTE_ROOT)
assert not (b/RESULT_ROOT/'execution_manifest.json').exists()
assert not (b/RESULT_ROOT/'queue_status.json').exists()
p=b/'results/detector_b1_light_20261003/preflight_failures/transport01'
p.mkdir(parents=True,exist_ok=False)
for name in ['training_source.tar.gz','staging_manifest.json','tools/detector_b1_light/train.py']:
 shutil.copyfile(b/name,p/Path(name).name)
proof={'reason':'missing transitive hwnas_fpga.hardware package during prepare import',
 'execution_manifest_written':False,'training_started':False,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
 'files':{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in p.iterdir() if x.is_file()}}
(p/'failure.json').write_text(json.dumps(proof,indent=2))
print(json.dumps(proof))
'''.replace('REMOTE_ROOT', repr(remote.REMOTE)).replace('RESULT_ROOT', repr(remote.RESULT))
result = json.loads(remote.execute(code))
target = ROOT / 'results/detector_b1_light_20261003/transport/preflight_failure01.json'
target.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result))
