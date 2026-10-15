"""Package a timing-passing, manifest-bound serial graph image."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tarfile
import zipfile

build=Path(sys.argv[1]).resolve()
source_manifest=json.loads((build/'manifest.json').read_text())
for name,digest in source_manifest['files'].items():
    if hashlib.sha256((build/name).read_bytes()).hexdigest()!=digest:
        raise ValueError('bound build artifact changed: '+name)
if not (build/'ROUTE_TIMING_PASS').is_file():raise RuntimeError('route gate not passed')
timing=(build/'timing.rpt').read_text()
if 'All user specified timing constraints are met.' not in timing:
    raise RuntimeError('timing report not clean')
output=build/'board_payload'
output.mkdir(exist_ok=False)
with zipfile.ZipFile(build/'h2_graph.xsa') as archive:
    for source,target in [('h2_graph.bit','h2_graph.bit'),('h2_graph_bd.hwh','h2_graph.hwh')]:
        if archive.namelist().count(source)!=1:raise ValueError('ambiguous XSA component')
        (output/target).write_bytes(archive.read(source))
header=next((build/'hls_project/solution1/impl/ip/drivers').rglob('*_hw.h'))
macros=dict(re.findall(r'#define\s+XH2_GRAPH_CONTROL_ADDR_(\w+)\s+(0x[0-9a-fA-F]+)',header.read_text()))
abi={name:int(macros[key],16) for name,key in {
    'arena':'ARENA_DATA','weights':'WEIGHTS_DATA','biases':'BIASES_DATA',
    'descriptor':'DESCRIPTOR_DATA','return':'AP_RETURN'}.items()}
for name in ['arena.bin','descriptors.bin','expected_layers.bin','pynq_graph_runner.py',
             'timing.rpt','route_status.rpt','utilization_hier.rpt']:
    shutil.copy2(build/name,output/name)
for name in ['graph.json','export_manifest.json','frames.json']:
    if (build/name).is_file():shutil.copy2(build/name,output/name)
frames=json.loads((build/'frames.json').read_text()) if (build/'frames.json').is_file() else None
if frames:
    if frames.get('graph_sha256')!=source_manifest['graph_sha256']:
        raise ValueError('frame set graph mismatch')
    if frames.get('test_evaluated') is not False:
        raise ValueError('frame set must explicitly seal test')
    for name,digest in frames['files'].items():
        target=(build/name).resolve()
        if not target.is_relative_to(build) or hashlib.sha256(target.read_bytes()).hexdigest()!=digest:
            raise ValueError('frame binding mismatch: '+name)
        (output/name).parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(target,output/name)
shutil.copy2(build/'manifest.json',output/'source_manifest.json')
files={p.relative_to(output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in output.rglob('*') if p.is_file()}
clock=re.search(r'^clk_pl_0\s+\{[^}]+\}\s+([\d.]+)\s+([\d.]+)',timing,re.M)
if not clock:raise ValueError('actual PL clock missing')
(output/'board_manifest.json').write_text(json.dumps(dict(files=files,abi=abi,
    implemented_clock_mhz=float(clock[2]),implemented_period_ns=float(clock[1]),
    scope=source_manifest['scope'],real_weights_bound=source_manifest.get('real_weights_bound',False),
    graph_sha256=source_manifest['graph_sha256'],test_evaluated=False,
    executor=source_manifest['implementation'],
    header_sha256=hashlib.sha256(header.read_bytes()).hexdigest()),indent=2))
archive=build/'board_payload.tar.gz'
with tarfile.open(archive,'w:gz') as tf:
    for p in output.rglob('*'):
        if p.is_file():tf.add(p,arcname=p.relative_to(output).as_posix())
print(json.dumps(dict(archive=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),bytes=archive.stat().st_size)))
