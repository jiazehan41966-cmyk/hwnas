"""Package a timing-passing isolated H2 witness image, never program the board."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tarfile
import zipfile

build=Path(sys.argv[1]).resolve()
route=build/'route_attempt02'
if not (route/'ROUTE_TIMING_PASS').is_file():raise RuntimeError('route gate not passed')
timing=(route/'timing.rpt').read_text()
if 'All user specified timing constraints are met.' not in timing:
    raise RuntimeError('timing report not clean')
output=build/(sys.argv[2] if len(sys.argv)>2 else 'board_payload')
output.mkdir(exist_ok=False)
with zipfile.ZipFile(route/'h2_dma.xsa') as z:
    for source,target in [('h2_dma.bit','h2_dma.bit'),('h2_dma_bd.hwh','h2_dma.hwh')]:
        # SmartConnect contributes additional HWH files; bind the top BD only.
        if z.namelist().count(source)!=1:raise ValueError('missing/ambiguous XSA component '+source)
        (output/target).write_bytes(z.read(source))
header=next((build/'hls_project/solution1/impl/ip/drivers').rglob('*_hw.h'))
macros=dict(re.findall(r'#define\s+XH2_DMA_CONTROL_ADDR_(\w+)\s+(0x[0-9a-fA-F]+)',header.read_text()))
abi={name:int(macros[key],16) for name,key in dict(weights='WEIGHTS_DATA',biases='BIASES_DATA',numeric='NUMERIC_DATA',low='LOW_R_DATA',mode='MODE_DATA',return_='AP_RETURN').items()}
abi['return']=abi.pop('return_')
for label in ('pw','fpn'):shutil.copytree(build/label,output/label)
shutil.copy2(build/'pynq_dma_runner.py',output/'pynq_dma_runner.py')
shutil.copy2(build/'manifest.json',output/'source_manifest.json')
for filename in ('timing.rpt','utilization_hier.rpt','route_status.rpt'):
    shutil.copy2(route/filename,output/filename)
files={p.relative_to(output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in output.rglob('*') if p.is_file()}
manifest=dict(files=files,abi=abi,timing_status='PASS',scope='H2 PW/FPN DMA witnesses only',full_detector=False,
    header_sha256=hashlib.sha256(header.read_bytes()).hexdigest())
(output/'board_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=build/'board_payload.tar.gz'
with tarfile.open(archive,'w:gz') as tf:
    for p in output.rglob('*'):
        if p.is_file():tf.add(p,arcname=p.relative_to(output).as_posix())
print(json.dumps(dict(archive=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),bytes=archive.stat().st_size)))
