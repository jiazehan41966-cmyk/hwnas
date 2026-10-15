"""One bounded read-only snapshot; no programming, inference or sensor writes."""
import argparse
import base64
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import zlib

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from tools.sonar_deploy import admission as a
from tools.sonar_deploy.board_campaign import BRIDGE


def board_code(board_directory):
    source=base64.b64encode(zlib.compress(Path(__file__).with_name('power.py').read_bytes())).decode()
    return f'''import base64,hashlib,json,time,types,statistics,zlib
from pathlib import Path
from pynq import Clocks
r=Path({board_directory!r});results=r/'results'
def read_json(p):
 for attempt in range(4):
  try:return json.loads(p.read_text())
  except json.JSONDecodeError:time.sleep(.05)
 raise RuntimeError('Concurrent result write did not settle')
module=types.ModuleType('bounded_power_source');source=zlib.decompress(base64.b64decode({source!r}))
exec(compile(source,'power.py','exec'),module.__dict__)
sensor_path='/sys/bus/iio/devices/iio:device1'
power=dict(status='NOT_MEASURED');samples=[]
try:
 sensor=module.validate_iio_sensor(sensor_path)
 start=time.monotonic();deadline=start+30
 while True:
  t=time.monotonic();row=module.iio_sample(sensor_path)
  samples.append(dict(monotonic_s=t,**row))
  if t>=deadline:break
  if len(samples)%100==0:print('BOUNDED_POWER_CAPTURE',len(samples),flush=True)
  time.sleep(.1)
 if module.validate_iio_sensor(sensor_path)!=sensor:raise ValueError('Sensor changed')
 energy=sum((b['monotonic_s']-x['monotonic_s'])*(b['power_w']+x['power_w'])/2 for x,b in zip(samples,samples[1:]))
 duration=samples[-1]['monotonic_s']-samples[0]['monotonic_s']
 power=dict(status='MEASURED_ACTIVE_WINDOW_ONLY',mean_power_w=energy/duration,
   min_power_w=min(v['power_w'] for v in samples),max_power_w=max(v['power_w'] for v in samples),
   duration_seconds=duration,sample_count=len(samples),sensor=sensor,
   source_sha256=hashlib.sha256(source).hexdigest(),idle_power_w='NOT_MEASURED',
   energy_per_inference_mj='NOT_MEASURED',images_per_joule='NOT_MEASURED',
   limitation='30s snapshot during running validation; not aligned to inference boundaries; includes PS/PL board input, excludes host')
except Exception as exc:power=dict(status='NOT_MEASURED',reason=type(exc).__name__+': '+str(exc))
payload=dict(captured_at_unix=time.time(),board_directory=str(r),network='D',seed=42,
 actual_clock_mhz=float(Clocks.fclk0_mhz),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
 graph_sha256=hashlib.sha256((r/'graph.json').read_bytes()).hexdigest(),
 bitstream_sha256=hashlib.sha256((r/'h2_graph.bit').read_bytes()).hexdigest(),
 board_manifest_sha256=hashlib.sha256((r/'board_manifest.json').read_bytes()).hexdigest(),
 source_manifest_sha256=hashlib.sha256((r/'source_manifest.json').read_bytes()).hexdigest(),
 layers=read_json(results/'layers.json'),frames=read_json(results/'frames.json'),
 completion=read_json(results/'completion.json') if (results/'completion.json').exists() else None,
 power=power,power_samples=samples,test_evaluated=False)
print('SNAPSHOT_BASE64 '+base64.b64encode(json.dumps(payload).encode()).decode(),flush=True)
'''


def save_csv(path,rows,fields):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--board-directory',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    script=out/'capture_board.py';script.write_text(board_code(args.board_directory),encoding='utf-8')
    with (out/'transport.log').open('w',encoding='utf-8') as log:
        subprocess.run([sys.executable,str(BRIDGE),str(script),str(out/'board.log'),'--keep-kernel'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=100)
    lines=(out/'board.log').read_text().splitlines()
    rows=[v.removeprefix('SNAPSHOT_BASE64 ') for v in lines if v.startswith('SNAPSHOT_BASE64 ')]
    if len(rows)!=1:raise ValueError('Missing unique board snapshot')
    data=json.loads(base64.b64decode(rows[0]));a.atomic_json(out/'snapshot.json',data)
    save_csv(out/'layer_parity.csv',data['layers'],list(data['layers'][0]))
    frames=[{key:r[key] for key in ('index','sample_id','split','graph_submit_wait_seconds','frame_input_and_capture_seconds','mismatch_count')} for r in data['frames']]
    save_csv(out/'frame_times.csv',frames,list(frames[0]))
    samples=data['power_samples']
    if samples:save_csv(out/'power_samples.csv',samples,list(samples[0]))
    a.atomic_json(out/'manifest.json',dict(status='PARTIAL_BOARD_EVIDENCE',test_evaluated=False,
        source_sha256=a.sha256(__file__),files={x.name:a.sha256(x) for x in out.iterdir() if x.is_file()}))
    print(json.dumps(dict(output=str(out),frames=len(frames),train=sum(x['split']=='train' for x in frames),valid=sum(x['split']=='valid' for x in frames),power=data['power'],test_evaluated=False)))


if __name__=='__main__':main()
