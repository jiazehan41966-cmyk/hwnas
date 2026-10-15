"""Recover an already completed board return; no accelerator execution."""
import json
from pathlib import Path
import tarfile

from tools.sonar_deploy import admission as a, board_campaign as c
from tools.sonar_deploy.board_result import validate_actual_frames

ROOT = Path(__file__).resolve().parents[2]
folder = ROOT / 'results/sonar_real_int8_deployment_v1/valid99_restart_20260930T213829/D/graph_build'
evidence = folder / 'board_evidence'
owner = a.read_json(ROOT / 'reports/sonar_real_int8_deployment_v1/D_RECOVERY_TRANSPORT_20261002.json')
tx = c.transport()
archive = evidence / 'board_return.tar.gz'
expected = 'fbb0e5ef3a208afb975289136da3d7fc83f62e957b02919a615ce5ee12288c11'
if not archive.exists():
    c._scp([tx.HOST + ':' + owner['directory'] + '/board_return.tar.gz'], archive)
if a.sha256(archive) != expected:
    raise ValueError('Original board return SHA mismatch')
with tarfile.open(archive, 'r:gz') as tf:
    members = tf.getmembers()
    names = [m.name for m in members]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate archive member')
    for member in members:
        target = (evidence / member.name).resolve()
        if not member.isfile() or not target.is_relative_to(evidence.resolve()) or member.size > 64 * 1024 * 1024:
            raise ValueError('Invalid original evidence member')
        data = tf.extractfile(member).read()
        if target.exists() and target.read_bytes() != data:
            raise ValueError('Conflicting retained evidence: ' + member.name)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
frames = validate_actual_frames(evidence, a.read_json(folder / 'graph.json'),
                                a.read_json(folder / 'frames.json'), a.read_json(folder / 'manifest.json'))
receipt = dict(status='ORIGINAL_RETURN_RECOVERED', archive_sha256=expected,
               archive_bytes=archive.stat().st_size, frame_count=len(frames),
               completion=a.read_json(evidence / 'completion.json'),
               hardware_rerun=False, test_evaluated=False)
a.atomic_json(ROOT / 'reports/sonar_real_int8_deployment_v1/D_ORIGINAL_RETURN_RECOVERY_20261002.json', receipt)
print(json.dumps(receipt))
print(c._stop_transfer(tx.REMOTE, owner))
