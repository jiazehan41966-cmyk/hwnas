"""Performance-only admission after verified smoke; never full board acceptance."""
from pathlib import Path
from tools.sonar_deploy import admission as a


def check(root, network, evidence):
    root=Path(root);evidence=Path(evidence)
    receipt=a.read_json(evidence/'VERIFIED.json')
    if receipt.get('status')!='SINGLE_TRAIN_SMOKE_PASS' or receipt.get('test_evaluated') is not False:
        raise ValueError('Verified single-train smoke required')
    for name,digest in receipt['files'].items():
        p=(evidence/name).resolve()
        if not p.is_relative_to(evidence.resolve()) or a.sha256(p)!=digest:
            raise ValueError('Smoke evidence changed: '+name)
    binding=a.load_binding(root);quality=a.accuracy_admission(root,binding)
    if not quality['allowed'] or quality['gates'][network]['status'] not in ('CONTROL_PASS','PASS'):
        raise ValueError('Integer accuracy gate did not pass')
    expected=next(x for x in binding['runs'] if x['network']==network and x['seed']==42)
    result=a.load_result(root,expected)
    for stage in ('integer_parity','hls','route'):a.stage_evidence(root,network,stage,result)
    data=a.read_json(evidence/'completion.json');smoke=data['binding']
    if data['network']!=network or data['seed']!=42 or data['mismatch_count'] or data['test_evaluated'] is not False:
        raise ValueError('Wrong or failed smoke')
    if smoke['deployment_binding_sha256']!=a.sha256(root/'DEPLOYMENT_BINDING.json'):
        raise ValueError('Deployment identity changed')
    for key in ('checkpoint_sha256','package_manifest_sha256'):
        if smoke[key]!=result[key]:raise ValueError('Smoke differs: '+key)
    if smoke['graph_sha256']!=a.sha256(Path(result['package_manifest_path']).parent/'graph.json'):
        raise ValueError('Graph differs')
    return dict(allowed=True,status='PERFORMANCE_ONLY_SMOKE_ADMITTED',full_board_acceptance=False,
                smoke_receipt_sha256=a.sha256(evidence/'VERIFIED.json'),test_evaluated=False)
