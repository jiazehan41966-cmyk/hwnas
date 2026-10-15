"""Completion/report fixtures only; never substitutes route or board evidence."""
import json
import zipfile
from pathlib import Path

import pytest

from tools.sonar_deploy import hardware as h

TIMING='''WNS(ns) TNS(ns) TNS Failing Endpoints TNS Total Endpoints WHS(ns) THS(ns)
------- ------- --------------------- ------------------- ------- -------
0.161 0.000 0 123 0.010 0.000
All user specified timing constraints are met.
clk_pl_0 {0.000 2.667} 5.333 187.512
'''
# Exact summary/distribution row layout from D route02 Vivado 2022.2.
UTIL='''Table of Contents
1. CLB Logic
2. CLB Logic Distribution
3. BLOCKRAM
4. ARITHMETIC
1. CLB Logic
------------
| Site Type | Used | Fixed | Prohibited | Available | Util% |
| CLB LUTs | 25531 | 0 | 0 | 230400 | 11.08 |
| CLB Registers | 28926 | 0 | 0 | 460800 | 6.28 |
1.1 Summary of Registers by Type
--------------------------------
| Total | Clock Enable | Synchronous | Asynchronous |
| 28451 | Yes | Reset | - |
2. CLB Logic Distribution
-------------------------
| Site Type | Used | Fixed | Prohibited | Available | Util% |
| CLB Registers | 28926 | 0 | 0 | 460800 | 6.28 |
3. BLOCKRAM
-----------
| Block RAM Tile | 51 | 0 | 0 | 312 | 16.35 |
4. ARITHMETIC
-------------
| DSPs | 83 | 0 | 0 | 1728 | 4.80 |
'''


def test_actual_vivado_summary_ignores_repeated_distribution_registers():
    report=h.parse_route(TIMING,UTIL)
    assert report['ff_used']==28926 and report['ff_capacity']==460800
    assert report['lut_used']==25531 and report['bram_used']==51 and report['dsp_used']==83
    assert report['actual_clock_mhz']==187.512 and report['wns_ns']==.161
    assert report['ff_utilization_pct']==100*28926/460800


def test_missing_summary_does_not_fall_back_to_distribution():
    changed=UTIL.replace('| CLB Registers | 28926 | 0 | 0 | 460800 | 6.28 |','',1)
    with pytest.raises(ValueError,match='ff'):h.parse_route(TIMING,changed)


def test_ambiguous_summary_still_rejected():
    changed=UTIL.replace('| CLB LUTs | 25531 | 0 | 0 | 230400 | 11.08 |',
                         '| CLB LUTs | 25531 | 0 | 0 | 230400 | 11.08 |\n| CLB LUTs | 9 | 0 | 0 | 230400 | 0.01 |')
    with pytest.raises(ValueError,match='lut'):h.parse_route(TIMING,changed)


@pytest.fixture
def completed(tmp_path):
    build=tmp_path/'D/seed_42/hardware/graph_build';build.mkdir(parents=True)
    (build/'core.cpp').write_text('frozen source fixture')
    h.gate.atomic_json(build/'manifest.json',dict(files={'core.cpp':h.gate.sha256(build/'core.cpp')},layers=[dict(node='n')]))
    (build/'hls.log').write_text('node=0 bytes=1 mismatch=0 status=0\nFinished Command export_design\n')
    (build/'timing.rpt').write_text(TIMING);(build/'utilization.rpt').write_text(UTIL)
    (build/'utilization_hier.rpt').write_text('hierarchical report fixture')
    (build/'route_status.rpt').write_text('# of routable nets..................... : 51599 :\n# of fully routed nets................. : 51599 :\n# of nets with routing errors.......... : 0 :\n')
    (build/'ROUTE_TIMING_PASS').write_text('Tool 2022.2; serial DDR graph executor timing PASS; candidate board validation pending\n')
    (build/'route.log').write_text('write_bitstream completed successfully\nINFO: [Project 1-1941] Successfully created Hardware Platform: fixture\nINFO: [Common 17-206] Exiting Vivado at fixture\n')
    h.gate.atomic_json(build/'route.log.process.json',dict(command=[str(h.VIVADO),'-mode','batch','-source','build_graph_bd.tcl'],
        cwd=str(build.resolve()),returncode=0,elapsed_seconds=962.48))
    with zipfile.ZipFile(build/'h2_graph.xsa','w') as archive:
        archive.writestr('h2_graph.bit',b'fixturebit');archive.writestr('h2_graph_bd.hwh',b'fixturehwh')
    return tmp_path,build


@pytest.mark.parametrize('mutation',['missing','tool_error','log_incomplete','process_incomplete','process_failed','wrong_cwd','wrong_source','bad_xsa','unrouted','timing_fail'])
def test_existing_route_rejects_incomplete_or_corrupt_artifacts(completed,mutation):
    root,build=completed
    if mutation=='missing':(build/'ROUTE_TIMING_PASS').unlink()
    elif mutation=='tool_error':
        path=build/'route.log';path.write_text('ERROR: fixture tool error\n'+path.read_text())
    elif mutation=='log_incomplete':(build/'route.log').write_text('still routing\n')
    elif mutation in ('process_incomplete','process_failed','wrong_cwd'):
        path=build/'route.log.process.json';data=h.gate.read_json(path)
        if mutation=='process_incomplete':data.pop('returncode')
        elif mutation=='process_failed':data['returncode']=1
        else:data['cwd']=str(root)
        h.gate.atomic_json(path,data)
    elif mutation=='wrong_source':(build/'core.cpp').write_text('changed')
    elif mutation=='bad_xsa':(build/'h2_graph.xsa').write_bytes(b'not a zip')
    elif mutation=='unrouted':(build/'route_status.rpt').write_text('# of routable nets... : 2 :\n# of fully routed nets... : 1 :\n# of nets with routing errors... : 1 :\n')
    else:(build/'timing.rpt').write_text(TIMING.replace('0.161','-0.161'))
    with pytest.raises((ValueError,zipfile.BadZipFile)):h._completed_route_execution(build)


def test_finalizer_publishes_without_rerunning_or_mutating_build(monkeypatch,completed):
    root,build=completed
    before={str(p):h.gate.sha256(p) for p in build.iterdir() if p.is_file()}
    monkeypatch.setattr(h.gate,'action_admission',lambda *args:dict(allowed=True))
    monkeypatch.setattr(h.gate,'load_binding',lambda root:dict(runs=[dict(network='D',seed=42)]))
    monkeypatch.setattr(h.gate,'load_result',lambda *args:dict(checkpoint_sha256='a'*64))
    checked=[]
    monkeypatch.setattr(h.gate,'stage_evidence',lambda root,n,stage,result:checked.append(stage))
    monkeypatch.setattr(h,'_run',lambda *args:pytest.fail('finalizer must not run any tool'))
    def publish(root,n,stage,result,artifacts,extra,measured):
        assert stage=='route' and extra['ff_used']==28926
        assert extra['execution']['vivado_executed_by_finalizer'] is False
        assert 'route.log.process.json' in artifacts
        return dict(status='PASS')
    monkeypatch.setattr(h,'_publish',publish)
    assert h.finalize_existing_route(root)==dict(status='PASS')
    assert checked==['hls']
    assert {str(p):h.gate.sha256(p) for p in build.iterdir() if p.is_file()}==before


def test_finalizer_obeys_admission_without_reading_build(monkeypatch,tmp_path):
    denied=dict(allowed=False,status='BLOCKED_EVIDENCE')
    monkeypatch.setattr(h.gate,'action_admission',lambda *args:denied)
    assert h.finalize_existing_route(tmp_path)==denied


def test_route_publisher_uses_real_durable_metric_ledger(monkeypatch,completed):
    root,build=completed
    h.gate.atomic_json(root/'DEPLOYMENT_BINDING.json',dict(fixture=True))
    h.gate.atomic_json(root/'D/seed_42/hls.json',dict(status='PASS',fixture=True))
    result={key:'a'*64 for key in h.gate.IDENTITY}
    result.update(network='D',seed=42,architecture_id=h.gate.ARCHITECTURES['D'],package_manifest_sha256='b'*64)
    # Only preceding deployment admission is fixture-provided. The actual
    # publisher, build_metric_record, append/read ledger and metric checks run.
    def final_check(root,network,stage,result):
        evidence=h.gate.read_json(root/network/'seed_42'/f'{stage}.json')
        h.gate._check_stage_metrics(root,evidence,stage)
        return evidence
    monkeypatch.setattr(h.gate,'stage_evidence',final_check)
    execution=h._completed_route_execution(build)
    evidence=h._publish_route(root,'D',result,build,execution)
    records=h.gate.read_metric_records(root/'METRICS.jsonl')
    assert len(records)==1 and records[0]['event_id']==evidence['metric_event_id']
    record=records[0]
    for key in ['wns_ns','tns_ns','timing_met','lut_used','lut_utilization_pct','ff_used','ff_utilization_pct',
                'dsp_used','dsp_utilization_pct','bram_used','bram_utilization_pct']:
        metric=record['metrics'][key]
        assert metric['status']=='MEASURED'
        source=metric['provenance']
        assert source['evidence_layer']=='place_and_route'
        assert source['target_device']=='ZCU104' and source['tool_version']=='2022.2'
        expected=build/('timing.rpt' if key in ('wns_ns','tns_ns','timing_met') else 'utilization.rpt')
        assert source['artifact_sha256']==h.gate.sha256(expected)
    assert record['metrics']['ff_used']['value']==28926


def test_route_publisher_does_not_invent_tool_version(completed):
    root,build=completed
    (build/'ROUTE_TIMING_PASS').write_text('PASS without actual tool version')
    with pytest.raises(ValueError,match='tool version'):
        h._publish_route(root,'D',dict(checkpoint_sha256='a'*64),build,{})
    assert not (root/'METRICS.jsonl').exists()
