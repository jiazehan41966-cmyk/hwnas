"""Real-package graph ABI, storage, and board preflight; no FPGA/test data."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import types

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools/detector_preflight'))
import build_graph_witness as build
import prepare
spec = importlib.util.spec_from_file_location('real_graph_driver', ROOT/'tools/detector_preflight/pynq_graph_runner.py')
driver = importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)


def graph_package(letter='A'):
    source = ROOT/f'artifacts/fpga_h2/20260916_v1/graph_{letter}'
    graph = json.loads((source/'graph.json').read_text())
    sizes = {f:(source/f).stat().st_size for f in ('weights.bin','biases.bin','layers.bin')}
    return graph, sizes


def insert_requant(graph):
    node = dict(id='image_requant', op='requant', inputs=['image'],
                input_exp=graph['edges']['image']['exp'], output_exp=graph['edges']['image']['exp']-1,
                multiplier=2, shift=0)
    graph['edges'][node['id']] = dict(shape=graph['edges']['image']['shape'].copy(), exp=node['output_exp'])
    graph['nodes'][0]['inputs'] = [node['id']]
    graph['nodes'].insert(0,node)
    return graph


@pytest.mark.parametrize('letter', ['A','B','C'])
def test_requant_descriptor_and_all_heads_preserved(letter):
    graph, sizes = graph_package(letter); insert_requant(graph)
    plan = prepare.plan_graph(graph, sizes)
    offsets = {n['id']:index*16 for index,n in enumerate(graph['nodes'])}
    descriptors = build.make_descriptors(graph, plan, offsets)
    assert descriptors.shape == (len(graph['nodes']),32)
    assert descriptors.dtype == np.dtype('<i4')
    assert descriptors[0,0] == 4 and descriptors[0,16] == 2 and descriptors[0,17] == 0
    assert descriptors[0,24] == 0
    assert descriptors[0,26] == build.MAGIC
    assert [o['level'] for o in plan['outputs']] == list(graph['outputs'])
    for output in plan['outputs']:
        assert plan['allocations'][output['edge']]['last_use'] == len(graph['nodes'])


@pytest.mark.parametrize('mutation', ['shape','edge_exp','ratio','shift','multiplier'])
def test_reject_invalid_requant_before_hardware(mutation):
    graph,sizes = graph_package(); insert_requant(graph)
    if mutation == 'shape': graph['edges']['image_requant']['shape'] = [1,320,320]
    if mutation == 'edge_exp': graph['edges']['image_requant']['exp'] += 1
    if mutation == 'ratio': graph['nodes'][0]['multiplier'] = 1
    if mutation == 'shift': graph['nodes'][0]['shift'] = 32
    if mutation == 'multiplier': graph['nodes'][0]['multiplier'] = 2**31
    with pytest.raises(ValueError, match='requant'):
        prepare.plan_graph(graph,sizes)


def test_legacy_descriptor_bytes_unchanged():
    graph,sizes = graph_package('A')
    plan = prepare.plan_graph(graph,sizes)
    offsets = {}; cursor = 0
    for n in graph['nodes']:
        offsets[n['id']] = cursor
        cursor += plan['allocations'][n['id']]['real_bytes']
    current = build.make_descriptors(graph,plan,offsets)
    archived = np.fromfile(ROOT/'results/detector_execution_20260922/graph_attempt04/descriptors.bin',dtype='<i4').reshape(-1,32)
    assert np.array_equal(current,archived)


def fake_real_package(tmp_path):
    source = ROOT/'artifacts/fpga_h2/20260916_v1/graph_A'
    for name in ('graph.json','weights.bin','biases.bin','layers.bin'):
        shutil.copy2(source/name,tmp_path/name)
    graph=json.loads((tmp_path/'graph.json').read_text())
    graph.update(real_weights_bound=True,synthetic_weights=False,
                 provenance=dict(checkpoint_sha256='a'*64,profile_sha256='b'*64))
    (tmp_path/'graph.json').write_text(json.dumps(graph))
    (tmp_path/'export_manifest.json').write_text(json.dumps(dict(graph_sha256=build.sha(tmp_path/'graph.json'),provenance=graph['provenance'])))
    return graph


def test_real_package_hash_binding(tmp_path):
    fake_real_package(tmp_path)
    graph,binding=build.validate_package(tmp_path)
    assert binding['export_manifest_sha256']==build.sha(tmp_path/'export_manifest.json')
    (tmp_path/'weights.bin').write_bytes(b'wrong')
    with pytest.raises(ValueError,match='parameter hash mismatch'):
        build.validate_package(tmp_path)


def test_real_package_cannot_substitute_graph(tmp_path):
    graph=fake_real_package(tmp_path)
    graph['provenance']['checkpoint_sha256']='c'*64
    (tmp_path/'graph.json').write_text(json.dumps(graph))
    with pytest.raises(ValueError,match='graph SHA'):
        build.validate_package(tmp_path)


def frames_fixture(tmp_path):
    value=np.arange(4,dtype=np.int8).reshape(1,2,2)
    np.save(tmp_path/'image.npy',value)
    rows=[]
    for i in range(107):
        rows.append(dict(sample_id=str(i),split='train' if i<8 else 'valid',input='image.npy',
                         expected_heads={'P3':'x'},expected_layers={'layer':'x'}))
    source=dict(graph_sha256='g',real_weights_bound=True,plan=dict(outputs=[dict(level='P3')]),layers=[dict(node='layer')])
    payload=dict(frames=rows,graph_sha256='g',test_evaluated=False,calibration_first8_sample_ids=[str(i) for i in range(8)],
                 files={'image.npy':build.sha(tmp_path/'image.npy')},warmup=1,repetitions=10)
    (tmp_path/'frames.json').write_text(json.dumps(payload))
    return source,payload


def test_frames_fixed_manifest_valid(tmp_path):
    source,payload=frames_fixture(tmp_path)
    assert driver.validate_frames(tmp_path,source)==payload


@pytest.mark.parametrize('mutation',['test','order','heads','traces','repetitions','duplicate','missing'])
def test_frames_reject_unbound_or_unsealed_batch(tmp_path,mutation):
    source,payload=frames_fixture(tmp_path)
    if mutation=='test': payload['frames'][-1]['split']='test'
    if mutation=='order': payload['calibration_first8_sample_ids'].reverse()
    if mutation=='heads': payload['frames'][0]['expected_heads']={}
    if mutation=='traces': payload['frames'][0]['expected_layers']={}
    if mutation=='repetitions': payload['repetitions']=1
    if mutation=='duplicate': payload['frames'][1]['sample_id']='0'
    if mutation=='missing': payload['files']={}
    (tmp_path/'frames.json').write_text(json.dumps(payload))
    with pytest.raises(ValueError): driver.validate_frames(tmp_path,source)


class FakeBuffer:
    def __init__(self): self.physical_address=0x100000000; self.events=[]
    def __setitem__(self,key,value): self.events.append('write')
    def flush(self): self.events.append('flush')
    def invalidate(self): self.events.append('invalidate')


class FakeCore:
    def __init__(self,stall=False): self.running=False; self.regs={}; self.stall=stall
    def write(self,offset,value):
        self.regs[offset]=value
        if offset==0:self.running=True
    def read(self,offset):
        if offset==0:return (0 if self.stall else 6) if self.running else 4
        return 0


def test_graph_node_timeout_does_not_invalidate_owned_buffer():
    core=FakeCore(stall=True); arena=FakeBuffer(); descriptor=FakeBuffer()
    with pytest.raises(TimeoutError):
        driver.execute_node(core,dict(arena=16,weights=32,biases=48,descriptor=64,**{'return':80}),arena,descriptor,np.zeros(32,dtype=np.int32),timeout=0.001)
    assert 'invalidate' not in arena.events


def test_graph_node_64bit_pointer_and_optional_invalidation():
    core=FakeCore(); arena=FakeBuffer(); descriptor=FakeBuffer()
    driver.execute_node(core,dict(arena=16,weights=32,biases=48,descriptor=64,**{'return':80}),arena,descriptor,np.zeros(32,dtype=np.int32),timeout=1,invalidate=False)
    assert core.regs[20]==1
    assert 'invalidate' not in arena.events


def test_oversized_hypothetical_bridge_rejected_not_assumed_real_geometry():
    shape=[96,320,320];size=int(np.prod(shape))
    graph=dict(inputs={'image':{}},edges={'image':dict(shape=shape,exp=-5)},nodes=[],
               outputs={'P3':'r2','P4':'r2','P5':'r2'},descriptors=[])
    for index in (1,2):
        source='image' if index==1 else 'r1'
        graph['edges']['r'+str(index)]=dict(shape=shape,exp=-5-index)
        graph['nodes'].append(dict(id='r'+str(index),op='requant',inputs=[source],
            input_exp=-4-index,output_exp=-5-index,multiplier=2,shift=0))
    graph['descriptors']=[dict(name=level,channels=96,height=320,width=320,real_bytes=size,
        padded_bytes=size,ddr_offset=index*size) for index,level in enumerate(graph['outputs'])]
    with pytest.raises(ValueError,match='capacity exceeded'):
        prepare.plan_graph(graph,dict.fromkeys(('weights.bin','biases.bin','layers.bin'),128))


def test_csim_admission_is_exact_not_substring_count():
    from tools.sonar_deploy.hardware import check_csim
    good='node=0 bytes=8 mismatch=0 status=0\nnode=1 bytes=4 mismatch=0 status=0\nFinished Command export_design'
    assert check_csim(good,2)['nodes']==2
    for text in (good.replace('node=1','node=0'),good.replace('mismatch=0','mismatch=1'),good+'\nERROR: fail'):
        with pytest.raises(ValueError):check_csim(text,2)


def test_route_parser_requires_actual_resources_and_both_timing_checks():
    from tools.sonar_deploy.hardware import parse_route
    timing='''    WNS(ns) TNS(ns) TNS Failing Endpoints TNS Total Endpoints WHS(ns) THS(ns)
    ------- ------- ---- ---- ------ ------
    0.186 0.000 0 100 0.010 0.000
All user specified timing constraints are met.
clk_pl_0 {0.000 2.667} 5.333 187.512
'''
    resources='''| CLB LUTs* | 24000 | 0 | 230400 | 10.42 |
| CLB Registers | 28000 | 0 | 460800 | 6.08 |
| DSPs | 81 | 0 | 1728 | 4.69 |
| Block RAM Tile | 51 | 0 | 312 | 16.35 |
'''
    out=parse_route(timing,resources)
    assert out['bram_used']==51 and out['lut_utilization_pct']==100*24000/230400
    assert out['actual_clock_mhz']==187.512
    with pytest.raises(ValueError):parse_route(timing.replace('0.010','-0.010'),resources)
    with pytest.raises(ValueError):parse_route(timing,resources.replace('DSPs','Unknown'))


def test_numeric_regression_source_bound_without_execution(tmp_path):
    from tools.detector_preflight.build_requant_regression import build as regression_build
    out=tmp_path/'numeric'
    result=regression_build(out)
    assert result['status']=='PREPARED_NOT_EXECUTED'
    assert result['files']['h2_graph.cpp']==build.sha(ROOT/'tools/detector_preflight/h2_graph.cpp')
    assert result['files']['numeric_v1.h']==build.sha(ROOT/'tools/fpga_h2/numeric_v1.h')
    assert 'cosim_design' in (out/'run.tcl').read_text()


def test_timing02_build_binds_new_schedule_without_numeric_or_clock_change(tmp_path):
    source=tmp_path/'package';source.mkdir()
    graph=fake_real_package(source)
    # Metadata/build fixture only: zero values are not hardware parity evidence.
    values={name:np.zeros(edge['shape'],dtype=np.int8) for name,edge in graph['edges'].items()}
    np.save(source/'input.npy',values['image'],allow_pickle=False)
    out=tmp_path/'build'
    manifest=build.build(out,source,source/'input.npy',values=values)
    assert manifest['implementation']['executor_version']=='serial_ddr_graph_v2_requant_timing02'
    assert manifest['implementation']['requant_pipeline']=='off'
    assert manifest['implementation']['requested_clock_mhz']==200
    assert 'create_clock -period 5' in (out/'run_graph.tcl').read_text()
    bd=(out/'build_graph_bd.tcl').read_text()
    assert 'serial DDR graph executor timing PASS; candidate board validation pending' in bd
    assert 'scoped H2 DMA bring-up only; not full detector' not in bd
    assert (out/'numeric_v1.h').read_bytes()==(ROOT/'tools/fpga_h2/numeric_v1.h').read_bytes()
    core=(out/'h2_graph.cpp').read_text()
    requant=core.split('if(op==4) {',1)[1].split('if(op==0) {',1)[0]
    assert '#pragma HLS PIPELINE off' in requant and '#pragma HLS PIPELINE II=1' not in requant
    assert 'h2::requant(ap_int<32>(arena[a+i]),d[16],d[17])' in requant
    assert '#pragma HLS PIPELINE II=1' in core.split('if(op==0) {',1)[1]


def test_hardware_dispatch_obeys_software_admission(monkeypatch,tmp_path):
    from tools.sonar_deploy import hardware
    denied=dict(status='BLOCKED_EVIDENCE',allowed=False)
    monkeypatch.setattr(hardware.gate,'action_admission',lambda *args:denied)
    monkeypatch.setattr(hardware,'_run',lambda *args:pytest.fail('must not launch'))
    assert hardware.execute_stage(tmp_path,'D','hls')==denied


def test_board_batch_retains_all_heads_padding_and_raw_timing(monkeypatch,tmp_path):
    source,frames=frames_fixture(tmp_path)
    value=np.arange(4,dtype=np.int8)
    digest=hashlib.sha256(value.tobytes()).hexdigest()
    for row in frames['frames']:
        row['expected_heads']={'P3':digest};row['expected_layers']={'layer':digest}
    source['plan']['allocations']={'image':dict(shape=[1,2,2],offset=0)}
    source['plan']['outputs']=[dict(level='P3',arena_offset=64,real_bytes=4,tail_zero_bytes=28)]
    class Arena(np.ndarray):
        def flush(self):pass
        def invalidate(self):pass
    arena=np.full(128,99,dtype=np.int8).view(Arena)
    words=np.zeros((1,32),dtype=np.int32);words[0,13]=64;words[0,25]=4
    calls=[]
    def run_node(core,abi,arena,descriptor,words,timeout,invalidate=True):
        arena[64:68]=arena[:4];calls.append(invalidate);return .001
    monkeypatch.setattr(driver,'execute_node',run_node)
    output=tmp_path/'results';output.mkdir()
    status=driver.execute_frames(tmp_path,source,frames,None,{},arena,None,words,1,output)
    assert status['train_frames']==8 and status['valid_frames']==99
    assert len(calls)==107+11 and calls[:8]==[True]*8 and calls[8:]==[False]*(99+11)
    records=json.loads((output/'frames.json').read_text());assert len(records)==107
    assert (output/'frames/106/P3.bin').read_bytes()==value.tobytes()
    assert (output/'frames/106/P3.padded.bin').read_bytes()==value.tobytes()+bytes(28)
    timing=json.loads((output/'benchmark.json').read_text())
    assert timing['warmup']==1 and len(timing['samples'])==10
    assert timing['end_to_end_status']=='PENDING_HOST_PREPROCESS_DECODE_TIMING'


def session_fixture(monkeypatch,tmp_path):
    implementation=dict(executor_version='serial_ddr_graph_v2_requant',arena_capacity_bytes=16777216,supported_opcodes=[0,1,2,3,4])
    source=dict(real_weights_bound=True,implementation=implementation,layers=[dict(node='layer')],plan=dict(
        allocations={'image':dict(shape=[1,2,2],offset=0)},
        outputs=[dict(level='P3',arena_offset=64,real_bytes=4)]))
    (tmp_path/'source_manifest.json').write_text(json.dumps(source))
    np.zeros(128,dtype=np.int8).tofile(tmp_path/'arena.bin')
    words=np.zeros((1,32),dtype='<i4');words[0,0]=4;words[0,22]=128;words[0,26]=build.MAGIC
    words.tofile(tmp_path/'descriptors.bin')
    manifest=dict(files={name:build.sha(tmp_path/name) for name in ['source_manifest.json','arena.bin','descriptors.bin']},
        executor=implementation,
        implemented_clock_mhz=187.5,abi={})
    (tmp_path/'board_manifest.json').write_text(json.dumps(manifest))
    allocated=[]
    class Buffer(np.ndarray):
        def flush(self):pass
        def invalidate(self):pass
        def freebuffer(self):self.freed=True
    def allocate(shape,dtype):
        value=np.zeros(shape,dtype=dtype).view(Buffer);value.freed=False;allocated.append(value);return value
    class Overlay:
        def __init__(self,*args,**kwargs):self.ip_dict={'core':dict(type='h2_graph',phys_addr=0,addr_range=4096)}
        def download(self):pass
    monkeypatch.setitem(sys.modules,'pynq',types.SimpleNamespace(Overlay=Overlay,allocate=allocate,
        MMIO=lambda *args:FakeCore(),Clocks=types.SimpleNamespace(fclk0_mhz=187.5)))
    return allocated


def test_graph_session_reuses_weights_across_rpc_calls(monkeypatch,tmp_path):
    buffers=session_fixture(monkeypatch,tmp_path)
    def node(core,abi,arena,descriptor,words,timeout,invalidate=True):arena[64:68]=arena[:4]
    monkeypatch.setattr(driver,'execute_node',node)
    session=driver.GraphSession(tmp_path)
    for i in range(11):
        heads,timing=session.infer(np.full((1,2,2),i,dtype=np.int8))
        assert heads['P3']==bytes([i])*4 and timing['board_clock_mhz']==187.5
    assert len(buffers)==2
    session.close();assert all(buffer.freed for buffer in buffers)


def test_graph_session_timeout_quarantines_device_owned_buffers(monkeypatch,tmp_path):
    buffers=session_fixture(monkeypatch,tmp_path)
    def node(*args,**kwargs):raise TimeoutError('device owned')
    monkeypatch.setattr(driver,'execute_node',node)
    session=driver.GraphSession(tmp_path)
    with pytest.raises(TimeoutError):session.infer(np.zeros((1,2,2),dtype=np.int8))
    session.close();assert all(not buffer.freed for buffer in buffers)
    with pytest.raises(RuntimeError):session.infer(np.zeros((1,2,2),dtype=np.int8))


def test_install_frame_bundle_is_immutable_and_hash_bound(tmp_path):
    from tools.sonar_deploy.frames import install_frames
    build_dir=tmp_path/'build';build_dir.mkdir()
    bundle=tmp_path/'bundle';bundle.mkdir()
    (build_dir/'manifest.json').write_text(json.dumps(dict(graph_sha256='g')))
    (bundle/'image.npy').write_bytes(b'input')
    frames=dict(graph_sha256='g',files={'image.npy':build.sha(bundle/'image.npy')})
    (bundle/'frames.json').write_text(json.dumps(frames))
    assert install_frames(build_dir,bundle)['status']=='FRAMES_BOUND'
    assert install_frames(build_dir,bundle)['status']=='FRAMES_BOUND'
    (build_dir/'image.npy').write_bytes(b'changed')
    with pytest.raises(ValueError,match='overwrite'):install_frames(build_dir,bundle)


def test_install_frame_bundle_rejects_different_graph(tmp_path):
    from tools.sonar_deploy.frames import install_frames
    build_dir=tmp_path/'build';build_dir.mkdir();bundle=tmp_path/'bundle';bundle.mkdir()
    (build_dir/'manifest.json').write_text(json.dumps(dict(graph_sha256='a')))
    (bundle/'frames.json').write_text(json.dumps(dict(graph_sha256='b',files={})))
    with pytest.raises(ValueError,match='graph mismatch'):install_frames(build_dir,bundle)


def common_build_fixture(root,network):
    folder=root/network/'seed_42/hardware/graph_build';folder.mkdir(parents=True)
    for name in ('h2_graph.cpp','numeric_v1.h','run_graph.tcl','build_graph_bd.tcl'):
        (folder/name).write_text('identical '+name)
    graph=dict(network=network,numeric_contract_hash='a'*64,interface_contract_hash='b'*64)
    (folder/'graph.json').write_text(json.dumps(graph))
    for name in ('arena.bin','descriptors.bin','expected_layers.bin'):
        (folder/name).write_bytes((network+name).encode())
    (folder/'hls.log').write_text('node=0 bytes=8 mismatch=0 status=0\nFinished Command export_design\n')
    rtl=folder/'hls_project/solution1/syn/verilog';rtl.mkdir(parents=True)
    (rtl/'h2_graph.v').write_text('module h2_graph(input a,output b);assign b=a;endmodule')
    headers=folder/'hls_project/solution1/impl/ip/drivers/h2_graph_v1_0/src';headers.mkdir(parents=True)
    text='\n'.join(f'#define XH2_GRAPH_CONTROL_ADDR_{name}_DATA {hex(16+i*12)}\n#define XH2_GRAPH_CONTROL_BITS_{name}_DATA 64'
                   for i,name in enumerate(('ARENA','WEIGHTS','BIASES','DESCRIPTOR')))
    text+='\n#define XH2_GRAPH_CONTROL_ADDR_AP_RETURN 0x4c\n'
    (headers/'xh2_graph_hw.h').write_text(text)
    manifest=dict(real_weights_bound=True,graph_sha256=build.sha(folder/'graph.json'),
        implementation=dict(executor_version='serial_ddr_graph_v2_requant',arena_capacity_bytes=16777216,requested_clock_mhz=200),
        plan=dict(executor_arena_capacity_bytes=16777216,arena_required_bytes=256,
                  outputs=[dict(edge='head')],allocations={'head':dict(last_use=1)}),layers=[dict(node='head')])
    manifest['files']={p.name:build.sha(p) for p in folder.iterdir() if p.is_file() and p.name!='hls.log'}
    (folder/'manifest.json').write_text(json.dumps(manifest))
    return folder


def rebind_fixture(build_dir):
    path=build_dir/'manifest.json';manifest=json.loads(path.read_text())
    for name in manifest['files']:manifest['files'][name]=build.sha(build_dir/name)
    manifest['graph_sha256']=build.sha(build_dir/'graph.json')
    path.write_text(json.dumps(manifest))


def test_common_executor_has_same_rtl_abi_without_same_graph(tmp_path):
    from tools.sonar_deploy.hardware import verify_common_executor
    donor=common_build_fixture(tmp_path,'D');candidate=common_build_fixture(tmp_path,'C')
    result=verify_common_executor(candidate,donor)
    assert result['status']=='PASS' and result['weights_reused'] is False
    assert result['candidate_graph_sha256']!=result['donor_graph_sha256']


@pytest.mark.parametrize('mutation',['rtl','core','clock','abi','capacity','heads','numeric','csim'])
def test_common_executor_rejects_unmatched_hardware(tmp_path,mutation):
    from tools.sonar_deploy.hardware import verify_common_executor
    donor=common_build_fixture(tmp_path,'D');candidate=common_build_fixture(tmp_path,'B1')
    path=candidate/'manifest.json';manifest=json.loads(path.read_text())
    if mutation=='rtl':(candidate/'hls_project/solution1/syn/verilog/h2_graph.v').write_text('module h2_graph(input a,output b);assign b=~a;endmodule')
    if mutation=='core':(candidate/'h2_graph.cpp').write_text('different core')
    if mutation=='clock':manifest['implementation']['requested_clock_mhz']=100
    if mutation=='abi':
        header=next(candidate.rglob('xh2_graph_hw.h'));header.write_text(header.read_text().replace('_BITS_ARENA_DATA 64','_BITS_ARENA_DATA 32'))
    if mutation=='capacity':manifest['plan']['arena_required_bytes']=16777217
    if mutation=='heads':manifest['plan']['allocations']['head']['last_use']=0
    if mutation=='numeric':
        graph=json.loads((candidate/'graph.json').read_text());graph['numeric_contract_hash']='c'*64
        (candidate/'graph.json').write_text(json.dumps(graph))
    if mutation=='csim':(candidate/'hls.log').write_text('node=0 bytes=8 mismatch=1 status=0\nFinished Command export_design\n')
    path.write_text(json.dumps(manifest));rebind_fixture(candidate)
    with pytest.raises(ValueError):verify_common_executor(candidate,donor)


def test_route_reuse_preserves_candidate_parameters_and_binds_d_route(monkeypatch,tmp_path):
    from tools.sonar_deploy import hardware
    donor=common_build_fixture(tmp_path,'D');candidate=common_build_fixture(tmp_path,'C')
    donor_route=tmp_path/'D/seed_42/route.json';donor_route.write_text('{"status":"PASS"}')
    (tmp_path/'DEPLOYMENT_BINDING.json').write_text('{"test_evaluated":false}')
    copied=('timing.rpt','utilization.rpt','utilization_hier.rpt','route_status.rpt','ROUTE_TIMING_PASS','route.log','h2_graph.xsa')
    for name in copied:(donor/name).write_text('actual donor '+name)
    evidence=dict(artifacts={name:dict(path=name,sha256=build.sha(donor/name)) for name in copied})
    checked=[]
    def stage(root,network,name,result):
        checked.append((network,name));return evidence if network=='D' else dict(status='PASS')
    monkeypatch.setattr(hardware.gate,'stage_evidence',stage)
    monkeypatch.setattr(hardware.gate,'load_binding',lambda root:dict(runs=[dict(network='D',seed=42)]))
    monkeypatch.setattr(hardware.gate,'load_result',lambda *args:dict(package_manifest_sha256='d'*64))
    before={name:build.sha(candidate/name) for name in ('arena.bin','graph.json','descriptors.bin','manifest.json')}
    result=hardware.reuse_common_route(tmp_path,'C',dict(package_manifest_sha256='c'*64),candidate)
    assert result['common_image_from']=='D' and result['vivado_executed_for_candidate'] is False
    assert checked==[('C','hls'),('D','route')]
    assert all(build.sha(candidate/name)==digest for name,digest in before.items())
    assert all((candidate/name).read_bytes()==(donor/name).read_bytes() for name in copied)
    proof=json.loads((candidate/'COMMON_IMAGE_BINDING.json').read_text())
    assert proof['donor_route_sha256']==build.sha(donor_route)
    assert proof['candidate_package_manifest_sha256']=='c'*64
    assert hardware.reuse_common_route(tmp_path,'C',dict(package_manifest_sha256='c'*64),candidate)==result


def test_common_route_cannot_overwrite_existing_different_artifact(tmp_path):
    from tools.sonar_deploy.hardware import _copy_frozen
    a=tmp_path/'a';b=tmp_path/'b';a.write_text('donor');b.write_text('different')
    with pytest.raises(ValueError,match='refuse replacement'):_copy_frozen(a,b)
    assert b.read_text()=='different'


def test_common_route_waits_for_admitted_d_route(monkeypatch,tmp_path):
    from tools.sonar_deploy import hardware
    candidate=common_build_fixture(tmp_path,'B1')
    def stage(root,network,name,result):
        if network=='D':raise ValueError('D route incomplete')
        return dict(status='PASS')
    monkeypatch.setattr(hardware.gate,'stage_evidence',stage)
    monkeypatch.setattr(hardware.gate,'load_binding',lambda root:dict(runs=[dict(network='D',seed=42)]))
    monkeypatch.setattr(hardware.gate,'load_result',lambda *args:dict(package_manifest_sha256='d'*64))
    with pytest.raises(ValueError,match='D route incomplete'):
        hardware.reuse_common_route(tmp_path,'B1',dict(package_manifest_sha256='b'*64),candidate)
    assert not (candidate/'h2_graph.xsa').exists()


@pytest.mark.parametrize('status',['WAITING_FOR_LOCAL_BOARD_RESULT','PASS'])
def test_board_dispatch_uses_bound_result_verifier(monkeypatch,tmp_path,status):
    import sys
    import types
    from tools.sonar_deploy import hardware
    calls=[]
    expected=dict(status=status,network='D',test_evaluated=False)
    def verify(root,network):
        calls.append((root,network))
        return expected
    monkeypatch.setitem(sys.modules,'tools.sonar_deploy.board_result',types.SimpleNamespace(verify_board_result=verify))
    monkeypatch.setattr(hardware.gate,'action_admission',lambda *args:dict(allowed=True))
    monkeypatch.setattr(hardware.gate,'load_binding',lambda root:dict(runs=[dict(network='D',seed=42)]))
    monkeypatch.setattr(hardware.gate,'load_result',lambda *args:dict(package_manifest_sha256='d'*64))
    def forbidden(*args,**kwargs):raise AssertionError('board import must not launch HLS or Vivado')
    monkeypatch.setattr(hardware,'_run',forbidden)
    assert hardware.execute_stage(tmp_path,'D','board') is expected
    assert calls==[(tmp_path.resolve(),'D')]
