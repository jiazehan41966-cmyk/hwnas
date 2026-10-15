"""Native 2022.2 execution handlers for admitted, real INT8 detector graphs.

Does not SSH from the CUDA host back into itself. Board programming remains a
separate Windows/Jupyter transaction; no synthetic evidence is promoted.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'tools/detector_preflight')]
from tools.sonar_deploy import admission as gate
from tools.detector_preflight.build_graph_witness import build as build_graph, validate_package
from tools.detector_preflight.build_requant_regression import build as build_regression
from tools.detector_preflight.rtl_equivalence import compare_rtl
from tools.detector_preflight.prepare import plan_graph

HLS = Path('/opt/Xilinx/Vitis_HLS/2022.2/bin/vitis_hls')
VIVADO = Path('/opt/Xilinx/Vivado/2022.2/bin/vivado')


def audit_metadata_capacities(root, output):
    """Exact allocator plan from exported metadata when binary transfer is pending.

    This never substitutes for validate_package or hardware action admission.
    The canonical builder aligns before each tensor and adds no trailing pad.
    """
    rows=[]
    for network in gate.NETWORKS:
        package=Path(root)/network/'seed_42/package'
        graph=gate.read_json(package/'graph.json');manifest=gate.read_json(package/'manifest.json')
        if gate.sha256(package/'graph.json')!=manifest['files']['graph.json']:
            raise ValueError('graph SHA differs from fetched package manifest')
        conv=[node for node in graph['nodes'] if node['op']=='conv']
        sizes={'weights.bin':max(n['weight_offset']+math.prod(n['weight_shape']) for n in conv),
               'biases.bin':max(n['bias_offset']+4*n['bias_count'] for n in conv),
               'layers.bin':128*(1+max(n['layer_id'] for n in conv))}
        plan=plan_graph(graph,sizes)
        rows.append(dict(network=network,seed=42,graph_sha256=manifest['files']['graph.json'],
            nodes=len(graph['nodes']),requant_nodes=sum(n['op']=='requant' for n in graph['nodes']),
            declared_binary_sizes=sizes,arena_required_bytes=plan['arena_required_bytes'],
            activation_high_water_bytes=plan['activation_high_water_bytes'],
            executor_arena_capacity_bytes=plan['executor_arena_capacity_bytes'],
            all_heads_retained=all(plan['allocations'][h['edge']]['last_use']==len(graph['nodes']) for h in plan['outputs']),
            levels=list(graph['outputs']),legacy_16mib_fits=plan['arena_required_bytes']<=16777216))
    report=dict(status='METADATA_PLAN_PASS',models=rows,test_evaluated=False,
                binaries_verified=False,hardware_executed=False,
                correction='The earlier 96-channel stage1 estimate was wrong: frozen stage1 expands 16 to 48 channels; all actual plans fit 16 MiB. No capacity increase implemented.',
                limitation='Binary lengths derived from bound graph offset/count metadata; actual binary SHA remains a separate export/hardware gate')
    gate.atomic_json(output,report)
    return report


def audit_capacities(root):
    """Measure all three real seed42 packages, not synthetic approximations."""
    root=Path(root).resolve();binding=gate.load_binding(root);rows=[]
    for network in gate.NETWORKS:
        expected=next(r for r in binding['runs'] if r['network']==network and r['seed']==42)
        result=gate.load_result(root,expected)
        path=Path(result['package_manifest_path']);package=path.parent if path.is_absolute() else (root/path).parent
        graph,_=validate_package(package)
        plan=plan_graph(graph,{name:(package/name).stat().st_size for name in ('weights.bin','biases.bin','layers.bin')})
        retained=all(plan['allocations'][o['edge']]['last_use']==len(graph['nodes']) for o in plan['outputs'])
        if not retained:raise ValueError('output head lifetime not retained')
        rows.append(dict(network=network,seed=42,graph_sha256=gate.sha256(package/'graph.json'),
            nodes=len(graph['nodes']),requant_nodes=sum(n['op']=='requant' for n in graph['nodes']),
            arena_required_bytes=plan['arena_required_bytes'],activation_high_water_bytes=plan['activation_high_water_bytes'],
            executor_arena_capacity_bytes=plan['executor_arena_capacity_bytes'],all_heads_retained=True,
            legacy_16mib_fits=plan['arena_required_bytes']<=16777216,levels=[o['level'] for o in plan['outputs']]))
    report=dict(status='PASS',models=rows,test_evaluated=False,
        note='Real B1/C/D plans retain the existing 16 MiB arena; explicit requant still requires a new synthesized image')
    gate.atomic_json(root/'GRAPH_CAPACITY_AUDIT.json',report)
    return report


def check_csim(log, expected_nodes):
    rows = re.findall(r'node=(\d+) bytes=(\d+) mismatch=(\d+) status=(\d+)',log)
    if len(rows) != expected_nodes or [int(r[0]) for r in rows] != list(range(expected_nodes)):
        raise ValueError('full graph C simulation node coverage incomplete')
    if any(int(r[2]) or int(r[3]) for r in rows) or 'ERROR:' in log:
        raise ValueError('full graph C simulation failed')
    if 'Finished Command export_design' not in log:
        raise ValueError('HLS IP export incomplete')
    return dict(nodes=len(rows),mismatch_count=0)


def parse_route(timing, utilization):
    match = re.search(r'^\s*WNS\(ns\)\s+TNS\(ns\)[^\n]*\n[^\n]*\n\s*([-\d.]+)\s+([-\d.]+)\s+\d+\s+\d+\s+([-\d.]+)\s+([-\d.]+)',timing,re.M)
    clock = re.search(r'^clk_pl_0\s+\{[^}]+\}\s+([\d.]+)\s+([\d.]+)',timing,re.M)
    if not match or not clock:
        raise ValueError('required timing/clock report fields missing')
    wns,tns,whs,ths=map(float,match.groups())
    if min(wns,tns,whs,ths)<0 or 'All user specified timing constraints are met.' not in timing:
        raise ValueError('setup/hold timing failed')
    values=dict(wns_ns=wns,tns_ns=tns,whs_ns=whs,ths_ns=ths,timing_met=True,
                actual_clock_mhz=float(clock[2]),implemented_period_ns=float(clock[1]))
    labels={'lut':r'(?:CLB|Slice) LUTs\*?','ff':r'(?:CLB Registers|Slice Registers)\*?',
            'dsp':r'DSPs','bram':r'Block RAM Tile'}
    # Vivado repeats CLB Registers in Logic Distribution. Read the summary
    # section, not every matching line in the report or its table of contents.
    lines=utilization.splitlines();section=None;scoped=[]
    for index,line in enumerate(lines):
        title=re.match(r'^\s*\d+(?:\.\d+)*\.?\s+(.+?)\s*$',line)
        if title and index+1<len(lines) and re.fullmatch(r'-{3,}',lines[index+1].strip()):
            section=title[1].strip().upper()
        scoped.append((section,line))
    has_sections=any(section is not None for section,_ in scoped)
    sections={'lut':{'CLB LOGIC','SLICE LOGIC'},'ff':{'CLB LOGIC','SLICE LOGIC'},
              'bram':{'BLOCKRAM','BLOCK RAM'},'dsp':{'ARITHMETIC'}}
    for name,label in labels.items():
        rows=[line for section,line in scoped if (not has_sections or section in sections[name])
              and re.match(r'^\s*\|\s*'+label+r'\s*\|',line)]
        if len(rows)!=1:raise ValueError('required utilization row missing/ambiguous: '+name)
        cells=[c.strip() for c in rows[0].strip().split('|')[1:-1]]
        used=float(cells[1].replace(',',''));capacity=float(cells[-2].replace(',',''))
        if not math.isfinite(used) or not math.isfinite(capacity) or used<0 or capacity<=0 or used>capacity:
            raise ValueError('invalid resource capacity: '+name)
        values[name+'_used']=used;values[name+'_capacity']=capacity
        values[name+'_utilization_pct']=100*used/capacity
    return values


def _completed_route_execution(build):
    """Validate native route completion without executing or rewriting tools."""
    build=Path(build).resolve();manifest=_verify_build(build)
    check_csim((build/'hls.log').read_text(errors='replace'),len(manifest['layers']))
    names=('timing.rpt','utilization.rpt','utilization_hier.rpt','route_status.rpt',
           'ROUTE_TIMING_PASS','route.log','route.log.process.json','h2_graph.xsa')
    for name in names:
        if not (build/name).is_file() or (build/name).stat().st_size==0:
            raise ValueError('completed route artifact missing/empty: '+name)
    meta=gate.read_json(build/'route.log.process.json')
    command=[str(VIVADO),'-mode','batch','-source','build_graph_bd.tcl']
    if (meta.get('command')!=command or Path(meta.get('cwd','')).resolve()!=build or
            type(meta.get('returncode')) is not int or meta['returncode']!=0):
        raise ValueError('native route process not completed successfully for this build')
    elapsed=meta.get('elapsed_seconds')
    if isinstance(elapsed,bool) or not isinstance(elapsed,(int,float)) or not math.isfinite(elapsed) or elapsed<=0:
        raise ValueError('native route elapsed time missing/invalid')
    log=(build/'route.log').read_text(errors='replace')
    if (re.search(r'^\s*(?:ERROR|FATAL):',log,re.M) or
            'write_bitstream completed successfully' not in log or
            'Successfully created Hardware Platform:' not in log or
            not re.match(r'^INFO: \[Common 17-206\] Exiting Vivado at ',log.rstrip().splitlines()[-1])):
        raise ValueError('native route log incomplete or contains a tool error')
    marker=(build/'ROUTE_TIMING_PASS').read_text()
    if 'Tool 2022.2;' not in marker or 'serial DDR graph executor timing PASS; candidate board validation pending' not in marker:
        raise ValueError('completed route marker does not describe this detector executor')
    status=(build/'route_status.rpt').read_text()
    def nets(label):
        match=re.search(r'# of '+label+r'\.+\s*:\s*(\d+)\s*:',status)
        if not match:raise ValueError('route net status missing: '+label)
        return int(match[1])
    routable=nets('routable nets')
    if routable<=0 or nets('fully routed nets')!=routable or nets('nets with routing errors')!=0:
        raise ValueError('route net coverage incomplete or routing errors present')
    with zipfile.ZipFile(build/'h2_graph.xsa') as archive:
        for name in ('h2_graph.bit','h2_graph_bd.hwh'):
            if archive.namelist().count(name)!=1 or not archive.read(name):
                raise ValueError('completed XSA missing unique bitstream/HWH')
    parse_route((build/'timing.rpt').read_text(),(build/'utilization.rpt').read_text())
    return dict(command=command,reused_existing_output=True,finalized_existing_artifacts=True,
                vivado_executed_by_finalizer=False,elapsed_seconds=elapsed,
                completion_artifact_sha256={name:gate.sha256(build/name) for name in names})


def _run(binary, arguments, folder, logfile, timeout):
    if sys.platform!='linux' or not binary.is_file():
        raise RuntimeError('HARDWARE_TOOLCHAIN_NOT_READY: native Linux '+str(binary))
    command=[str(binary),*arguments]
    if (folder/logfile).exists():
        meta=gate.read_json(folder/(logfile+'.process.json'))
        proc=Path('/proc')/str(meta['pid'])
        if proc.exists() and (proc/'cwd').resolve()==folder.resolve():
            raise RuntimeError('ALREADY_RUNNING: '+str(meta['pid']))
        if meta.get('returncode',0)!=0:
            raise RuntimeError('Prior tool failed; preserve evidence and use an explicit new attempt')
        return dict(command=command,reused_existing_output=True,
                    elapsed_seconds=meta.get('elapsed_seconds'))
    begin=time.time()
    with (folder/logfile).open('x') as log:
        process=subprocess.Popen(command,cwd=folder,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        gate.atomic_json(folder/(logfile+'.process.json'),dict(pid=process.pid,command=command,cwd=str(folder),start_time=begin))
        try:
            returncode=process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Keep process identity and evidence. Caller can inspect; no duplicate retry.
            raise TimeoutError('Tool still running past handler deadline; inspect '+str(folder/(logfile+'.process.json')))
    gate.atomic_json(folder/(logfile+'.process.json'),dict(pid=process.pid,command=command,cwd=str(folder),
        start_time=begin,returncode=returncode,elapsed_seconds=time.time()-begin))
    if returncode:raise RuntimeError('tool failed with exit '+str(returncode)+': '+str(folder/logfile))
    return dict(command=command,elapsed_seconds=time.time()-begin)


def _artifact(path, base):
    path=Path(path).resolve()
    return dict(path=os.path.relpath(path,base),sha256=gate.sha256(path))


def _trace_manifest(result_dir, parity):
    found=[]
    for item in parity['artifacts'].values():
        path=Path(item['path']);path=path if path.is_absolute() else result_dir/path
        if path.name=='TRACE_MANIFEST.json':found.append(path)
    if len(found)!=1:raise ValueError('integer_parity must bind one TRACE_MANIFEST.json artifact')
    manifest=gate.read_json(found[0])
    if manifest.get('split')!='train' or manifest.get('count')!=8 or manifest.get('test_evaluated') is not False:
        raise ValueError('fixed train8 trace required')
    for name,digest in manifest['files'].items():
        path=(found[0].parent/name).resolve()
        if not path.is_relative_to(found[0].parent.resolve()) or gate.sha256(path)!=digest:
            raise ValueError('bound trace file changed: '+name)
    return found[0],manifest


def _verify_build(build):
    manifest=gate.read_json(build/'manifest.json')
    for name,digest in manifest['files'].items():
        if gate.sha256(build/name)!=digest:raise ValueError('build input changed: '+name)
    return manifest


def _control_abi(build):
    """Extract generated register addresses and widths, not guessed offsets."""
    headers=[]
    for path in (Path(build)/'hls_project/solution1/impl/ip/drivers').rglob('*_hw.h'):
        fields=dict(re.findall(r'#define\s+(XH2_GRAPH_CONTROL_(?:ADDR|BITS)_\w+)\s+(0x[0-9a-fA-F]+|\d+)',path.read_text()))
        if fields:headers.append((path,{key:int(value,0) for key,value in fields.items()}))
    if len(headers)!=1:raise ValueError('exactly one generated H2 graph control ABI required')
    path,fields=headers[0]
    for label in ('ARENA','WEIGHTS','BIASES','DESCRIPTOR'):
        if f'XH2_GRAPH_CONTROL_ADDR_{label}_DATA' not in fields or fields.get(f'XH2_GRAPH_CONTROL_BITS_{label}_DATA')!=64:
            raise ValueError('64-bit generated AXI pointer ABI required: '+label)
    if 'XH2_GRAPH_CONTROL_ADDR_AP_RETURN' not in fields:raise ValueError('generated return-register ABI missing')
    return dict(fields=fields,header_sha256=gate.sha256(path),header_path=str(path))


def verify_common_executor(candidate_build, donor_build):
    """Prove parameter-independent executor identity before reusing a routed image."""
    candidate_build,donor_build=Path(candidate_build),Path(donor_build)
    candidate,donor=_verify_build(candidate_build),_verify_build(donor_build)
    if not candidate.get('real_weights_bound') or not donor.get('real_weights_bound'):
        raise ValueError('common image requires actual trained package bindings')
    implementation=candidate.get('implementation')
    if not implementation or implementation!=donor.get('implementation'):
        raise ValueError('common image implementation/clock/capacity contract differs')
    capacity=implementation.get('arena_capacity_bytes')
    if type(capacity) is not int or capacity<=0:raise ValueError('missing executor arena capacity')
    for manifest in (candidate,donor):
        plan=manifest['plan']
        if plan.get('executor_arena_capacity_bytes')!=capacity or not 0<plan['arena_required_bytes']<=capacity:
            raise ValueError('common image arena contract/capacity mismatch')
        if any(plan['allocations'][output['edge']]['last_use']!=len(manifest['layers']) for output in plan['outputs']):
            raise ValueError('common image cannot discard a live output head')
    source_hashes={}
    for name in ('h2_graph.cpp','numeric_v1.h','run_graph.tcl','build_graph_bd.tcl'):
        left,right=gate.sha256(candidate_build/name),gate.sha256(donor_build/name)
        if left!=right:raise ValueError('common image source/tool configuration differs: '+name)
        source_hashes[name]=left
    graph_candidate=gate.read_json(candidate_build/'graph.json');graph_donor=gate.read_json(donor_build/'graph.json')
    for key in ('numeric_contract_hash','interface_contract_hash'):
        if not graph_candidate.get(key) or graph_candidate[key]!=graph_donor.get(key):
            raise ValueError('common image numeric/interface contract differs: '+key)
    abi_candidate,abi_donor=_control_abi(candidate_build),_control_abi(donor_build)
    if abi_candidate['fields']!=abi_donor['fields']:raise ValueError('common image generated ABI differs')
    check_csim((candidate_build/'hls.log').read_text(errors='replace'),len(candidate['layers']))
    check_csim((donor_build/'hls.log').read_text(errors='replace'),len(donor['layers']))
    rtl=compare_rtl(candidate_build/'hls_project/solution1/syn/verilog',donor_build/'hls_project/solution1/syn/verilog')
    return dict(status='PASS',scope='shared parameter-independent serial DDR executor only',
        candidate_graph_sha256=candidate['graph_sha256'],donor_graph_sha256=donor['graph_sha256'],
        source_sha256=source_hashes,implementation=implementation,abi=abi_candidate['fields'],
        candidate_abi_header_sha256=abi_candidate['header_sha256'],donor_abi_header_sha256=abi_donor['header_sha256'],
        candidate_arena_bytes=candidate['plan']['arena_required_bytes'],rtl_equivalence=rtl,
        weights_reused=False,graph_reused=False,test_evaluated=False)


def _copy_frozen(source, destination):
    """Copy unchanged evidence without overwriting a different artifact."""
    source,destination=Path(source),Path(destination)
    if destination.exists():
        if gate.sha256(source)!=gate.sha256(destination):raise ValueError('refuse replacement of differing route artifact: '+str(destination))
        return
    temporary=destination.with_name(destination.name+f'.{os.getpid()}.tmp')
    shutil.copy2(source,temporary)
    if gate.sha256(source)!=gate.sha256(temporary):raise ValueError('route evidence copy hash mismatch')
    os.replace(temporary,destination)


def reuse_common_route(root, network, result, build):
    """Reuse D's measured common shell only after this candidate's own HLS PASS."""
    root,build=Path(root).resolve(),Path(build)
    if network not in ('C','B1'):raise ValueError('only C/B1 reuse the D common image')
    # Both stage validators recursively bind checkpoints, software parity, logs,
    # own candidate package, deployment root, and durable stage metrics.
    gate.stage_evidence(root,network,'hls',result)
    binding=gate.load_binding(root)
    expected=next(row for row in binding['runs'] if row['network']=='D' and row['seed']==42)
    donor_result=gate.load_result(root,expected)
    donor_evidence=gate.stage_evidence(root,'D','route',donor_result)
    donor_folder=root/'D/seed_42';donor_build=donor_folder/'hardware/graph_build'
    comparison=verify_common_executor(build,donor_build)
    immutable_before={name:gate.sha256(build/name) for name in ('manifest.json','graph.json','arena.bin','descriptors.bin','expected_layers.bin')}
    copied=('timing.rpt','utilization.rpt','utilization_hier.rpt','route_status.rpt','ROUTE_TIMING_PASS','route.log','h2_graph.xsa')
    # The common image must be the exact one already admitted by D's route.
    bound={item['sha256'] for item in donor_evidence['artifacts'].values()}
    for name in copied:
        if gate.sha256(donor_build/name) not in bound:raise ValueError('donor route does not bind artifact: '+name)
        _copy_frozen(donor_build/name,build/name)
    if any(gate.sha256(build/name)!=digest for name,digest in immutable_before.items()):
        raise ValueError('candidate graph/parameters changed while copying shared shell')
    comparison.update(common_image_from='D',donor_route_sha256=gate.sha256(donor_folder/'route.json'),
        donor_package_manifest_sha256=donor_result['package_manifest_sha256'],
        candidate_package_manifest_sha256=result['package_manifest_sha256'],
        deployment_binding_sha256=gate.sha256(root/'DEPLOYMENT_BINDING.json'),
        copied_artifacts={name:gate.sha256(build/name) for name in copied},candidate_artifacts_unchanged=immutable_before)
    proof=build/'COMMON_IMAGE_BINDING.json'
    if proof.exists() and gate.read_json(proof)!=comparison:raise ValueError('common image binding changed')
    gate.atomic_json(proof,comparison)
    return dict(command=None,common_image_from='D',vivado_executed_for_candidate=False,
                donor_route_sha256=comparison['donor_route_sha256'],
                reason='same source, tool configuration, generated ABI and synthesized RTL; own candidate Csim/HLS PASS')


def _publish(root,network,stage,result,artifacts,extra,metric_values=None):
    folder=root/network/'seed_42';previous=gate.STAGES[gate.STAGES.index(stage)-1]
    evidence=dict(**{k:result[k] for k in gate.IDENTITY},stage=stage,status='PASS',scope='full_detector',
        test_evaluated=False,split='valid',package_manifest_sha256=result['package_manifest_sha256'],
        deployment_binding_sha256=gate.sha256(root/'DEPLOYMENT_BINDING.json'),
        dependencies={previous:gate.sha256(folder/(previous+'.json'))},
        artifacts={name:_artifact(path,folder) for name,path in artifacts.items()},
        metric_values=metric_values or {},**extra)
    evidence['metric_event_id']=gate.record_stage_metrics(root,evidence)
    gate.atomic_json(folder/(stage+'.json'),evidence)
    return gate.stage_evidence(root,network,stage,result)


def _publish_route(root,network,result,build,execution):
    if not (build/'ROUTE_TIMING_PASS').is_file():raise ValueError('route completion marker missing')
    values=parse_route((build/'timing.rpt').read_text(),(build/'utilization.rpt').read_text())
    artifacts={name:build/name for name in ('timing.rpt','utilization.rpt','utilization_hier.rpt','route_status.rpt',
              'ROUTE_TIMING_PASS','route.log','h2_graph.xsa','manifest.json')}
    if network!='D':
        artifacts.update(common_image_binding=build/'COMMON_IMAGE_BINDING.json',donor_route=root/'D/seed_42/route.json')
    elif (build/'route.log.process.json').is_file():
        artifacts['route.log.process.json']=build/'route.log.process.json'
    tool=re.search(r'\bTool\s+([^;\s]+);',(build/'ROUTE_TIMING_PASS').read_text())
    if not tool or tool[1]!='2022.2':raise ValueError('route tool version missing or differs from frozen 2022.2')
    provenance=dict(source=str(build/'timing.rpt'),sha256=gate.sha256(build/'timing.rpt'),
                    artifact_sha256=gate.sha256(build/'timing.rpt'),evidence_layer='place_and_route',
                    target_device='ZCU104',tool_version=tool[1],
                    checkpoint_sha256=result['checkpoint_sha256'],scope='routed full serial detector shell')
    if network!='D':
        provenance.update(common_image_from='D',donor_route_sha256=execution['donor_route_sha256'],
                          cost_interpretation='identical routed generic executor; candidate runtime measured separately')
    keys=['wns_ns','tns_ns','timing_met']+[f'{r}_{k}' for r in ('lut','ff','dsp','bram') for k in ('used','utilization_pct')]
    measured={key:dict(value=values[key],provenance=(dict(provenance,source=str(build/'utilization.rpt'),
        sha256=gate.sha256(build/'utilization.rpt'),artifact_sha256=gate.sha256(build/'utilization.rpt'))
        if key.startswith(('lut_','ff_','dsp_','bram_')) else provenance)) for key in keys}
    return _publish(root,network,'route',result,artifacts,
                    dict(**values,execution=execution,common_image_from='D' if network!='D' else None),measured)


def finalize_existing_route(root, network='D'):
    """Publish D's completed native route after a parser failure; never rerun."""
    if network!='D':raise ValueError('only D has native route artifacts; C/B1 use common-image admission')
    root=Path(root).resolve()
    admitted=gate.action_admission(root,network,'route')
    if not admitted['allowed']:return admitted
    binding=gate.load_binding(root)
    expected=next(r for r in binding['runs'] if r['network']==network and r['seed']==42)
    result=gate.load_result(root,expected)
    gate.stage_evidence(root,network,'hls',result)
    if (root/network/'seed_42/route.json').exists():
        gate.stage_evidence(root,network,'route',result)
        return dict(status='ALREADY_COMPLETE',network=network,stage='route')
    build=root/network/'seed_42/hardware/graph_build'
    execution=_completed_route_execution(build)
    return _publish_route(root,network,result,build,execution)


def execute_stage(root, network, stage):
    root=Path(root).resolve()
    if stage not in ('hls','route','board'):raise ValueError('hardware handler supports hls/route/board')
    admitted=gate.action_admission(root,network,stage)
    if not admitted['allowed']:return admitted
    binding=gate.load_binding(root)
    expected=next(r for r in binding['runs'] if r['network']==network and r['seed']==42)
    result=gate.load_result(root,expected)
    folder=root/network/'seed_42';output=folder/(stage+'.json')
    if output.exists():
        gate.stage_evidence(root,network,stage,result)
        return dict(status='ALREADY_COMPLETE',network=network,stage=stage)
    if stage=='board':
        from tools.sonar_deploy.board_result import verify_board_result
        return verify_board_result(root,network)
    if sys.platform!='linux' or not (HLS if stage=='hls' else VIVADO).is_file():
        raise RuntimeError('HARDWARE_TOOLCHAIN_NOT_READY')
    import fcntl
    lockfile=folder/('hardware_'+stage+'.lock')
    with lockfile.open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        build=folder/'hardware'/'graph_build'
        if stage=='hls':
            parity=gate.stage_evidence(root,network,'integer_parity',result)
            trace_path,traces=_trace_manifest(folder,parity)
            package_path=Path(result['package_manifest_path'])
            package=package_path.parent if package_path.is_absolute() else (root/package_path).parent
            graph,_=validate_package(package)
            if traces['graph_sha256']!=gate.sha256(package/'graph.json'):
                raise ValueError('integer trace graph mismatch')
            frame=traces['frames'][0];frame_folder=trace_path.parent/frame['folder']
            layers=gate.read_json(frame_folder/'layers.json')
            image_path=trace_path.parent/frame['input_path']
            values={'image':np.load(image_path,allow_pickle=False)}
            values.update({row['node']:np.load(frame_folder/row['chw_path'],allow_pickle=False) for row in layers})
            if build.exists():
                built=_verify_build(build)
                if built['graph_sha256']!=gate.sha256(package/'graph.json') or built['input_sha256']!=gate.sha256(image_path):
                    raise ValueError('Existing build belongs to a different graph/image')
                if (build/'h2_graph.cpp').read_bytes()!=(ROOT/'tools/detector_preflight/h2_graph.cpp').read_bytes():
                    raise ValueError('Core changed since build; explicit new hardware attempt required')
            else:
                build_graph(build,package,image_path,values=values)
            regression=folder/'hardware'/'numeric_regression'
            if regression.exists():
                regression_manifest=gate.read_json(regression/'regression_manifest.json')
                for name,digest in regression_manifest['files'].items():
                    if gate.sha256(regression/name)!=digest:raise ValueError('Regression input changed')
            else:
                build_regression(regression)
            execution=_run(HLS,['-f','run_graph.tcl'],build,'hls.log',10800)
            manifest=_verify_build(build)
            check_csim((build/'hls.log').read_text(errors='replace'),len(manifest['layers']))
            numeric_execution=_run(HLS,['-f','run.tcl'],regression,'hls.log',10800)
            log=(regression/'hls.log').read_text(errors='replace')
            if 'REAL_PTQ_NUMERIC_REGRESSION_PASS' not in log or 'C/RTL co-simulation finished: PASS' not in log or 'ERROR:' in log:
                raise ValueError('representative numeric RTL regression failed')
            for name in ('h2_graph.cpp','numeric_v1.h'):
                if (build/name).read_bytes()!=(regression/name).read_bytes():raise ValueError('C/RTL source mismatch')
            rtl=compare_rtl(build/'hls_project/solution1/syn/verilog',regression/'hls_project/solution1/syn/verilog')
            gate.atomic_json(build/'rtl_equivalence.json',rtl)
            artifacts={'build_manifest':build/'manifest.json','csim_log':build/'hls.log',
                'cosim_log':regression/'hls.log','regression_manifest':regression/'regression_manifest.json',
                'rtl_equivalence':build/'rtl_equivalence.json','core_source':build/'h2_graph.cpp',
                'numeric_source':build/'numeric_v1.h','graph':build/'graph.json'}
            return _publish(root,network,stage,result,artifacts,
                dict(csim='PASS',cosim='PASS',mismatch_count=0,nodes=len(graph['nodes']),
                     csim_scope='all actual graph nodes first frozen train image',
                     cosim_scope='representative arithmetic regression with exact synthesized RTL match',
                     execution=execution,numeric_execution=numeric_execution))
        manifest=_verify_build(build)
        check_csim((build/'hls.log').read_text(errors='replace'),len(manifest['layers']))
        if network=='D':
            execution=_run(VIVADO,['-mode','batch','-source','build_graph_bd.tcl'],build,'route.log',14400)
        else:
            execution=reuse_common_route(root,network,result,build)
        return _publish_route(root,network,result,build,execution)
