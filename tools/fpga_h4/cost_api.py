"""Exact-lookup cost API for (network_graph, quant_config, mapping).

No nearest neighbour, interpolation, regression, or other substitute cost.
Unmeasured (x, q, m) returns NEEDS_MEASUREMENT with primary metrics null.
"""
import json
from pathlib import Path
from accuracy_api import lookup_accuracy
from q_admission import annotate_final_eligibility
from candidate_graphs import architectures, candidate_graph, identify
from quant_contracts import PROFILES, canonical_profile, node_precision, operator_contract, quant_contract, quantization_regions
from common import *
from search import (
    admit_aux_rows, admit_rows, bind_quantized_accuracy, evaluate_assignment, load_h3,
    named_assignment, packing_cost, validate_h3_row,
)

NAMED_MAPPINGS={
    'low':'low','reference_low':'low','reference_low_parallelism':'low',
    'high':'high','reference_high':'high','reference_high_parallelism':'high',
    'min_latency':'min_latency','reference_min_latency':'min_latency',
    'min_dsp':'min_dsp','reference_min_dsp':'min_dsp',
    'min_lut':'min_lut','reference_min_lut':'min_lut',
    'min_bram':'min_bram','reference_min_bram':'min_bram',
}
V1_BITS=dict(input_bitwidth=8,weight_bitwidth=8,accumulator_bitwidth=32,output_bitwidth=8)
PLATFORM=dict(part=PART,tool_version='2022.2',target_clock_ns=5)
LABEL_BY_ARCH={v:k for k,v in ARCH.items() if k!='baseline'}
LABEL_BY_ARCH[ARCH['baseline']]='D'


def architecture_id_from_graph(graph):
    arch=graph.get('architecture') or {}
    oc=arch.get('output_combination')
    kernels=arch.get('kernels') or {}
    if not oc:return None
    parts=[]
    for stage in (1,2,4,5,6,7):
        kernel=kernels.get(f'stage{stage}')
        if kernel is None:return None
        parts.append(f's{stage}k{int(kernel)}')
    return oc+'_'+'_'.join(parts)


def node_description(node,graph):
    op=node.get('op')
    if op=='upsample_add':
        shape=graph['edges'][node['inputs'][0]]['shape']
        cin,h,w=int(shape[0]),int(shape[1]),int(shape[2])
        return dict(operator_type='upsample_add',Cin=cin,Cout=cin,H=h,W=w,kernel=1,stride=1,groups=1,activation='aligned_add',expanded_channels=0)
    if op=='conv':
        groups=int(node.get('groups',1))
        operator='DW' if groups>1 else 'Conv'
        return dict(operator_type=operator,Cin=int(node['Cin']),Cout=int(node['Cout']),H=int(node['H']),W=int(node['W']),
                    kernel=int(node['kernel']),stride=int(node['stride']),groups=groups,activation=node['activation'],expanded_channels=0)
    return None


def atomic_usage(graph):
    usage=[];unknown=[]
    for node in graph['nodes']:
        desc=node_description(node,graph)
        if desc is None:
            unknown.append(dict(node=node.get('id'),operator_type=node.get('op'),status='NEEDS_MEASUREMENT',reason='unsupported_or_unknown_operator'))
            continue
        usage.append(dict(node=node['id'],shape_id=shape_key(desc),**desc))
    return usage,unknown


CONTRACT_REASONS={'graph_contract_mismatch','graph_quant_contract_mismatch','graph_interface_contract_mismatch'}

def abstain(reason,missing=None,status=None,**extra):
    if status is None:
        status='INVALID' if reason in {'unsupported_mapping_type','malformed_candidate'} else (
            'CONTRACT_MISMATCH' if reason in CONTRACT_REASONS else 'NEEDS_MEASUREMENT')
    row=dict(status=status,hardware_status=status,reason=reason,missing=missing or [],
             Accuracy=None,Latency=None,DSP=None,LUT=None,BRAM=None,
             compute_latency_us=None,DSP_peak=None,LUT_peak=None,BRAM18_peak=None,
             packing_latency_us=None,shell_reservation='UNKNOWN',deployment='NOT_CLAIMED',
             measurement_valid=None,timing_eligible=None,eligible=False,
             accuracy_status=None,quantized_accuracy_bound=False,
             latency_scope='HLS_CSYNTH_serial_operator_sum_not_board_e2e',
             resource_scope='kernel-only peak_operator; sum=static instantiation of operators, not whole-network post-route resource; exclude shell/adapters/DDR',
             constraint_status='REPORTED_NOT_BUDGETED',measurement_scope='OP_LEVEL_HLS',
             latency=dict(serial_op_estimate_us=None,serial_op_estimate_cycles=None,board_measured_ms=None),
             resource=dict(peak_operator=None,integrated_post_synth=None,board_routed=None),
             latency_metric='serial_op_estimate_us',resource_metric='peak_operator')
    row.update(extra)
    row['Latency']=None
    row['DSP']=None
    row['LUT']=None
    row['BRAM']=None
    row['eligible']=False
    if row.get('hardware_status') is None:
        row['hardware_status']=status
    return row


class CostDatabase:
    def __init__(self):
        man,hashes,rows,info=load_h3()
        self.manifest=man
        self.hashes=hashes
        self.rows=rows
        self.info=info
        self.options=admit_rows(rows,hashes)
        self.pack_rows=admit_aux_rows(rows,hashes)
        self.agent_queries=0
        self._arch_graphs={}
        self.extension_rows=[]
        extension=ROOT/'artifacts/sonar_joint_cost_v1/clean.json'
        if extension.is_file():
            for row in json.loads(extension.read_text()):
                bits={k:row[k] for k in V1_BITS}
                expected=operator_contract(bits['weight_bitwidth'],bits['input_bitwidth'],self.hashes)
                checked,errors=validate_h3_row(row,expected,bits)
                if errors:
                    raise ValueError(f"Extension measurement invalid: {row.get('point_id')}: {errors}")
                if checked['timing_eligible']:
                    self.extension_rows.append(checked)
        self.graphs={}
        self.graph_hashes={}
        self.label_by_graph_hash={}
        for label,rel in info['packages'].items():
            graph=json.loads((ROOT/rel/'graph.json').read_text(encoding='utf-8'))
            self.graphs[label]=graph
            gh=canonical_graph_hash(graph)
            self.graph_hashes[label]=gh
            self.label_by_graph_hash.setdefault(gh,label)

    def _graph_contract_error(self,graph):
        for key,val in self.hashes.items():
            if graph.get(key) not in (None,val):
                return abstain('graph_contract_mismatch',graph_contract_key=key,
                               graph_contract_hash=graph.get(key),expected_contract_hash=val)
        return None

    def resolve_graph(self,network_graph):
        if isinstance(network_graph,dict) and 'nodes' in network_graph:
            err=self._graph_contract_error(network_graph)
            if err:return None,None,None,None,err
            gh=canonical_graph_hash(network_graph)
            aid=architecture_id_from_graph(network_graph)
            label=self.label_by_graph_hash.get(gh)
            if label is None:
                admitted=identify(network_graph,self.hashes)
                if admitted:
                    return network_graph,admitted,admitted,gh,None
                return None,None,aid,gh,abstain('unregistered_or_modified_graph',
                    network_graph_canonical_hash=gh,architecture_id=aid)
            return network_graph,label,aid,gh,None
        if isinstance(network_graph,(str,Path)):
            token=str(network_graph)
            if token in self.graphs:
                graph=self.graphs[token]
                return graph,token,ARCH[token],self.graph_hashes[token],None
            if token in architectures():
                graph=self._arch_graphs.get(token)
                if graph is None:
                    graph=candidate_graph(token,self.hashes)
                    self._arch_graphs[token]=graph
                return graph,token,token,canonical_graph_hash(graph),None
            path=Path(token)
            if not path.is_absolute():
                path=ROOT/path
            if path.is_file():
                graph=json.loads(path.read_text(encoding='utf-8'))
                err=self._graph_contract_error(graph)
                if err:return None,None,None,None,err
                gh=canonical_graph_hash(graph)
                aid=architecture_id_from_graph(graph)
                label=self.label_by_graph_hash.get(gh)
                if label is None:
                    admitted=identify(graph,self.hashes)
                    if admitted:
                        return graph,admitted,admitted,gh,None
                    return None,None,aid,gh,abstain('unregistered_or_modified_graph',
                        network_graph_canonical_hash=gh,architecture_id=aid,network_graph=token)
                return graph,label,aid,gh,None
            return None,None,None,None,abstain('unknown_network_graph',network_graph=token)
        return None,None,None,None,abstain('unknown_network_graph')

    def _quant_record(self,name,profile):
        spec=quant_contract(profile,self.hashes)
        v1=self.hashes['numeric_contract_hash']
        return dict(status='PROFILE_RESOLVED',name=name,profile=profile,
            numeric_contract_hash=v1,interface_contract_hash=self.hashes['interface_contract_hash'],
            quant_contract_hash=spec['quant_contract_hash'],quantization_config=quantization_regions(profile),
            **V1_BITS)

    def resolve_quant(self,quant_config):
        v1=self.hashes['numeric_contract_hash']
        if isinstance(quant_config,str):
            profile=canonical_profile(quant_config)
            if profile:
                name='NUMERIC_CONTRACT_V1' if quant_config in (None,'NUMERIC_CONTRACT_V1','numeric_contract_v1',v1) else profile
                if quant_config in PROFILES or quant_config in ('W8A16','W16A16','HEAD_W8A16','HEAD_W16A16','W8A8'):
                    name=profile
                return self._quant_record(name,profile),None
        if quant_config in (None,v1):
            return self._quant_record('NUMERIC_CONTRACT_V1','W8A8'),None
        if isinstance(quant_config,dict):
            if hashobj(quant_config)==v1:
                return self._quant_record('NUMERIC_CONTRACT_V1','W8A8'),None
            bits={}
            alias=dict(activation_bits='input_bitwidth',weight_bits='weight_bitwidth',accumulator_bits='accumulator_bitwidth',output_bits='output_bitwidth')
            for src,dst in alias.items():
                if src in quant_config:bits[dst]=int(quant_config[src])
            for key in V1_BITS:
                if key in quant_config:bits[key]=int(quant_config[key])
            extra=dict(quant_config=quant_config,quant_bits=bits or None)
            return None,abstain('unmeasured_quant_config',**extra)
        return None,abstain('unmeasured_quant_config',quant_config=quant_config)

    def _same_value(self,row,key,val):
        if key in {'PE','SIMD','input_tile','output_tile','input_bitwidth','weight_bitwidth','accumulator_bitwidth','output_bitwidth','target_clock_ns'}:
            try:
                return float(row.get(key))==float(val)
            except (TypeError,ValueError):
                return False
        return str(row.get(key))==str(val)

    def _row_spec(self,row):
        return {key:row[key] for key in EXACT_MAPPING_FIELDS}

    def _constraint_errors(self,row,spec,require_complete=True):
        unknown=sorted(set(spec)-set(EXACT_MAPPING_FIELDS)-{'name','policy','mapping_policy'})
        if unknown:
            return [dict(point_id=row.get('point_id'),status='NEEDS_MEASUREMENT',
                         reason='unsupported_mapping_constraint',unsupported_fields='|'.join(unknown),requested=spec)]
        fields=[key for key in EXACT_MAPPING_FIELDS if key in spec]
        missing=[key for key in EXACT_MAPPING_FIELDS if key not in spec]
        if require_complete and missing:
            return [dict(point_id=row.get('point_id'),status='NEEDS_MEASUREMENT',
                         reason='mapping_fields_incomplete',missing_fields='|'.join(missing),requested=spec)]
        mismatches=[key for key in fields if not self._same_value(row,key,spec[key])]
        if mismatches:
            return [dict(point_id=row.get('point_id'),status='NEEDS_MEASUREMENT',
                         reason='requested_mapping_constraint_mismatch',
                         mismatched_fields='|'.join(mismatches),requested=spec)]
        bits={k:row[k] for k in V1_BITS}
        expected=operator_contract(bits['weight_bitwidth'],bits['input_bitwidth'],self.hashes)
        checked,errors=validate_h3_row(row,expected,bits)
        if errors or not checked.get('timing_eligible'):
            return [dict(point_id=row.get('point_id'),status='NEEDS_MEASUREMENT',
                         reason='measurement_not_eligible',
                         measurement_valid=checked.get('measurement_valid'),
                         timing_eligible=checked.get('timing_eligible'),
                         admission_errors=checked.get('admission_errors'))]
        return []

    def _explicit_hit(self,shape_id,spec,options=None):
        pool=(self.options if options is None else options).get(shape_id,[])
        if not pool:
            return None,dict(shape_id=shape_id,status='NEEDS_MEASUREMENT',reason='no_valid_csynth_row')
        missing=[key for key in EXACT_MAPPING_FIELDS if key not in spec]
        if missing:
            return None,dict(shape_id=shape_id,status='NEEDS_MEASUREMENT',
                             reason='mapping_fields_incomplete',
                             missing_fields='|'.join(missing),requested=spec)
        hits=[r for r in pool if all(self._same_value(r,key,spec[key]) for key in EXACT_MAPPING_FIELDS)]
        if len(hits)==1:
            errors=self._constraint_errors(hits[0],spec)
            return (None,errors[0]) if errors else (hits[0],None)
        if not hits:
            return None,dict(shape_id=shape_id,requested_PE=spec.get('PE'),requested_SIMD=spec.get('SIMD'),requested_buffer_profile=spec.get('buffer_profile'),
                             status='NEEDS_MEASUREMENT',reason='requested_mapping_absent')
        return None,dict(shape_id=shape_id,status='NEEDS_MEASUREMENT',reason='mapping_not_unique',n_hits=len(hits))

    def _apply_constraints(self,assignment,usage,mapping):
        top={k:mapping[k] for k in EXACT_MAPPING_FIELDS if k in mapping}
        by_node=dict(mapping.get('by_node') or mapping.get('nodes') or {})
        by_shape=dict(mapping.get('by_shape') or mapping.get('shapes') or {})
        missing=[]
        for u in usage:
            row=assignment.get(u['node'])
            if row is None:
                continue
            specs=[]
            if top:specs.append(top)
            if u['shape_id'] in by_shape:specs.append(by_shape[u['shape_id']])
            if u['node'] in by_node:specs.append(by_node[u['node']])
            for spec in specs:
                errors=self._constraint_errors(row,spec,require_complete=False)
                for err in errors:
                    err=dict(err,node=u['node'],shape_id=u['shape_id'])
                    missing.append(err)
        return missing

    def resolve_assignment(self,mapping,usage,options=None):
        options=self.options if options is None else options
        if mapping is None:
            return None,None,[dict(status='NEEDS_MEASUREMENT',reason='mapping_unspecified')]
        if isinstance(mapping,str):
            mode=NAMED_MAPPINGS.get(mapping)
            if mode is None:
                return None,mapping,[dict(status='NEEDS_MEASUREMENT',reason='unknown_named_mapping',mapping=mapping)]
            assignment,missing=named_assignment(usage,options,mode)
            return assignment,mapping,missing
        if not isinstance(mapping,dict):
            return None,None,[dict(status='INVALID',reason='unsupported_mapping_type')]
        policy=mapping.get('policy') or mapping.get('mapping_policy')
        if policy in NAMED_MAPPINGS:
            assignment,missing=named_assignment(usage,options,NAMED_MAPPINGS[policy])
            if missing:
                return assignment,policy,missing
            constraint_errors=self._apply_constraints(assignment,usage,mapping)
            if constraint_errors:
                return assignment,policy,constraint_errors
            return assignment,policy,missing
        by_node=dict(mapping.get('by_node') or mapping.get('nodes') or {})
        by_shape=dict(mapping.get('by_shape') or mapping.get('shapes') or {})
        uniform={k:mapping[k] for k in EXACT_MAPPING_FIELDS if k in mapping}
        assignment={};missing=[]
        for u in usage:
            spec=by_node.get(u['node']) or by_shape.get(u['shape_id']) or (uniform if uniform else None)
            if spec is None:
                missing.append(dict(node=u['node'],shape_id=u['shape_id'],status='NEEDS_MEASUREMENT',reason='node_mapping_unspecified'))
                continue
            hit,err=self._explicit_hit(u['shape_id'],spec,options)
            if err:
                err=dict(err,node=u['node']);missing.append(err)
            else:
                assignment[u['node']]=hit
        return assignment,mapping.get('name') or 'explicit',missing

    def _architecture_hash(self,aid):
        spec=architectures().get(aid) if aid else None
        return hashobj(spec) if spec else None

    def _stamp(self,row,aid,graph,quant,mapping,canonical,bind_accuracy):
        arch_hash=self._architecture_hash(aid)
        graph_hash=canonical or (canonical_graph_hash(graph) if graph else row.get('network_graph_canonical_hash'))
        row.update(architecture_id=aid or row.get('architecture_id'),architecture_hash=arch_hash,
            graph_hash=graph_hash,canonical_graph_hash=graph_hash,
            mapping_config=mapping if isinstance(mapping,(str,dict)) else None,
            mapping_hash=hashobj(mapping if isinstance(mapping,(str,dict)) else {'mapping':None}),
            target_device=PART,target_fpga=PART,clock_constraint_ns=PLATFORM['target_clock_ns'],
            toolchain_version=PLATFORM['tool_version'])
        if quant:
            row.update(quant_config=quant['name'],profile=quant['profile'],
                quantization_config=quant['quantization_config'],quant_contract_hash=quant['quant_contract_hash'],
                quant_numeric_contract_hash=quant['numeric_contract_hash'],
                quant_interface_contract_hash=quant['interface_contract_hash'])
        if bind_accuracy and quant and aid and row.get('status')!='CONTRACT_MISMATCH':
            acc=lookup_accuracy(aid,quant['profile'],self.hashes,quant_contract_hash=quant['quant_contract_hash'],architecture_hash=arch_hash)
            bind_quantized_accuracy(row,acc)
            row['accuracy_key']=dict(architecture_id=aid,architecture_hash=arch_hash,
                quant_contract_hash=quant['quant_contract_hash'],eval_contract_hash=acc.get('eval_contract_hash'))
        else:
            row['accuracy_status']='CONTRACT_MISMATCH' if row.get('status')=='CONTRACT_MISMATCH' else (row.get('accuracy_status') or 'NEEDS_ACCURACY')
            row['Accuracy']=None
            row['mean_mAP50_95']=None
            row['std_mAP50_95']=None
            row['quantized_accuracy_bound']=False
        measured=row.get('hardware_status')=='MEASURED' and row.get('status')=='LOOKED_UP'
        if measured:
            row['Latency']=row['latency']['serial_op_estimate_us']
            peak=row['resource']['peak_operator']
            row['DSP']=peak['dsp'];row['LUT']=peak['lut'];row['BRAM']=peak['bram']
        else:
            row['Latency']=None;row['DSP']=None;row['LUT']=None;row['BRAM']=None
        row['deployment']='NOT_CLAIMED'
        row['shell_reservation']='UNKNOWN'
        annotate_final_eligibility(row)
        return row

    def estimate_cost(self,network_graph,quant_config,mapping):
        """Exact HLS lookup for one architecture + quantization + mapping candidate."""
        self.agent_queries+=1
        graph,label,aid,canonical,err=self.resolve_graph(network_graph)
        if err:
            return self._stamp(err,aid, None, None, mapping, canonical, bind_accuracy=False)
        quant,qerr=self.resolve_quant(quant_config)
        if qerr:
            return self._stamp(qerr,aid,graph,None,mapping,canonical,bind_accuracy=False)
        body_hash=graph.get('numeric_contract_hash')
        if body_hash and body_hash!=quant['numeric_contract_hash']:
            row=abstain('graph_quant_contract_mismatch',graph_numeric_contract_hash=body_hash,quant_numeric_contract_hash=quant['numeric_contract_hash'])
            return self._stamp(row,aid,graph,quant,mapping,canonical,bind_accuracy=False)
        iface_hash=graph.get('interface_contract_hash')
        if iface_hash and iface_hash!=quant['interface_contract_hash']:
            row=abstain('graph_interface_contract_mismatch',graph_interface_contract_hash=iface_hash,quant_interface_contract_hash=quant['interface_contract_hash'])
            return self._stamp(row,aid,graph,quant,mapping,canonical,bind_accuracy=False)
        usage,unknown=atomic_usage(graph)
        if unknown:
            row=abstain('unsupported_or_unknown_operator',missing=unknown,network_label=label,architecture_id=aid)
            return self._stamp(row,aid,graph,quant,mapping,canonical,bind_accuracy=False)
        # Validate against the entire graph before resolving one node at a time.
        if isinstance(mapping,dict):
            allowed=set(EXACT_MAPPING_FIELDS)|{'name','policy','mapping_policy','by_node','nodes','by_shape','shapes'}
            errors=[]
            if set(mapping)-allowed:
                errors.append('unsupported_mapping_constraint')
            for primary,alias,identities in (
                ('by_node','nodes',{u['node'] for u in usage}),
                ('by_shape','shapes',{u['shape_id'] for u in usage})):
                if primary in mapping and alias in mapping:
                    errors.append('ambiguous_mapping_alias')
                specs=mapping.get(primary,mapping.get(alias,{}))
                if not isinstance(specs,dict) or set(specs)-identities:
                    errors.append('unknown_mapping_target')
                elif any(not isinstance(spec,dict) or not spec or set(spec)-set(EXACT_MAPPING_FIELDS) for spec in specs.values()):
                    errors.append('unsupported_mapping_constraint')
            if 'policy' in mapping and 'mapping_policy' in mapping and mapping['policy']!=mapping['mapping_policy']:
                errors.append('ambiguous_mapping_policy')
            policy=mapping.get('policy',mapping.get('mapping_policy'))
            if policy is not None and policy not in NAMED_MAPPINGS:
                errors.append('unknown_named_mapping')
            if errors:
                row=abstain(errors[0],missing=[dict(reason=e) for e in errors])
                return self._stamp(row,aid,graph,quant,mapping,canonical,bind_accuracy=True)
        profile=quant['profile']
        assignment={};missing=[];policy=None
        for u in usage:
            bits=node_precision(profile,u['node'])
            pool=[r for r in self.options.get(u['shape_id'],[]) if all(r[k]==v for k,v in bits.items())]
            pool += [r for r in self.extension_rows if r['shape_id']==u['shape_id'] and all(r[k]==v for k,v in bits.items())]
            selected,policy,absent=self.resolve_assignment(mapping,[u],{u['shape_id']:pool})
            assignment.update(selected or {});missing.extend(absent)
        if missing:
            status='INVALID' if any(item.get('status')=='INVALID' for item in missing) else 'NEEDS_MEASUREMENT'
            row=abstain('requested_mapping_absent' if assignment else missing[0].get('reason','requested_mapping_absent'),
                status=status,missing=missing,network_label=label,architecture_id=aid,mapping_policy=policy)
            return self._stamp(row,aid,graph,quant,mapping,canonical,bind_accuracy=True)
        if isinstance(mapping,dict):
            errors=self._apply_constraints(assignment,usage,mapping)
            if errors:
                return self._stamp(abstain('requested_mapping_constraint_mismatch',missing=errors),aid,graph,quant,mapping,canonical,bind_accuracy=True)
        acc=lookup_accuracy(aid,profile,self.hashes,quant_contract_hash=quant['quant_contract_hash'],architecture_hash=self._architecture_hash(aid))
        acc['architecture_id']=aid or acc.get('architecture_id')
        pack_rows,pack_missing=packing_cost(graph,self.pack_rows)
        row=evaluate_assignment(label or aid or 'unregistered',usage,graph,assignment,acc,pack_rows,pack_missing,policy or 'explicit')
        row.pop('_details',None);row.pop('_families',None)
        if profile!='W8A8':
            # INT8 DMARepack rows are not a measurement of the head activation width.
            row.update(packing_latency_us=None,combined_uncalibrated_us=None,
                mixed_boundary_status='HEAD_BITWIDTH_COMPUTE_MEASURED_PACKING_NOT_REMEASURED')
        return self._stamp(row,aid,graph,quant,mapping,canonical,bind_accuracy=True)


_DB=None

def get_database():
    global _DB
    if _DB is None:
        _DB=CostDatabase()
    return _DB

def estimate_cost(network_graph,quant_config,mapping):
    """Look up HLS CSYNTH cost for an exact measured (graph, quant, mapping) triple."""
    return get_database().estimate_cost(network_graph,quant_config,mapping)


if __name__=='__main__':
    result=estimate_cost('B1','NUMERIC_CONTRACT_V1','reference_min_latency')
    print(result['status'],result.get('compute_latency_us'),result.get('DSP'),result.get('deployment'))
