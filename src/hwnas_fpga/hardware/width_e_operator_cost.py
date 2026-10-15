"""Exact width/e costs with mapping-bound whole-network storage feedback.

Uncalibrated operator sums remain available to the hardware preflight script.
The formal runtime requires a separately validated calibration and a routed seed.
"""
from dataclasses import replace
import math
from pathlib import Path

from .cost import FPGACostEstimator
from .storage_calibration import predict_coupled_storage
from .mobilenetA_operator_cost import (
    ZCU104MobileNetAOperatorCostEstimator, canonical_sha256, load_hashed_json,
)

VERSION = 'zcu104_width_e_sonar_serial_gap_v1'
SUPPORTED_VERSIONS = {VERSION, 'zcu104_width_e_sonar_serial_gap_v2'}
MAPPING = 'independent_blocks_pi4_po4_u1_int8_serial_gap_sync_distributed_fifo_d64_200mhz'
RESOURCE_FIELDS = ('lut', 'lut_as_logic', 'lut_as_memory', 'ff', 'bram36', 'dsp')
COUPLED_STORAGE_MODEL = 'coupled_bram_lutram_v1'
DEVICE_LIMITS = (('bram36', 312), ('lut_as_memory', 101760), ('ff', 460800))
# Device site counts. The BRAM18 count is also the physical point where Vivado
# spills storage into LUTRAM, so it is not a configurable search budget.
LUTRAM_SITES = 101760
BRAM18_SITES = 624

def operator_key(op, cin, cout, resolution, stride=1, kernel=1, expansion=1, relu=None):
    return (op, int(cin), int(cout), int(resolution), int(stride), int(kernel), int(expansion), relu)

def architecture_operator_keys(architecture, search_space):
    keys=[operator_key('stem',1,32,224,2,3,1,1)]
    for block in search_space.resolve_blocks(architecture):
        if block.op == 'skip':
            if block.in_channels != block.out_channels or block.stride != 1:
                raise ValueError('Only structural same-channel skip is supported')
            continue
        kind='pw_conv' if block.op=='conv' and block.kernel_size==1 else block.op
        keys.append(operator_key(kind,block.in_channels,block.out_channels,block.input_resolution,
                                 block.stride,block.kernel_size,block.expand_ratio,0 if kind=='pw_conv' else None))
    keys.extend([operator_key('pw_conv',architecture.stages[-1].channels,1280,7,relu=1),
                 operator_key('global_avg_pool',1280,1280,7),operator_key('fc_layer',1280,8,1)])
    return keys

class ZCU104WidthEOperatorCostEstimator(ZCU104MobileNetAOperatorCostEstimator):
    cost_source = 'zcu104_width_e_operator_cost_v1'

    def __init__(self, *args, operator_cost_path, calibration_path, diagnostic_only=False, **kwargs):
        FPGACostEstimator.__init__(self, *args, **kwargs)
        if self.hardware_spec.clock_mhz != 200 or 'ZCU104' not in self.hardware_spec.name or self.quantization_bits != 8:
            raise ValueError('Width/e costs require ZCU104 INT8 at 200 MHz')
        if self.lut_query_engine is not None or any(v != 1 for v in self.analytic_calibration.values()):
            raise ValueError('Width/e costs forbid legacy LUT or analytic calibration fallback')
        self.operator_cost_path, self.table = load_hashed_json(operator_cost_path)
        self.calibration_path, self.calibration = load_hashed_json(calibration_path)
        for payload in [self.table, self.calibration]:
            if payload.get('version') not in SUPPORTED_VERSIONS or payload.get('mapping_id') != MAPPING:
                raise ValueError('Width/e evidence version or FPGA mapping mismatch')
        if self.table['version'] != self.calibration['version']:
            raise ValueError('Calibration is bound to another search-space version')
        self.diagnostic_only = diagnostic_only
        self.coupled_storage = self.calibration.get('model_kind') == COUPLED_STORAGE_MODEL
        self.search_ready = (self.table.get('coverage_complete') is True
                             and all(self.calibration.get(k) is True for k in ['validated','routed_seed_accepted','heldout_architecture_check_passed']))
        if self.calibration.get('operator_cost_payload_sha256') != self.table['payload_sha256']:
            raise ValueError('Calibration is not bound to this exact operator table')
        if not diagnostic_only:
            self.require_search_ready()
        self.rows_by_key = {}
        for row in self.table['rows']:
            key = tuple(row['key'])
            if key in self.rows_by_key:
                raise ValueError(f'Duplicate exact operator key: {key}')
            values = row['resources']
            for field in RESOURCE_FIELDS:
                if field not in values or not math.isfinite(float(values[field])) or values[field] < 0:
                    raise ValueError(f'Missing/invalid resource {field}: {key}')
            if values['lut'] != values['lut_as_logic'] + values['lut_as_memory']:
                raise ValueError(f'LUT accounting mismatch: {key}')
            if row['latency_cycles'] <= 0 or not math.isfinite(row['power_w']) or row['power_w'] < 0:
                raise ValueError(f'Invalid latency/power: {key}')
            # The coupled model predicts storage from pre-fallback HLS demand,
            # which older tables do not carry. Reject at load, not mid-search.
            if self.coupled_storage:
                hls_bram18 = row.get('bram18_hls')
                if hls_bram18 is None or not math.isfinite(float(hls_bram18)) or float(hls_bram18) < 0:
                    raise ValueError(f'Coupled storage calibration requires a valid bram18_hls: {key}')
            self.rows_by_key[key] = row
        self.last_selected_row_keys = []
        self._operator_cost_queries = 0
        self._operator_cost_hits = 0
        self._operator_cost_misses = 0

    def require_search_ready(self):
        reasons=[]
        if self.table.get('coverage_complete') is not True: reasons.append('exact_shape_coverage_incomplete')
        if self.calibration.get('validated') is not True: reasons.append('whole_network_storage_calibration_unvalidated')
        if self.calibration.get('routed_seed_accepted') is not True: reasons.append('no_resource_and_timing_accepted_seed')
        if self.calibration.get('heldout_architecture_check_passed') is not True: reasons.append('heldout_architecture_check_missing')
        if reasons: raise ValueError('Width/e NAS admission blocked: '+', '.join(reasons))

    def _selected_rows(self, architecture, search_space):
        if search_space.validate(architecture): raise ValueError('Architecture is outside the configured width/e search space')
        for field in ['stage_channel_choices','stage_strides','stage_depth_choices']:
            if canonical_sha256(getattr(search_space.config,field)) != canonical_sha256(self.table['search_space'][field]):
                raise ValueError('Hardware table does not cover this search-space contract: '+field)
        blocks=[[b.to_dict() for b in group] for group in search_space.config.stage_block_choices]
        if canonical_sha256(blocks) != canonical_sha256(self.table['search_space']['stage_block_choices']):
            raise ValueError('Hardware table is bound to different stage operator choices')
        if (search_space.config.image_size, architecture.input_channels, architecture.stem_channels,
            architecture.stem_stride, architecture.post_stem_downsample_stride, architecture.head_conv_channels,
            architecture.head_channels, architecture.num_classes, len(architecture.stages)) != (224,1,32,2,1,1280,None,8,8):
            raise ValueError('Width/e estimator requires the frozen grayscale 224 / eight-stage / GAP1280 macro architecture')
        keys=architecture_operator_keys(architecture,search_space)
        self._operator_cost_queries+=len(keys)
        missing=[key for key in keys if key not in self.rows_by_key]
        self._operator_cost_hits+=len(keys)-len(missing)
        self._operator_cost_misses+=len(missing)
        if missing: raise ValueError(f'Unknown exact hardware costs (no extrapolation): {missing}')
        self.last_selected_row_keys=[self.rows_by_key[key]['case_name'] for key in keys]
        return [self.rows_by_key[key] for key in keys]

    def estimate(self, architecture, search_space, *, apply_calibration=True):
        rows=self._selected_rows(architecture, search_space)
        analytic=FPGACostEstimator.estimate(self,architecture,search_space)
        resolved_blocks=search_space.resolve_blocks(architecture)
        resources={key:sum(row['resources'][key] for row in rows) for key in RESOURCE_FIELDS}
        cycles=sum(row['latency_cycles'] for row in rows)
        latency=cycles/200000
        power=max(row['power_w'] for row in rows)
        energy=sum(row['power_w']*row['latency_cycles']/200000 for row in rows)
        # Coupled storage calibration is bound to this mapping and exact table.
        # No coefficients are inferred from the old 2.01 * total-LUT heuristic.
        calibrated=apply_calibration and self.calibration.get('validated') is True
        requested_bram18=None
        storage_domain_unknown=False
        if calibrated:
            if self.coupled_storage:
                resources,storage=predict_coupled_storage(resources,sum(row['bram18_hls'] for row in rows),self.calibration)
                requested_bram18=storage['requested_bram18']
                storage_domain_unknown=storage['domain_unknown']
            else:
                corrections=self.calibration['resource_upper_residuals']
                for key in RESOURCE_FIELDS:
                    value=float(corrections[key])
                    if not math.isfinite(value) or value<0: raise ValueError('Invalid calibrated resource residual')
                    resources[key]+=value
            resources['lut']=resources['lut_as_logic']+resources['lut_as_memory']
        total_dsp=math.ceil(resources['dsp']);total_bram=math.ceil(resources['bram36']);total_lut=math.ceil(resources['lut'])
        physical=self._physical_constraints()
        lutram_budget=self._optional_float(physical.get('board_max_lutram'))
        if lutram_budget is None: lutram_budget=float(LUTRAM_SITES)
        bram18_budget=self._optional_float(physical.get('board_max_bram18'))
        if bram18_budget is None: bram18_budget=float(BRAM18_SITES)
        # Analytic structural pressures stay in place so configured physical
        # axes and gates keep working; exact storage keeps its own metrics.
        metrics=dict(analytic.physical_metrics)
        metrics.update({'operator_sum_lut_as_logic':sum(r['resources']['lut_as_logic'] for r in rows),
                 'operator_sum_lut_as_memory':sum(r['resources']['lut_as_memory'] for r in rows),
                 'operator_sum_bram36':sum(r['resources']['bram36'] for r in rows),
                 'whole_network_storage_calibration_validated':float(calibrated),
                 'full_network_route_established':0.0,'full_network_power_established':0.0,
                 'zcu104_predicted_lut_as_memory':resources['lut_as_memory'],
                 'zcu104_predicted_lut_as_memory_pressure':resources['lut_as_memory']/lutram_budget,
                 'zcu104_predicted_bram18':resources['bram36']*2,
                 'zcu104_predicted_bram18_pressure':resources['bram36']*2/bram18_budget})
        if requested_bram18 is not None:
            metrics['zcu104_requested_bram18_before_fallback']=requested_bram18
            metrics['zcu104_predicted_bram18_overflow']=max(0,requested_bram18-BRAM18_SITES)
        violations=list(self._check_constraints(latency_ms=latency,energy_mj=energy,power_w=power,
            model_size_mb=analytic.model_size_mb,resource_dsp=total_dsp,
            resource_bram=total_bram,resource_lut=total_lut,
            memory_bandwidth_gbps=analytic.memory_bandwidth_gbps,offchip_mem_mb=analytic.offchip_mem_mb))
        violations.extend(self._check_physical_constraints(resource_dsp=total_dsp,resource_bram=total_bram,
            resource_lut=total_lut,physical_metrics=metrics,resolved_blocks=resolved_blocks))
        violations.extend(self._check_operator_policy(resolved_blocks))
        for key,limit in DEVICE_LIMITS:
            if resources[key]>limit: violations.append(f'{key}>{limit}')
        for row in rows:
            if row.get('timing_met') is not True: violations.append('operator_timing_not_met:'+row['case_name'])
        if storage_domain_unknown: violations.append('storage_overflow_extrapolation_unknown')
        return replace(analytic,total_lut=total_lut,total_bram=total_bram,total_dsp=total_dsp,
            latency_cycles=cycles,latency_ms=latency,power_w=power,
            peak_lut=math.ceil(max(r['resources']['lut'] for r in rows)),
            peak_bram=math.ceil(max(r['resources']['bram36'] for r in rows)),
            peak_dsp=math.ceil(max(r['resources']['dsp'] for r in rows)),
            energy_mj=energy,violations=tuple(dict.fromkeys(violations)),physical_metrics=metrics,per_layer=())

    def reset_lut_stats(self):
        FPGACostEstimator.reset_lut_stats(self)
        self.last_selected_row_keys=[]
        self._operator_cost_queries=0
        self._operator_cost_hits=0
        self._operator_cost_misses=0

    def get_lut_stats(self):
        stats=FPGACostEstimator.get_lut_stats(self)
        total=self._operator_cost_queries
        stats.update(operator_cost_queries=total,operator_cost_hits=self._operator_cost_hits,
                    operator_cost_misses=self._operator_cost_misses,
                    operator_cost_hit_rate=self._operator_cost_hits/total if total else 0.0,
                    fallback_rate=0.0 if self._operator_cost_misses==0 else 1.0,
                    fallback_policy='no_analytic_fallback_unknown_exact_keys_raise')
        stats.update(cost_source=self.cost_source,search_ready=self.search_ready,
                    operator_rows=len(self.rows_by_key),payload_sha256=self.table['payload_sha256'],
                    mapping_id=MAPPING,hardware_lut_version=self.table['version'],
                    operator_cost_path=str(self.operator_cost_path),calibration_path=str(self.calibration_path),
                    storage_calibration_model=self.calibration.get('model_kind'),
                    diagnostic_only=self.diagnostic_only,calibration_validated=self.calibration.get('validated') is True,
                    last_selected_row_keys=list(self.last_selected_row_keys),
                    hardware_cost_scope='Exact OOC costs plus mapping-bound resource calibration; no measured board latency or power',
                    latency_ms_semantics='sum(exact operator HLS cycles) / 200 MHz; no whole-network timing closure yet',
                    power_w_semantics='max over isolated-operator Vivado vectorless estimates; not whole-network or measured board power',
                    energy_mj_semantics='sum(isolated operator vectorless power_w * operator cycles / 200 MHz) under serial execution')
        return stats
