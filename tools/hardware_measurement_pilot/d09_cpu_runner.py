"""D09 CPU-controlled retest using unchanged attempt05 hardware/timing code.

This is a separate entry point.  It never changes the old runner, statistics,
bitstream, or phase contract.  CPU checks and their logging are outside the
unchanged _record_group host window.  The shared transport owns the board lock.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import re
import statistics
import time


HERE = Path(__file__).resolve().parent

def _module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE/filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

board = _module("d09_frozen_board_runner", "board_runner.py")
cpu = _module("d09_cpu_conditions", "cpu_conditions.py")
CASE_IDS = {"pw1x1_int8_reference_high", "dw5_int8_reference_low", "body8_to_head16_reference_high"}


def _bound_json(root, bundle, reference, label):
    if not isinstance(reference,dict) or not isinstance(reference.get("path"),str):
        raise ValueError(label + " requires a bound JSON reference")
    path = board._child(root, reference["path"])
    relative = path.relative_to(root).as_posix()
    digest = board._sha(path)
    if digest != reference.get("sha256") or bundle["files"].get(relative) != digest:
        raise ValueError(label + " file/hash is absent or differs from bundle")
    return board._load(path), digest


def validate_runtime(case_dir, config_path):
    root,bundle,case,metadata,registers = board.validate_bundle(case_dir,mode="timing")
    if case["case_id"] not in CASE_IDS:
        raise ValueError("case is outside the authorized three-case D09 retest")
    config_path = Path(config_path).resolve()
    if config_path != root/"cpu_conditions.json":
        raise ValueError("--cpu-config must be the bound case/cpu_conditions.json")
    for name,path in (("d09_cpu_runner.py",Path(__file__)),("cpu_conditions.py",Path(cpu.__file__)),("cpu_conditions.json",config_path)):
        if bundle["files"].get(name) != board._sha(path) or board._sha(root/name) != board._sha(path):
            raise ValueError("executing D09 source/config differs from package: " + name)
    config = cpu.validate_config(board._load(config_path))
    protocol,protocol_hash = _bound_json(root,bundle,config.get("protocol"),"D09 protocol")
    baseline,baseline_hash = _bound_json(root,bundle,config.get("pl_baseline"),"PL baseline")
    expected = cpu.integer(config.get("pl_compute_expected_cycles"),"pl_compute_expected_cycles",1)
    row = baseline.get("cases",{}).get(case["case_id"]) if isinstance(baseline.get("cases"),dict) else None
    if row is None and baseline.get("case_id") == case["case_id"]:
        row = baseline
    if (not isinstance(row,dict) or row.get("compute_cycles") != expected
            or row.get("case_sha256") != case["case_sha256"]
            or row.get("bitstream_sha256") != bundle["files"]["vivado_shell/top.bit"]):
        raise ValueError("PL baseline must bind exact case, bitstream and compute cycles")
    # The shared protocol must bind the same physical CPU condition in both phases.
    condition = protocol.get("cpu_conditions",protocol)
    for key in ("mode","cpu_index","policy_name","frequency_khz"):
        if condition.get(key) != config[key]:
            raise ValueError("CPU runtime condition differs from common protocol: " + key)
    for field in ("system_conditions","expected_versions","runtime_files"):
        if not isinstance(config.get(field),dict) or protocol.get(field) != config[field]:
            raise ValueError("runtime condition is missing or differs from common protocol: " + field)
    if set(config["expected_versions"]) != {"kernel_release","python_version","pynq","numpy"} or any(
            not isinstance(value,str) or not value for value in config["expected_versions"].values()):
        raise ValueError("all four exact software versions are required")
    systems=config["system_conditions"]
    if (not isinstance(systems.get("boot_id"),str) or not re.fullmatch(r"[0-9a-fA-F-]{36}",systems["boot_id"])
            or not isinstance(systems.get("stopped_units"),list) or not all(isinstance(unit,str) for unit in systems["stopped_units"])
            or not isinstance(systems.get("irq_affinity"),dict)):
        raise ValueError("explicit boot, stopped_units and IRQ conditions are required")
    for irq,cpus in systems["irq_affinity"].items():
        if not isinstance(irq,str) or not irq.isdigit() or config["cpu_index"] in cpu.parse_cpus(cpus):
            raise ValueError("invalid or non-isolating IRQ affinity condition")
    if not config["runtime_files"] or any(not isinstance(path,str) or not path.startswith("/")
            or not re.fullmatch(r"[0-9a-f]{64}",str(digest)) for path,digest in config["runtime_files"].items()):
        raise ValueError("absolute runtime file paths and exact SHA256 values are required")
    freeze_hash = None
    if config["experiment_phase"] == "validation":
        freeze,freeze_hash = _bound_json(root,bundle,config["calibration_freeze"],"calibration freeze")
        if freeze.get("host_median_relative_limit") != config["host_median_relative_limit"]:
            raise ValueError("validation host limit differs from frozen calibration threshold")
        runs = freeze.get("calibration_run_ids")
        if not isinstance(runs,list) or not runs or config["run_id"] in runs or len(set(runs)) != len(runs):
            raise ValueError("validation must use a new run after distinct calibration runs")
        if freeze.get("d09_protocol_sha256",freeze.get("protocol_sha256")) != protocol_hash:
            raise ValueError("calibration freeze does not bind the common D09 protocol")
        created = freeze.get("created_utc")
        if not isinstance(created,str) or datetime.fromisoformat(created) >= datetime.fromisoformat(board._utc()):
            raise ValueError("calibration freeze must precede validation start")
    external_hash = None
    if config.get("external_conditions_evidence") is not None:
        _,external_hash = _bound_json(root,bundle,config["external_conditions_evidence"],"external conditions evidence")
    binding = dict(experiment_phase=config["experiment_phase"],run_id=config["run_id"],
        d09_protocol_sha256=protocol_hash,cpu_conditions_config_sha256=board._sha(config_path),
        host_thresholds_sha256=freeze_hash,pl_baseline_sha256=baseline_hash,
        external_conditions_evidence_sha256=external_hash,
        source_files={name:bundle["files"][name] for name in ("board_runner.py","measurement.py","cpu_conditions.py","d09_cpu_runner.py")})
    return root,bundle,case,metadata,registers,config,binding


class CounterWriter:
    """Preserve original counter writes; retain last complete record for later checks."""
    def __init__(self, stream):
        self.stream, self.last_record = stream, None

    def write(self, text):
        result = self.stream.write(text)
        self.last_record = json.loads(text)
        return result

    def flush(self):
        self.stream.flush()


def _cpu_checkpoint(controller, label, stream=None):
    snapshot = controller.snapshot(label)
    if stream is not None:
        stream.write(json.dumps(snapshot,allow_nan=False)+"\n")
        stream.flush()
    controller.require(snapshot)
    return snapshot


def _record_checked_group(controller, config, session, row_base, batch_id, phase, iteration, writer, stream, counters, observations):
    # CPU reads/logging cannot enter the unchanged group's host timing interval.
    label = batch_id+":"+phase+":"+str(iteration)
    _cpu_checkpoint(controller,label+":before",observations)
    total = board._record_group(session,row_base,batch_id,phase,iteration,1,
                               phase+"_"+str(iteration),writer,stream,counters)
    after = controller.snapshot(label+":after")
    observations.write(json.dumps(after,allow_nan=False)+"\n")
    observations.flush()
    # _record_group has already persisted the original row before these gates.
    record = counters.last_record
    if record.get("compute_cycles") != config["pl_compute_expected_cycles"]:
        raise cpu.CpuConditionError("PL compute cycles differ from bound attempt05 baseline",record)
    controller.require(after)
    return total


def _restore(controller, output):
    try:
        result = (controller.restore() if controller is not None else
                  dict(status="PASS_RESTORED",no_mutations=True,reason="EXIT_BEFORE_CPU_CONTROLLER",failures=[]))
    except BaseException as exc:
        result = dict(status="FAIL_RESTORE",no_mutations=False,error=repr(exc),
                      transaction=controller.state_record() if controller is not None else None)
    board._save(output/"cpu_conditions_restore.json",result)
    if result.get("status") != "PASS_RESTORED":
        board._save(output/"cpu_conditions_retention.json",dict(status="CPU_CONDITIONS_UNKNOWN_RETAIN_BOARD_LOCK",
            reason="CPU restoration could not be verified",automatic_retry=False,restore=result))
    return result


class DeferredRestore:
    def __init__(self, controller, output):
        self.controller,self.output = controller,output

    def restore(self):
        return _restore(self.controller,self.output)


def run_case(case_dir, output, board_id, cpu_config, *, timeout=30.0, controller_factory=cpu.CpuConditions):
    output = board._new_output(case_dir,output,timeout)
    started,session,controller,success,summary = time.monotonic(),None,None,False,None
    try:
        root,bundle,case,metadata,registers,config,binding = validate_runtime(case_dir,cpu_config)
        identity = board.verified_board_identity(board_id)
        board._save(output/"board_identity.json",identity)
        external=cpu.external_snapshot(config,"startup")
        board._save(output/"system_conditions_startup.json",external)
        cpu.CpuConditions.require(external)
        controller = controller_factory(config,journal_callback=lambda record:board._save(output/"cpu_conditions_transaction.json",record))
        setup = controller.apply()
        board._save(output/"cpu_conditions_setup.json",setup)
        fixture = case["fixtures"][0]
        base = dict(case_id=case["case_id"],case_sha256=case["case_sha256"],numeric_contract_sha256=case["numeric_contract_sha256"],
                    bitstream_sha256=bundle["files"]["vivado_shell/top.bit"],board_id=board_id,
                    expected_output_sha256=fixture["expected_output_sha256"])
        manifest = dict(cases=[dict(**base,expected_elements=fixture["expected_elements"],clock_hz=metadata["clock_hz"])],batches=[],
            independent_batches_minimum=3,warmup_minimum=50,measured_minimum=1000,cv_limit=.05,
            host_median_relative_limit=config["host_median_relative_limit"],protocol_status="DRAFT",
            phase_clock_evidence_verified=False,actual_clock_register_evidence_verified=False,
            independent_clock_scale_calibration="NOT_RUN",**binding)
        board._save(output/"timing_manifest.json",manifest)
        board._save(output/"session.json",dict(**base,**binding,board_bundle_sha256=board._sha(root/"board_bundle.json"),
            phase_validation_sha256=bundle["files"][bundle["phase_validation"]],
            actual_clock_source="pynq.Clocks.fclk0_mhz_before_after_each_group_and_batch",
            independent_clock_scale_calibration="NOT_RUN",
            call_scope="Unchanged attempt05 group host interval, N=1: output poison + idle check + used-buffer flush + pointer writes + start + done/idle poll + output invalidate + preallocated output copy + raw counter capture",
            excludes="CPU condition reads, temperature/version reads, DMA/array allocation, input preparation, comparisons, hashes, logs, bitstream reload",
            compute_scope="validated phase 2 on-chip interval",transfer_scope="validated phase 1 load plus phase 3 store intervals",
            group_selection="D09 frozen N=1; exactly 50 warmup and 1000 measured calls per each of three fresh reload batches",
            cpu_mode=config["mode"],cpu_index=config["cpu_index"],frequency_khz=config["frequency_khz"],
            pl_compute_expected_cycles=config["pl_compute_expected_cycles"],protocol_status="DRAFT",formal_cost_eligible=False))
        csv_path = output/"timing_raw.csv"
        with csv_path.open("x",encoding="utf-8",newline="") as stream, \
                (output/"counter_raw.jsonl").open("x",encoding="utf-8") as raw_counters, \
                (output/"cpu_group_observations.jsonl").open("x",encoding="utf-8") as observations:
            writer = csv.DictWriter(stream,fieldnames=board.measurement.CSV_FIELDS)
            writer.writeheader(); stream.flush()
            counters = CounterWriter(raw_counters)
            for batch_index in range(3):
                batch_id="batch"+str(batch_index+1)
                folder=output/batch_id
                folder.mkdir(exist_ok=False)
                pre_reload=_cpu_checkpoint(controller,batch_id+":before_reload")
                board._save(folder/"cpu_before_reload.json",pre_reload)
                session=board.BoardSession(root,case,metadata,registers,timeout,progress=lambda event:board._progress(folder,event))
                environment_start=dict(cpu=_cpu_checkpoint(controller,batch_id+":start"),temperature=board.temperature_snapshot(),
                    clock_hz=session._clock(),external_conditions=cpu.external_snapshot(config,batch_id+":start"),timestamp_utc=board._utc())
                environment_start["software_versions"]=environment_start["external_conditions"].get("software_versions")
                batch=dict(case_id=case["case_id"],batch_id=batch_id,independence_event="reload_bitstream",
                    independence_id=session.independence_id,clock_hz=session.clock_hz,clock_hz_end=None,
                    cpu_governor=environment_start["cpu"]["cpu_governor"],temperature_c=environment_start["temperature"]["temperature_c"],
                    temperature_c_end=None,started_utc=session.downloaded_at_utc,finished_utc=None,protocol_status="DRAFT",
                    experiment_phase=config["experiment_phase"],run_id=config["run_id"],cpu_index=config["cpu_index"])
                manifest["batches"].append(batch)
                board._save(output/"timing_manifest.json",manifest)
                board._save(folder/"register_abi.json",session.abi)
                board._save(folder/"environment_start.json",environment_start)
                cpu.CpuConditions.require(environment_start["external_conditions"])
                correctness=board._correctness(session,case,folder,validate_phase=True)
                if correctness[0]["compute_cycles"] != config["pl_compute_expected_cycles"]:
                    raise cpu.CpuConditionError("fixed fixture0 PL compute cycles differ from baseline",correctness[0])
                session.select_fixture(0)
                row_base=dict(**base,clock_hz=session.clock_hz)
                warmups=[]
                for iteration in range(50):
                    warmups.append(_record_checked_group(controller,config,session,row_base,batch_id,"warmup",iteration,
                                                        writer,stream,counters,observations))
                board._save(folder/"group_selection.json",dict(warmup_rows=50,warmup_host_median_ns=statistics.median(warmups),
                    measured_group_size=1,measured_calls=1000,selected_before_measured=True,
                    selection_basis="FROZEN_D09_N1_PROTOCOL",protocol_status="DRAFT"))
                for iteration in range(1000):
                    _record_checked_group(controller,config,session,row_base,batch_id,"measured",iteration,
                                          writer,stream,counters,observations)
                environment_end=dict(cpu=controller.snapshot(batch_id+":end"),temperature=board.temperature_snapshot(),
                    clock_hz=session._clock(),external_conditions=cpu.external_snapshot(config,batch_id+":end"),timestamp_utc=board._utc())
                environment_end["software_versions"]=environment_end["external_conditions"].get("software_versions")
                board._save(folder/"environment_end.json",environment_end)
                batch.update(clock_hz_end=environment_end["clock_hz"],temperature_c_end=environment_end["temperature"]["temperature_c"],
                    finished_utc=environment_end["timestamp_utc"],
                    condition_deviations=(environment_start["cpu"]["condition_deviations"]+
                        environment_end["cpu"]["condition_deviations"]+environment_start["temperature"]["condition_deviations"]+
                        environment_end["temperature"]["condition_deviations"]+environment_start["external_conditions"]["condition_deviations"]+
                        environment_end["external_conditions"]["condition_deviations"]))
                board._save(output/"timing_manifest.json",manifest)
                controller.require(environment_end["cpu"])
                cpu.CpuConditions.require(environment_end["external_conditions"])
                session.close(); session=None
        manifest["actual_clock_register_evidence_verified"]=all(b["clock_hz"] == b["clock_hz_end"] for b in manifest["batches"])
        manifest["phase_clock_evidence_verified"]=manifest["actual_clock_register_evidence_verified"]
        board._save(output/"timing_manifest.json",manifest)
        summary=board.measurement.summarize_csv(csv_path,manifest)
        summary.update(correctness_fixtures_passed=15,integrated_phase_validation="PASS_BOUND_ARTIFACT",
            board_bundle_sha256=board._sha(root/"board_bundle.json"),protocol_status="DRAFT",formal_cost_eligible=False,
            **binding,d09_cpu_conditions_passed=True,pl_compute_baseline_passed=True,
            cpu_observations_scope="before/after every N=1 group and every batch; all outside call window",
            old_performance_condition_unmodified=True,
            d09_cpu_equivalence_scope="Explicitly authorized userspace fixed-frequency alternative" if config["mode"] == "userspace" else "performance fixed-frequency control")
        board._save(output/"summary.json",summary)
        success=summary["status"] == "PASS"
        return summary
    except BaseException as exc:
        if isinstance(exc,cpu.CpuConditionError):
            board._save(output/"cpu_condition_failure.json",dict(status="FAIL",reason=str(exc),observation=exc.snapshot,
                raw_observations_retained=True,automatic_retry=False))
        board._failure(output,session,exc)
        raise
    finally:
        if board.QUARANTINE:
            board._save(output/"cpu_conditions_restore.json",dict(status="DEFERRED_DMA_QUARANTINE",no_mutations=False,
                reason="Restore deferred until the original runner proves the core idle and releases its buffers"))
            board.GOVERNOR_QUARANTINE.append((DeferredRestore(controller,output),output))
        else:
            restored=_restore(controller,output)
            if restored.get("status") != "PASS_RESTORED":
                success=False
                if summary is not None:
                    summary.update(status="FAIL",cpu_restoration_status="FAIL_RESTORE")
                    board._save(output/"summary.json",summary)
            elif summary is not None:
                summary["cpu_restoration_status"]="PASS_RESTORED"
                board._save(output/"summary.json",summary)
        board._stage(output,started,success,stage="d09_board_retest")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--board-id",required=True)
    parser.add_argument("--mode",choices=("timing",),default="timing")
    parser.add_argument("--timeout-seconds",type=float,default=30.0)
    parser.add_argument("--cpu-config",type=Path,required=True)
    args=parser.parse_args()
    try:
        result=run_case(args.case_dir,args.output,args.board_id,args.cpu_config,timeout=args.timeout_seconds)
    except board.UnsafeDeviceState:
        board._retain_cli(args.output)  # Original no-reset/no-free-until-idle quarantine loop.
        return 1
    print(result["status"],flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
