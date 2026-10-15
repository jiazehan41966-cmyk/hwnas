"""Strict, reversible Linux CPU conditions for the separate D09 experiment.

Importing this module does not read or change the host.  Only an explicitly
constructed controller's apply() writes the one selected cpufreq policy and
the current process's thread affinity.  Unsupported modes never fall back.

Interfaces follow Linux CPUFreq and sched_setaffinity documentation:
https://www.kernel.org/doc/html/latest/admin-guide/pm/cpufreq.html
https://www.man7.org/linux/man-pages/man2/sched_setaffinity.2.html
"""
from __future__ import annotations

import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import math
import os
from pathlib import Path
import platform
import re
import sys
import subprocess
import time


THREAD_ENV = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
WRITABLE = {"scaling_governor", "scaling_min_freq", "scaling_max_freq", "scaling_setspeed"}


class CpuConditionError(RuntimeError):
    def __init__(self, message, snapshot=None):
        super().__init__(message)
        self.snapshot = snapshot


def utc():
    return datetime.now(timezone.utc).isoformat()


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(name + " must be an integer >= " + str(minimum))
    return value


def parse_cpus(text):
    result = set()
    for token in text.replace(",", " ").split():
        if re.fullmatch(r"\d+-\d+", token):
            low, high = map(int, token.split("-"))
            if high < low:
                raise ValueError("reversed CPU range")
            result.update(range(low, high + 1))
        elif token.isdigit():
            result.add(int(token))
        else:
            raise ValueError("invalid CPU list")
    if not result:
        raise ValueError("empty CPU list")
    return result


def validate_config(raw):
    config = dict(raw)
    if config.get("schema") != "d09_cpu_conditions_v1":
        raise ValueError("unrecognized CPU conditions schema")
    for name in ("campaign_id", "run_id"):
        if not isinstance(config.get(name), str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", config[name]):
            raise ValueError("invalid " + name)
    if config.get("experiment_phase") not in ("calibration", "validation"):
        raise ValueError("experiment_phase must be calibration or validation")
    integer(config.get("cpu_index"), "cpu_index")
    integer(config.get("frequency_khz"), "frequency_khz", 1)
    if not re.fullmatch(r"policy\d+", str(config.get("policy_name", ""))):
        raise ValueError("one explicit policy_name is required")
    config.setdefault("mode", "performance")
    if config["mode"] not in ("performance", "userspace"):
        raise ValueError("unsupported requested governor; no fallback")
    if config["mode"] == "userspace":
        approval = config.get("userspace_authorization", {})
        if (approval.get("authorized") is not True or not isinstance(approval.get("source"), str)
                or not approval["source"].strip()):
            raise ValueError("userspace requires separately recorded explicit user authorization")
    for name, value in (("batches", 3), ("warmup_calls", 50), ("measured_calls", 1000), ("group_size", 1)):
        config.setdefault(name, value)
        if type(config[name]) is not int or config[name] != value:
            raise ValueError("D09 fixes " + name + "=" + str(value))
    config.setdefault("settle_seconds", 3.0)
    if (isinstance(config["settle_seconds"], bool) or not isinstance(config["settle_seconds"], (int,float))
            or not math.isfinite(config["settle_seconds"]) or not 0 <= config["settle_seconds"] <= 10):
        raise ValueError("settle_seconds must be in [0,10]")
    config.setdefault("host_median_relative_limit", None)
    limit = config["host_median_relative_limit"]
    if limit is not None and (isinstance(limit,bool) or not isinstance(limit,(int,float))
                             or not math.isfinite(limit) or not 0 <= limit <= 1):
        raise ValueError("invalid explicit host median relative limit")
    if config["experiment_phase"] == "calibration":
        if limit is not None:
            raise ValueError("calibration must remain descriptive, with no fitted host gate")
    else:
        freeze = config.get("calibration_freeze", {})
        if (not isinstance(freeze.get("path"), str) or not freeze["path"]
                or not re.fullmatch(r"[0-9a-f]{64}", str(freeze.get("sha256", "")))):
            raise ValueError("validation requires a bound prior calibration_freeze")
        if limit is None:
            raise ValueError("validation requires its pre-frozen host median limit")
    return config


class LinuxSystem:
    """Small injectable OS boundary; tests use an in-memory implementation."""
    def __init__(self):
        self.policy_root = Path("/sys/devices/system/cpu/cpufreq")
        self.environ = os.environ

    def read(self, path):
        return Path(path).read_text(encoding="ascii").strip()

    def write(self, path, value):
        Path(path).write_text(str(value), encoding="ascii")

    def exists(self, path):
        return Path(path).is_file()

    def thread_ids(self):
        return sorted(int(path.name) for path in Path("/proc/self/task").iterdir() if path.name.isdigit())

    def get_affinity(self, tid):
        return set(os.sched_getaffinity(tid))

    def set_affinity(self, tid, cpus):
        os.sched_setaffinity(tid, set(cpus))

    def current_cpu(self):
        libc = ctypes.CDLL(None, use_errno=True)
        function = libc.sched_getcpu
        function.argtypes, function.restype = [], ctypes.c_int
        value = function()
        if value < 0:
            raise OSError(ctypes.get_errno(), "sched_getcpu failed")
        return value

    def online_cpus(self):
        return parse_cpus(self.read("/sys/devices/system/cpu/online"))

    def unit_state(self, unit):
        if not re.fullmatch(r"[A-Za-z0-9_.@-]+\.(service|timer|socket)",unit):
            raise ValueError("invalid explicit service unit")
        result=subprocess.run(["systemctl","show",unit,"--property=LoadState","--property=ActiveState",
                               "--property=SubState","--no-pager"],capture_output=True,text=True,timeout=5,check=False)
        if result.returncode:
            raise OSError("systemctl show failed: "+result.stderr.strip())
        return dict(line.split("=",1) for line in result.stdout.splitlines() if "=" in line)

    def sha_file(self, path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def versions(self):
        return software_versions()


def software_versions():
    packages = {}
    for name in ("pynq", "numpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "NOT_INSTALLED"
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                kernel=platform.release(), kernel_release=platform.release(), python_version=platform.python_version(),
                pynq=packages["pynq"], numpy=packages["numpy"], packages=packages,
                threading_environment={name:os.environ.get(name) for name in THREAD_ENV})


def external_snapshot(config, label, *, system=None):
    """Read-only checks of root-managed services/IRQs, boot, versions, file hashes."""
    system=system if system is not None else LinuxSystem()
    expected=config["system_conditions"]
    errors=[]
    result=dict(label=label,timestamp_utc=utc(),services={},irq_affinity={},runtime_files={},
                ownership="Root manages services/IRQs; this runner only reads and validates them.")
    try:
        result["software_versions"]=system.versions()
        for name,wanted in config["expected_versions"].items():
            if result["software_versions"].get(name) != wanted:
                errors.append("VERSION_MISMATCH:"+name)
    except Exception as exc:
        errors.append("VERSION_READ_FAILED:"+repr(exc))
    try:
        result["boot_id"]=system.read("/proc/sys/kernel/random/boot_id")
        if result["boot_id"] != expected["boot_id"]:
            errors.append("BOOT_ID_CHANGED")
    except Exception as exc:
        errors.append("BOOT_ID_READ_FAILED:"+repr(exc))
    for unit in expected["stopped_units"]:
        try:
            state=system.unit_state(unit)
            result["services"][unit]=state
            if state.get("LoadState") != "loaded" or state.get("ActiveState") != "inactive":
                errors.append("SERVICE_NOT_INACTIVE:"+unit)
        except Exception as exc:
            errors.append("SERVICE_STATE_READ_FAILED:"+unit+":"+repr(exc))
    for irq,wanted in expected["irq_affinity"].items():
        try:
            configured=system.read("/proc/irq/"+irq+"/smp_affinity_list")
            effective=system.read("/proc/irq/"+irq+"/effective_affinity_list")
            result["irq_affinity"][irq]=dict(requested=wanted,configured=configured,effective=effective)
            allowed=parse_cpus(wanted)
            actual,effective_set=parse_cpus(configured),parse_cpus(effective)
            if actual != allowed or not effective_set <= allowed or config["cpu_index"] in effective_set:
                errors.append("IRQ_AFFINITY_MISMATCH:"+irq)
        except Exception as exc:
            errors.append("IRQ_AFFINITY_READ_FAILED:"+irq+":"+repr(exc))
    for path,wanted in config["runtime_files"].items():
        try:
            actual=system.sha_file(path)
            result["runtime_files"][path]=actual
            if actual != wanted:
                errors.append("RUNTIME_FILE_SHA_MISMATCH:"+path)
        except Exception as exc:
            errors.append("RUNTIME_FILE_READ_FAILED:"+path+":"+repr(exc))
    result.update(status="PASS_EXTERNAL_CONDITIONS" if not errors else "FAIL_EXTERNAL_CONDITIONS",condition_deviations=errors)
    return result


class CpuConditions:
    def __init__(self, config, *, system=None, journal_callback=None):
        self.config = validate_config(config)
        self.system = system if system is not None else LinuxSystem()
        self.policy = self.system.policy_root / self.config["policy_name"]
        self.original, self.original_affinity = None, {}
        self.journal, self.related_cpus = [], []
        self.callback = journal_callback
        self.state = "NOT_APPLIED"
        self.restore_result = None

    def state_record(self):
        return dict(schema="d09_cpu_transaction_v1", state=self.state, policy=str(self.policy),
                    config=self.config, original=self.original,
                    original_affinity={str(k):sorted(v) for k,v in self.original_affinity.items()},
                    journal=list(self.journal), related_cpus=self.related_cpus, timestamp_utc=utc())

    def _record(self, event):
        self.journal.append(dict(timestamp_utc=utc(), **event))
        if self.callback:
            self.callback(self.state_record())

    def _read(self, field):
        return self.system.read(self.policy / field)

    def _write(self, field, value):
        if field not in WRITABLE:
            raise ValueError("CPU write outside explicit policy whitelist")
        value = str(value)
        if self._read(field) == value:
            return
        self._record(dict(action="POLICY_WRITE_ATTEMPT", field=field, requested=value))
        self.system.write(self.policy / field, value)
        actual = self._read(field)
        self._record(dict(action="POLICY_WRITE_READBACK", field=field, requested=value, actual=actual))
        if actual != value:
            raise CpuConditionError("CPU policy write readback differs: " + field)

    def _limits(self, low, high):
        # Never ask sysfs to cross the currently installed opposing bound.
        if low > high:
            raise ValueError("invalid CPU frequency bounds")
        if high < int(self._read("scaling_min_freq")):
            self._write("scaling_min_freq", low)
            self._write("scaling_max_freq", high)
        else:
            self._write("scaling_max_freq", high)
            self._write("scaling_min_freq", low)

    def preflight(self):
        errors = []
        env = {name:self.system.environ.get(name) for name in THREAD_ENV}
        if any(value != "1" for value in env.values()):
            errors.append("THREAD_ENVIRONMENT_MUST_ALL_EQUAL_1")
        tids = self.system.thread_ids()
        if len(tids) != 1:
            errors.append("SINGLE_OS_THREAD_REQUIRED")
        affinities = {tid:self.system.get_affinity(tid) for tid in tids}
        cpu = self.config["cpu_index"]
        if cpu not in self.system.online_cpus() or any(cpu not in mask for mask in affinities.values()):
            errors.append("REQUESTED_CPU_NOT_ONLINE_OR_ALLOWED")
        available = self._read("scaling_available_governors").split()
        if self.config["mode"] not in available:
            errors.append("REQUESTED_GOVERNOR_NOT_AVAILABLE:" + self.config["mode"])
        related = parse_cpus(self._read("related_cpus"))
        if cpu not in related:
            errors.append("CPU_NOT_IN_SELECTED_POLICY")
        supported = [int(value) for value in self._read("scaling_available_frequencies").split()]
        if self.config["frequency_khz"] not in supported:
            errors.append("REQUESTED_FREQUENCY_NOT_ADVERTISED")
        original = {name:self._read(name) for name in ("scaling_governor","scaling_min_freq","scaling_max_freq")}
        original["scaling_setspeed"] = self._read("scaling_setspeed") if self.system.exists(self.policy/"scaling_setspeed") else None
        if original["scaling_governor"] == "userspace" and not str(original["scaling_setspeed"]).isdigit():
            errors.append("ORIGINAL_USERSPACE_FREQUENCY_NOT_RESTORABLE")
        if self.config["mode"] == "userspace" and original["scaling_setspeed"] is None:
            errors.append("USERSPACE_SET_SPEED_INTERFACE_MISSING")
        if int(original["scaling_min_freq"]) > int(original["scaling_max_freq"]):
            errors.append("INVALID_ORIGINAL_FREQUENCY_BOUNDS")
        for name in ("scaling_cur_freq", "cpuinfo_cur_freq"):
            if not self.system.exists(self.policy/name) or not self._read(name).isdigit():
                errors.append("STRICT_CURRENT_FREQUENCY_EVIDENCE_MISSING:" + name)
        report = dict(status="PASS_PREFLIGHT" if not errors else "FAIL_PREFLIGHT", errors=errors,
                      available_governors=available, available_frequencies_khz=supported,
                      thread_ids=tids, affinities={str(k):sorted(v) for k,v in affinities.items()},
                      threading_environment=env, related_cpus=sorted(related), original=original)
        self._record(dict(action="PREFLIGHT", result=report))
        if errors:
            raise CpuConditionError("CPU conditions preflight failed", report)
        self.original, self.original_affinity = original, affinities
        self.related_cpus, self.state = sorted(related), "PREFLIGHTED"
        self._record(dict(action="ORIGINAL_STATE_CAPTURED"))
        return report

    def apply(self):
        if self.state != "NOT_APPLIED":
            raise RuntimeError("CPU controller cannot be applied twice")
        self.preflight()
        self.state = "APPLYING"
        for tid in self.original_affinity:
            self._record(dict(action="AFFINITY_WRITE_ATTEMPT", tid=tid, requested=[self.config["cpu_index"]]))
            self.system.set_affinity(tid, {self.config["cpu_index"]})
        self._write("scaling_governor", self.config["mode"])
        frequency = self.config["frequency_khz"]
        self._limits(frequency, frequency)
        if self.config["mode"] == "userspace":
            self._write("scaling_setspeed", frequency)
        deadline = time.monotonic() + self.config["settle_seconds"]
        while True:
            record = self.snapshot("after_apply")
            if not record["condition_deviations"]:
                self.state = "APPLIED"
                self._record(dict(action="APPLIED", result=record))
                return record
            if time.monotonic() >= deadline:
                raise CpuConditionError("requested CPU conditions did not settle", record)
            time.sleep(0.02)

    def snapshot(self, label):
        errors, frequencies, policies, affinities = [], {}, [], {}
        try:
            tids = self.system.thread_ids()
            affinities = {str(tid):sorted(self.system.get_affinity(tid)) for tid in tids}
            cpu = self.system.current_cpu()
            if len(tids) != 1:
                errors.append("SINGLE_OS_THREAD_REQUIRED")
            if any(mask != [self.config["cpu_index"]] for mask in affinities.values()) or cpu != self.config["cpu_index"]:
                errors.append("CPU_AFFINITY_OR_CURRENT_CPU_DRIFT")
        except (OSError, ValueError) as exc:
            tids, cpu = [], None
            errors.append("CPU_AFFINITY_READ_FAILED:" + str(exc))
        actual = "UNAVAILABLE"
        try:
            actual = self._read("scaling_governor")
            available = self._read("scaling_available_governors").split()
            policies = [dict(path=str(self.policy/"scaling_governor"), actual=actual, available=available)]
            if actual != self.config["mode"]:
                errors.append("CPU_GOVERNOR_DRIFT")
            for name in ("scaling_min_freq","scaling_max_freq","scaling_cur_freq","cpuinfo_cur_freq"):
                raw = self._read(name)
                frequencies[str(self.policy/name)] = raw
                if not raw.isdigit() or int(raw) != self.config["frequency_khz"]:
                    errors.append("CPU_FREQUENCY_DRIFT:" + name)
            if self.config["mode"] == "userspace":
                raw = self._read("scaling_setspeed")
                frequencies[str(self.policy/"scaling_setspeed")] = raw
                if not raw.isdigit() or int(raw) != self.config["frequency_khz"]:
                    errors.append("CPU_USERSPACE_SET_SPEED_DRIFT")
        except (OSError,ValueError) as exc:
            errors.append("CPU_POLICY_READ_FAILED:" + str(exc))
        return dict(schema="d09_cpu_observation_v1", label=label, timestamp_utc=utc(),
                    cpu_governor=actual, policies=policies, frequency_raw_khz=frequencies,
                    thread_ids=tids, thread_count=len(tids), affinities=affinities, current_cpu=cpu,
                    policy_related_cpus=self.related_cpus, requested_mode=self.config["mode"],
                    requested_frequency_khz=self.config["frequency_khz"], condition_deviations=errors,
                    status="PASS_CPU_CONDITIONS" if not errors else "FAIL_CPU_CONDITIONS",
                    evidence_scope="Affinity and cpufreq register/sysfs observations outside host call timing; no independent physical frequency measurement.")

    @staticmethod
    def require(record):
        if record["condition_deviations"]:
            raise CpuConditionError("CPU condition drift detected", record)

    def restore(self):
        if self.restore_result is not None:
            return self.restore_result
        if self.original is None:
            self.restore_result = dict(status="PASS_RESTORED", no_mutations=True, changes="NONE_PREFLIGHT_DID_NOT_PASS", failures=[])
            return self.restore_result
        self.state, failures = "RESTORING", []
        try:
            self._limits(int(self.original["scaling_min_freq"]), int(self.original["scaling_max_freq"]))
        except Exception as exc:
            failures.append(dict(step="frequency_bounds", error=repr(exc)))
        try:
            self._write("scaling_governor", self.original["scaling_governor"])
        except Exception as exc:
            failures.append(dict(step="governor", error=repr(exc)))
        if self.original["scaling_governor"] == "userspace":
            try:
                self._write("scaling_setspeed", self.original["scaling_setspeed"])
            except Exception as exc:
                failures.append(dict(step="userspace_frequency", error=repr(exc)))
        restored_affinity = {}
        try:
            tids = self.system.thread_ids()
            fallback = next(iter(self.original_affinity.values()))
            for tid in tids:
                wanted = self.original_affinity.get(tid, fallback)
                try:
                    self.system.set_affinity(tid, wanted)
                    actual = self.system.get_affinity(tid)
                    restored_affinity[str(tid)] = sorted(actual)
                    if actual != wanted:
                        raise RuntimeError("affinity restore readback differs")
                except Exception as exc:
                    failures.append(dict(step="affinity", tid=tid, error=repr(exc)))
        except Exception as exc:
            failures.append(dict(step="thread_list", error=repr(exc)))
        restored = {}
        for field, expected in self.original.items():
            if field == "scaling_setspeed" and self.original["scaling_governor"] != "userspace":
                continue  # This attribute is only functional in userspace mode.
            try:
                restored[field] = self._read(field)
                if restored[field] != expected:
                    failures.append(dict(step="final_readback", field=field, expected=expected, actual=restored[field]))
            except Exception as exc:
                failures.append(dict(step="final_readback", field=field, error=repr(exc)))
        restored_frequency = {}
        for field in ("scaling_cur_freq", "cpuinfo_cur_freq"):
            try:
                restored_frequency[field] = self._read(field)
            except Exception as exc:
                failures.append(dict(step="restored_current_frequency_read", field=field, error=repr(exc)))
        self.state = "RESTORED" if not failures else "RESTORE_FAILED"
        self.restore_result = dict(status="PASS_RESTORED" if not failures else "FAIL_RESTORE", no_mutations=False,
            original=self.original, restored=restored, restored_affinity=restored_affinity,
            original_affinity={str(tid):sorted(cpus) for tid,cpus in self.original_affinity.items()},
            restored_current_frequency_raw_khz=restored_frequency,
            failures=failures, policy=str(self.policy), timestamp_utc=utc())
        self._record(dict(action="RESTORE_FINISHED", result=self.restore_result))
        return self.restore_result
