"""Read-only, single-sensor power capture with explicit rail evidence.

The verified ZCU104 U183 IIO channel or an explicitly verified hwmon channel
is used alone, never summed. Sensor configuration is never written. Capture alone is not a
candidate power PASS; energy needs a same-clock completed-inference window.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import threading
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _digest(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def inventory(hwmon):
    """Explicit path only. Read identities and scalar attributes, no probing."""
    folder = Path(hwmon)
    result = dict(path=str(folder), resolved_path=str(folder.resolve()), attributes={}, links={})
    for name in ("device", "device/driver", "device/of_node", "of_node"):
        path = folder / name
        if path.exists(): result["links"][name] = str(path.resolve())
    for path in sorted(folder.iterdir()):
        if path.is_file() and (path.name in ("name", "shunt_resistor", "update_interval") or
                re.fullmatch(r"(?:power|curr|in)\d+_(?:input|label)", path.name)):
            result["attributes"][path.name] = path.read_text().strip()
    return result


def validate_sensor(manifest):
    if manifest.get("schema_version") != "zcu104-power-sensor-v1" or manifest.get("status") != "VERIFIED":
        raise ValueError("explicit verified sensor/rail manifest required")
    if manifest.get("board") != "ZCU104" or manifest.get("test_evaluated") is not False:
        raise ValueError("wrong board or test seal")
    if manifest.get("interface") != "hwmon" or manifest.get("power_unit") != "microwatt":
        raise ValueError("only one hwmon power_input channel is admitted; no IIO conversion/sum")
    for key in ("deployment_binding_sha256", "package_manifest_sha256", "board_manifest_sha256"):
        if not _digest(manifest.get(key)): raise ValueError("missing deployment identity: " + key)
    folder = Path(manifest["hwmon_path"])
    if str(folder.resolve()) != manifest.get("resolved_hwmon_path"):
        raise ValueError("hwmon symlink identity changed")
    actual = inventory(folder)
    if actual["attributes"].get("name") != manifest.get("sensor_name"):
        raise ValueError("sensor name changed")
    evidence = manifest.get("rail_mapping_evidence", [])
    if not evidence:
        raise ValueError("rail mapping evidence missing; sensor name alone does not prove total board power")
    for item in evidence:
        if not _digest(item.get("sha256")) or sha(item["path"]) != item["sha256"]:
            raise ValueError("rail mapping evidence SHA mismatch")
    name = manifest["power_attribute"]
    if not re.fullmatch(r"power[1-9][0-9]*_input", name) or name not in actual["attributes"]:
        raise ValueError("explicit power_input attribute unavailable")
    scope = manifest.get("power_scope")
    if scope == "board_input_total_12v":
        if (manifest["sensor_name"] not in ("ina226", "ina2xx") or name != "power1_input" or
                manifest.get("reference_designator") != "U183" or manifest.get("rail") != "VCC12" or
                actual["attributes"].get("shunt_resistor") != "5000"):
            raise ValueError("VCC12 U183 identity/shunt not verified")
        voltage = float(actual["attributes"]["in1_input"]) / 1000
        if not math.isfinite(voltage) or not 9 <= voltage <= 15:
            raise ValueError("VCC12 bus voltage implausible")
    elif scope != "auxiliary_rail" or not manifest.get("rail"):
        raise ValueError("power scope must be explicit total input or named auxiliary rail")
    if not manifest.get("verified_links"):
        raise ValueError("physical-device/driver topology evidence required")
    for key, expected in manifest["verified_links"].items():
        if actual["links"].get(key) != expected: raise ValueError("physical sensor topology changed")
    return actual


def power_sample(manifest):
    value = float((Path(manifest["hwmon_path"]) / manifest["power_attribute"]).read_text().strip()) / 1e6
    if not math.isfinite(value) or value < 0:
        raise ValueError("nonfinite/negative power sensor reading")
    return value


def capture(manifest_path, output, *, phase, duration_s=60., sample_rate_hz=10., clock=time.monotonic, sleep=time.sleep):
    """Sample one verified sensor; external runner owns inference and timings."""
    output = Path(output); manifest = read(manifest_path)
    if phase not in ("idle", "active") or not math.isfinite(duration_s) or duration_s < 60:
        raise ValueError("idle/active capture requires at least60 seconds")
    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("positive finite sample rate required")
    try:
        sensor = validate_sensor(manifest)
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except (OSError, ValueError, KeyError) as exc:
        return dict(status="NOT_MEASURED", reason=str(exc), test_evaluated=False)
    output.mkdir(parents=True, exist_ok=False)
    csv_path = output / "samples.csv"; rows = []; start = clock(); due = start
    result = dict(status="NOT_MEASURED", phase=phase, sensor_manifest_sha256=sha(manifest_path),
        sensor=manifest, actual_sensor=sensor, boot_id=boot_id, clock="CLOCK_MONOTONIC",
        requested_rate_hz=sample_rate_hz, test_evaluated=False,
        claim_boundary="sensor capture only; no FPGA power admission inferred")
    try:
        with csv_path.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["monotonic_s", "read_duration_s", "power_w"]); writer.writeheader()
            while True:
                now = clock()
                if now < due:
                    sleep(min(due - now, .05)); continue
                before = clock(); value = power_sample(manifest); after = clock()
                row = dict(monotonic_s=(before + after) / 2, read_duration_s=after-before, power_w=value)
                rows.append(row); writer.writerow(row); stream.flush()
                if after - start >= duration_s: break
                due += 1 / sample_rate_hz
        if len(rows) < 2 or any(b["monotonic_s"] <= a["monotonic_s"] for a,b in zip(rows,rows[1:])):
            raise ValueError("insufficient/non-monotonic samples")
        elapsed = rows[-1]["monotonic_s"] - rows[0]["monotonic_s"]
        result.update(status="CAPTURE_COMPLETE", sample_count=len(rows), duration_s=elapsed,
            effective_poll_rate_hz=(len(rows)-1)/elapsed, min_w=min(r["power_w"] for r in rows),
            max_w=max(r["power_w"] for r in rows), median_w=statistics.median(r["power_w"] for r in rows),
            repeated_adjacent_values=sum(a["power_w"]==b["power_w"] for a,b in zip(rows,rows[1:])),
            independence="polls are not independent sensor conversions; update_interval remains unmodified")
    except Exception as exc:
        result.update(reason=type(exc).__name__ + ": " + str(exc), sample_count=len(rows))
    result.update(csv_sha256=sha(csv_path), source_sha256=sha(__file__))
    (output / "capture.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


def integrate_window(samples, start, end):
    """Trapezoidal W*s on a fully bracketed same-clock window; never extrapolate."""
    if not math.isfinite(start) or not math.isfinite(end) or end <= start or len(samples) < 2:
        raise ValueError("invalid power integration window")
    pairs = [(float(r["monotonic_s"]), float(r["power_w"])) for r in samples]
    if any(not math.isfinite(t) or not math.isfinite(p) or p < 0 for t,p in pairs) or any(b[0]<=a[0] for a,b in zip(pairs,pairs[1:])):
        raise ValueError("invalid samples")
    if pairs[0][0] > start or pairs[-1][0] < end:
        raise ValueError("power samples do not bracket completed-inference window")
    energy = 0.
    for (t0,p0),(t1,p1) in zip(pairs,pairs[1:]):
        left, right = max(t0,start), min(t1,end)
        if right <= left: continue
        a = p0+(p1-p0)*(left-t0)/(t1-t0); b = p0+(p1-p0)*(right-t0)/(t1-t0)
        energy += (a+b)*.5*(right-left)
    return dict(energy_j=energy, mean_power_w=energy/(end-start), duration_s=end-start)


def paired_energy(idle_samples, active_samples, *, idle_window, active_window, completed_inferences):
    """Numerical helper only: caller must bind run receipt, sensor, boot and clock."""
    if type(completed_inferences) is not int or completed_inferences <= 0:
        raise ValueError("actual positive completed inference count required")
    idle = integrate_window(idle_samples, *idle_window); active = integrate_window(active_samples, *active_window)
    dynamic = active["energy_j"] - idle["mean_power_w"]*active["duration_s"]
    return dict(idle_power_w=idle["mean_power_w"], active_power_w=active["mean_power_w"],
        total_energy_mj_per_image=1000*active["energy_j"]/completed_inferences,
        dynamic_energy_mj_per_image=1000*dynamic/completed_inferences,
        dynamic_status="NONPOSITIVE_OR_NOISE_LIMITED" if dynamic <= 0 else "POSITIVE_NOT_YET_REPEATABILITY_VERIFIED",
        claimable=False, binding_status="CALLER_MUST_VERIFY_SAME_SENSOR_BOOT_CLOCK_AND_RUNTIME_RECEIPT")


IIO_OF_SUFFIX = "/axi/i2c@ff030000/i2c-mux@74/i2c@3/ina226@40"
IIO_SCALES = {"in_power2_scale": 12.5, "in_voltage1_scale": 1.25, "in_current3_scale": .5}
POWER_SCOPE = "whole_board_12v_input"


def validate_iio_sensor(path):
    """Match the live identity read on 2026-09-28; no register/config writes."""
    path = Path(path); of = path / "of_node"
    if (path / "name").read_text().strip() != "ina226": raise ValueError("INA226 name mismatch")
    if not str(of.resolve()).replace("\\", "/").endswith(IIO_OF_SUFFIX):
        raise ValueError("sensor not on verified U183 I2C mux channel3 address0x40")
    attributes = {name: (of/name).read_bytes() for name in ("compatible", "reg", "shunt-resistor")}
    if attributes != {"compatible": b"ti,ina226\0", "reg": b"\0\0\0\x40", "shunt-resistor": b"\0\0\x13\x88"}:
        raise ValueError("sensor OF identity/shunt differs from verified 5000uOhm")
    scales = {name: float((path/name).read_text()) for name in IIO_SCALES}
    if scales != IIO_SCALES: raise ValueError("INA226 raw/scale contract changed")
    frequency = float((path/"in_sampling_frequency").read_text())
    if not math.isfinite(frequency) or frequency <= 0: raise ValueError("sensor sampling frequency unavailable")
    optional = {}
    for name in ("in_shunt_resistor", "in_oversampling_ratio", "in_voltage0_integration_time", "in_voltage1_integration_time", "in_allow_async_readout", "current_timestamp_clock"):
        if (path/name).is_file(): optional[name] = (path/name).read_text().strip()
    return dict(path=str(path), realpath=str(path.resolve()), of_node=str(of.resolve()),
        of_values_hex={k:v.hex() for k,v in attributes.items()}, scales=scales,
        sampling_frequency_hz=frequency, configuration=optional,
        scope=POWER_SCOPE, sensor="INA226", rail="VCC12", reference_designator="U183",
        unit_definition="power raw*scale=mW; voltage raw*scale=mV; current raw*scale=mA",
        source_url="https://raw.githubusercontent.com/torvalds/linux/v5.15/drivers/iio/adc/ina2xx-adc.c",
        aliases_excluded=["iio_hwmon mirror", "IRPS5401 auxiliary rails"], sensor_configuration_modified=False)


def iio_sample(path):
    path = Path(path)
    scales = {name: float((path/name).read_text()) for name in IIO_SCALES}
    if scales != IIO_SCALES: raise ValueError("IIO scale changed during capture")
    power_raw = int((path/"in_power2_raw").read_text()); voltage_raw = int((path/"in_voltage1_raw").read_text())
    current_raw = int((path/"in_current3_raw").read_text())
    result = dict(power_raw=power_raw, voltage_raw=voltage_raw, current_raw=current_raw,
        power_w=power_raw*scales["in_power2_scale"] / 1000,
        voltage_v=voltage_raw*scales["in_voltage1_scale"] / 1000,
        current_a=current_raw*scales["in_current3_scale"] / 1000)
    if result["power_w"] < 0 or result["current_a"] < 0 or not 9 <= result["voltage_v"] <= 15:
        raise ValueError("INA226 reading implausible")
    return result


def summarize_iio_windows(samples, idle_window, calls, sensor_frequency_hz):
    if len(calls) != 11 or [c["index"] for c in calls] != list(range(11)):
        raise ValueError("warmup0 plus measured calls1..10 required")
    idle = integrate_window(samples, *idle_window)
    computed = []
    for call in calls:
        start, end = call["start_monotonic_s"], call["end_monotonic_s"]
        inside = sum(start <= row["monotonic_s"] <= end for row in samples)
        if inside < 2 or end-start < 2/sensor_frequency_hz:
            raise ValueError("call too short for resolved sensor integration")
        measured = integrate_window(samples, start, end)
        computed.append(dict(**call, warmup=call["index"]==0, sample_count_inside=inside,
            total_energy_mj=1000*measured["energy_j"], duration_s=end-start,
            dynamic_energy_mj=1000*(measured["energy_j"]-idle["mean_power_w"]*(end-start)),
            mean_power_w=measured["mean_power_w"]))
    measured = computed[1:]; active_seconds=sum(c["duration_s"] for c in measured)
    total_mj=sum(c["total_energy_mj"] for c in measured)
    return dict(idle=idle, calls=computed, measured_inferences=10,
        active_duration_s=active_seconds, idle_power_w=idle["mean_power_w"],
        power_w=total_mj/1000/active_seconds,
        energy_per_inference_mj=total_mj/10,
        dynamic_energy_per_inference_mj=sum(c["dynamic_energy_mj"] for c in measured)/10,
        idle_subtracted=False,
        energy_scope="integral of board input power over each board GraphSession.infer call; excludes host/network/decode",
        repeatability="ten calls in one session; no multi-session repeatability claim")


class IioPowerMonitor:
    """Optional read-only sampler; measurement failures never control the FPGA."""
    def __init__(self, output, binding, sensor_path="/sys/bus/iio/devices/iio:device1", sample_rate_hz=10.):
        self.output, self.binding, self.sensor_path = Path(output), dict(binding), Path(sensor_path)
        self.rate = sample_rate_hz; self.samples=[]; self.calls=[]; self.error=None; self.sensor=None
        self.idle_window=None; self.current=None; self.finished=None
        self.stop=threading.Event(); self.ready=threading.Event(); self.thread=None

    def _failure(self, exc):
        self.error = self.error or (type(exc).__name__ + ": " + str(exc))

    def _loop(self):
        fields=["monotonic_s","read_duration_s","power_raw","voltage_raw","current_raw","power_w","voltage_v","current_a"]
        try:
            with (self.output/"power_samples.csv").open("x", newline="", encoding="utf-8") as stream:
                writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();due=time.monotonic()
                while not self.stop.is_set():
                    before=time.monotonic(); row=iio_sample(self.sensor_path);after=time.monotonic()
                    row.update(monotonic_s=(before+after)/2,read_duration_s=after-before)
                    self.samples.append(row);writer.writerow(row);stream.flush();self.ready.set()
                    due += 1/self.rate;self.stop.wait(max(0,due-time.monotonic()))
        except Exception as exc:
            self._failure(exc);self.ready.set()

    def start(self, idle_seconds=5.):
        try:
            if self.thread is not None or self.output.exists(): raise ValueError("power capture cannot be restarted/overwritten")
            if not math.isfinite(self.rate) or self.rate <= 0 or idle_seconds != 5:
                raise ValueError("positive polling rate and frozen5s idle required")
            for key in ("deployment_binding_sha256","package_manifest_sha256","board_manifest_sha256","graph_sha256","benchmark_manifest_sha256"):
                if not _digest(self.binding.get(key)): raise ValueError("missing power run binding: "+key)
            self.sensor=validate_iio_sensor(self.sensor_path)
            self.boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            self.output.mkdir(parents=True,exist_ok=False)
            self.thread=threading.Thread(target=self._loop,daemon=True,name="read_only_u183_power")
            self.thread.start()
            if not self.ready.wait(10) or self.error: raise RuntimeError(self.error or "sensor first sample timed out")
            begin=time.monotonic();time.sleep(idle_seconds);self.idle_window=[begin,time.monotonic()]
            return dict(status="SAMPLING",scope=POWER_SCOPE)
        except Exception as exc:
            self._failure(exc);self.close();return dict(status="NOT_MEASURED",reason=self.error)

    def begin_inference(self, index):
        if self.error:return
        try:
            if self.current is not None or type(index) is not int or index != len(self.calls) or index not in range(11):raise ValueError("power call index mismatch")
            self.current=dict(index=index,start_monotonic_s=time.monotonic())
        except Exception as exc:self._failure(exc)

    def end_inference(self,index):
        if self.error:return
        try:
            if self.current is None or self.current["index"]!=index:raise ValueError("power call end mismatch")
            self.current["end_monotonic_s"]=time.monotonic();self.calls.append(self.current);self.current=None
        except Exception as exc:self._failure(exc)

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=10)
            if self.thread.is_alive():self._failure(RuntimeError("power sensor read did not terminate"))

    def finish(self):
        if self.finished is not None:return self.finished
        if not self.error and self.calls:
            deadline=time.monotonic()+max(2.,2/self.rate)
            while (not self.samples or self.samples[-1]["monotonic_s"]<=self.calls[-1]["end_monotonic_s"]) and time.monotonic()<deadline:
                time.sleep(min(.05,1/self.rate))
        self.close()
        result=dict(schema_version="zcu104-iio-power-v1",status="NOT_MEASURED",scope=POWER_SCOPE,
            binding=self.binding,sensor=self.sensor,source_sha256=sha(__file__),
            clock="CLOCK_MONOTONIC",boot_id=getattr(self,"boot_id",None),sample_rate_hz=self.rate,
            idle_window=self.idle_window,raw_calls=self.calls,test_evaluated=False,sensor_configuration_modified=False)
        try:
            if self.error:raise RuntimeError(self.error)
            if validate_iio_sensor(self.sensor_path)!=self.sensor:raise ValueError("sensor identity/config changed")
            summary=summarize_iio_windows(self.samples,self.idle_window,self.calls,self.sensor["sampling_frequency_hz"])
            result.update(status="MEASURED",summary=summary,sample_count=len(self.samples),
                sample_power_min_w=min(r["power_w"] for r in self.samples),sample_power_max_w=max(r["power_w"] for r in self.samples),
                effective_poll_rate_hz=(len(self.samples)-1)/(self.samples[-1]["monotonic_s"]-self.samples[0]["monotonic_s"]),
                effective_independent_samples="NOT_CLAIMED",polling_can_repeat_sensor_values=True)
        except Exception as exc:result["reason"]=str(exc)
        try:
            self.output.mkdir(parents=True,exist_ok=True)
            if (self.output/"power_samples.csv").is_file():result["samples_sha256"]=sha(self.output/"power_samples.csv")
            with (self.output/"power.json").open("x",encoding="utf-8") as stream:json.dump(result,stream,indent=2,allow_nan=False)
        except Exception as exc:
            result.update(status="NOT_MEASURED",reason="power result persistence failed: "+str(exc))
        self.finished=result;return result


def compare_power_summaries(expected, recorded, path="summary"):
    """Allow at most eight binary64 ULPs for derived power/energy only.

    Counts, flags, time boundaries, schema and integer inference stay exact.
    Raw CSV and source hashes are verified independently before this comparison.
    """
    derived = {"energy_j", "mean_power_w", "total_energy_mj", "dynamic_energy_mj",
               "idle_power_w", "power_w", "energy_per_inference_mj",
               "dynamic_energy_per_inference_mj"}
    differences = []
    if isinstance(expected, dict):
        if not isinstance(recorded, dict) or expected.keys() != recorded.keys():
            raise ValueError('board power summary differs from raw integration: '+path)
        for key in expected:
            differences += compare_power_summaries(expected[key], recorded[key], path+'/'+key)
    elif isinstance(expected, list):
        if not isinstance(recorded, list) or len(expected) != len(recorded):
            raise ValueError('board power summary differs from raw integration: '+path)
        for i, (left, right) in enumerate(zip(expected, recorded)):
            differences += compare_power_summaries(left, right, path+'/'+str(i))
    elif type(expected) is not type(recorded):
        raise ValueError('board power summary differs from raw integration: '+path)
    elif isinstance(expected, float):
        if not math.isfinite(expected) or not math.isfinite(recorded):
            raise ValueError('board power summary differs from raw integration: '+path)
        if expected != recorded:
            bound = 8 * max(math.ulp(expected), math.ulp(recorded)) if path.rsplit('/',1)[-1] in derived else 0
            if abs(expected-recorded) > bound:
                raise ValueError('board power summary differs from raw integration: '+path)
            differences.append(dict(path=path, recomputed=expected, recorded=recorded,
                                    absolute_difference=abs(expected-recorded), max_difference=bound))
    elif expected != recorded:
        raise ValueError('board power summary differs from raw integration: '+path)
    return differences


def verify_iio_evidence(folder, expected_binding, source_sha256, rpc_rows):
    """Recompute actual returned CSV integrals; absent power remains optional."""
    folder=Path(folder); path=folder/"power.json"
    if not path.is_file():return dict(status="NOT_MEASURED",reason="no returned board sensor capture")
    result=read(path)
    if result.get("status")=="NOT_MEASURED":return result
    if (result.get("schema_version")!="zcu104-iio-power-v1" or result.get("status")!="MEASURED" or
            result.get("scope")!=POWER_SCOPE or result.get("clock")!="CLOCK_MONOTONIC" or
            result.get("test_evaluated") is not False or result.get("sensor_configuration_modified") is not False or
            result.get("source_sha256")!=source_sha256 or result.get("binding")!=expected_binding):
        raise ValueError("board power identity/source/scope mismatch")
    if not isinstance(result.get("boot_id"),str) or not result["boot_id"]:
        raise ValueError("board power clock boot identity missing")
    sensor=result["sensor"]
    if (sensor.get("scope")!=POWER_SCOPE or sensor.get("sensor")!="INA226" or sensor.get("reference_designator")!="U183" or
            sensor.get("rail")!="VCC12" or sensor.get("scales")!=IIO_SCALES or
            not sensor.get("of_node", "").replace("\\", "/").endswith(IIO_OF_SUFFIX) or
            sensor.get("of_values_hex")!={"compatible":"74692c696e6132323600","reg":"00000040","shunt-resistor":"00001388"}):
        raise ValueError("unverified returned board power sensor")
    sample_path=folder/"power_samples.csv"
    if sha(sample_path)!=result.get("samples_sha256"):raise ValueError("power CSV digest changed")
    with sample_path.open(newline="",encoding="utf-8") as stream:
        samples=[{k:float(v) for k,v in row.items()} for row in csv.DictReader(stream)]
    for row in samples:
        if (row["power_w"]!=row["power_raw"]*.0125 or row["voltage_v"]!=row["voltage_raw"]*.00125 or
                row["current_a"]!=row["current_raw"]*.0005):
            # Use precisely the recorded raw*scale/1000 operation to avoid
            # comparing two differently rounded floating point expressions.
            if (row["power_w"]!=row["power_raw"]*12.5/1000 or row["voltage_v"]!=row["voltage_raw"]*1.25/1000 or
                    row["current_a"]!=row["current_raw"]*.5/1000):
                raise ValueError("IIO raw/scale/unit reconstruction mismatch")
        if not all(math.isfinite(v) for v in row.values()) or not 9<=row["voltage_v"]<=15 or row["current_a"]<0:
            raise ValueError("nonfinite/implausible power samples")
    if result.get("sample_count")!=len(samples) or result.get("sample_rate_hz")!=10:
        raise ValueError("power count or frozen polling rate mismatch")
    idle=result["idle_window"]
    if idle[1]-idle[0]<5 or idle[1]>result["raw_calls"][0]["start_monotonic_s"]:
        raise ValueError("fixed idle window not before warmup")
    summary=summarize_iio_windows(samples,idle,result["raw_calls"],sensor["sampling_frequency_hz"])
    compare_power_summaries(summary,result["summary"])
    if len(rpc_rows)!=11:raise ValueError("power needs matching11 completed board calls")
    for call,rpc in zip(summary["calls"],rpc_rows):
        if call["index"]!=rpc["call_index"] or call["duration_s"]<rpc["input_graph_capture_seconds"]:
            raise ValueError("power window does not bracket real board inference")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="action", required=True)
    scan = sub.add_parser("inventory"); scan.add_argument("--hwmon", type=Path, required=True)
    sample = sub.add_parser("capture"); sample.add_argument("--manifest", type=Path, required=True)
    sample.add_argument("--output", type=Path, required=True); sample.add_argument("--phase", choices=["idle","active"], required=True)
    sample.add_argument("--duration", type=float, default=60); sample.add_argument("--rate", type=float, default=10)
    args = parser.parse_args()
    value = inventory(args.hwmon) if args.action == "inventory" else capture(args.manifest, args.output, phase=args.phase, duration_s=args.duration, sample_rate_hz=args.rate)
    print(json.dumps(value, indent=2, allow_nan=False))


if __name__ == "__main__": main()
