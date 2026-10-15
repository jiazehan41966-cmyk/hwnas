"""Synthetic filesystem/arithmetic fixtures, never hardware measurement."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from tools.sonar_deploy import power


@pytest.fixture
def sensor(tmp_path):
    for key,value in {"name":"ina226", "shunt_resistor":"5000", "in1_input":"12000", "power1_input":"12000000", "curr1_input":"1000"}.items():
        (tmp_path/key).write_text(value)
    (tmp_path/"device").mkdir(); proof=tmp_path/"proof.json";proof.write_text('{"fixture":true}')
    return dict(schema_version="zcu104-power-sensor-v1", status="VERIFIED", board="ZCU104", interface="hwmon",
        power_unit="microwatt", hwmon_path=str(tmp_path), resolved_hwmon_path=str(tmp_path.resolve()),
        sensor_name="ina226", power_attribute="power1_input", power_scope="board_input_total_12v",
        reference_designator="U183", rail="VCC12", rail_mapping_evidence=[dict(path=str(proof),sha256=power.sha(proof))],
        verified_links={"device":str((tmp_path/"device").resolve())},test_evaluated=False,
        deployment_binding_sha256="a"*64,package_manifest_sha256="b"*64,board_manifest_sha256="c"*64)


def test_hwmon_units_readonly(sensor):
    root=Path(sensor["hwmon_path"]); before={p.name:p.read_bytes() for p in root.iterdir() if p.is_file()}
    assert power.validate_sensor(sensor)["attributes"]["shunt_resistor"]=="5000"
    assert power.power_sample(sensor)==12.
    assert before=={p.name:p.read_bytes() for p in root.iterdir() if p.is_file()}


@pytest.mark.parametrize("key,value",[("status","UNVERIFIED"),("interface","iio"),("power_unit","milliwatt"),
    ("sensor_name","irps5401"),("rail_mapping_evidence",[]),("verified_links",{}),("power_attribute","power1_input+power2_input"),
    ("test_evaluated",True)])
def test_sensor_rejects_unverified_total_or_duplicate_source(sensor,key,value):
    sensor[key]=value
    with pytest.raises(ValueError): power.validate_sensor(sensor)


def test_missing_scope_evidence_not_measured(sensor,tmp_path):
    sensor["rail_mapping_evidence"]=[]; path=tmp_path/"manifest.json";path.write_text(json.dumps(sensor))
    assert power.capture(path,tmp_path/"output",phase="active")["status"]=="NOT_MEASURED"
    assert not (tmp_path/"output").exists()


def test_energy_integrates_matching_window_without_latency_product():
    idle=[dict(monotonic_s=t,power_w=10.) for t in (0,1,2)]
    active=[dict(monotonic_s=t,power_w=20.) for t in (3,4,5)]
    value=power.paired_energy(idle,active,idle_window=(.25,1.75),active_window=(3.25,4.75),completed_inferences=3)
    assert value["total_energy_mj_per_image"]==10000.
    assert value["dynamic_energy_mj_per_image"]==5000.
    assert value["claimable"] is False


def test_energy_never_extrapolates_or_clamps_negative():
    samples=[dict(monotonic_s=0,power_w=2.),dict(monotonic_s=2,power_w=2.)]
    with pytest.raises(ValueError): power.integrate_window(samples,-1,1)
    value=power.paired_energy(samples,[dict(monotonic_s=0,power_w=1.),dict(monotonic_s=2,power_w=1.)],
        idle_window=(0,2),active_window=(0,2),completed_inferences=2)
    assert value["dynamic_energy_mj_per_image"]==-1000 and value["dynamic_status"]=="NONPOSITIVE_OR_NOISE_LIMITED"


@pytest.fixture
def iio(tmp_path,monkeypatch):
    node=tmp_path/"of_node";node.mkdir()
    for name,raw in {"compatible":b"ti,ina226\0","reg":b"\0\0\0\x40","shunt-resistor":b"\0\0\x13\x88"}.items():
        (node/name).write_bytes(raw)
    for name,value in {"name":"ina226", "in_sampling_frequency":"100", "in_power2_scale":"12.5", "in_voltage1_scale":"1.25",
        "in_current3_scale":".5", "in_power2_raw":"1193", "in_voltage1_raw":"9576", "in_current3_raw":"2491"}.items():
        (tmp_path/name).write_text(value)
    original=Path.resolve
    monkeypatch.setattr(Path,"resolve",lambda p,*a,**k:Path("/sys/firmware/devicetree/base"+power.IIO_OF_SUFFIX) if p==node else original(p,*a,**k))
    return tmp_path


def test_verified_iio_units_and_identity(iio):
    assert power.validate_iio_sensor(iio)["scales"]==power.IIO_SCALES
    row=power.iio_sample(iio)
    assert row["power_w"]==14.9125 and row["voltage_v"]==11.97 and row["current_a"]==1.2455


@pytest.mark.parametrize("name,value",[("in_power2_scale","12500"),("in_voltage1_scale",".00125"),("name","irps5401"),("in_sampling_frequency","nan")])
def test_iio_changed_identity_or_units_rejected(iio,name,value):
    (iio/name).write_text(value)
    with pytest.raises(ValueError):power.validate_iio_sensor(iio)


def test_iio_bad_shunt_rejected(iio):
    (iio/"of_node/shunt-resistor").write_bytes(b"\0\0\x27\x10")
    with pytest.raises(ValueError,match="shunt"):power.validate_iio_sensor(iio)


def test_power_monitor_optional_failure_has_no_fpga_side_effect(tmp_path):
    monitor=power.IioPowerMonitor(tmp_path/"missing",binding={})
    assert monitor.start()["status"]=="NOT_MEASURED"
    monitor.begin_inference(0);monitor.end_inference(0)
    result=monitor.finish()
    assert result["status"]=="NOT_MEASURED" and "summary" not in result
    assert monitor.finish()==result


def test_iio_integration_excludes_warmup_and_idle():
    rows=[dict(monotonic_s=i/10,power_w=12.) for i in range(181)]
    calls=[dict(index=i,start_monotonic_s=6.+i,end_monotonic_s=6.8+i) for i in range(11)]
    value=power.summarize_iio_windows(rows,[0.,5.],calls,100.)
    assert value["measured_inferences"]==10
    assert value["energy_per_inference_mj"]==pytest.approx(9600)
    assert value["power_w"]==pytest.approx(12)
    assert value["idle_subtracted"] is False
    assert value["dynamic_energy_per_inference_mj"]==pytest.approx(0,abs=1e-9)
    with pytest.raises(ValueError):power.summarize_iio_windows(rows,[0.,5.],calls[:10],100.)


def test_absent_power_remains_not_measured(tmp_path):
    assert power.verify_iio_evidence(tmp_path,{},"a"*64,[])["status"]=="NOT_MEASURED"


def test_summary_accepts_only_tiny_derived_roundoff():
    import math
    actual={'power_w':10.435630546916567,'dynamic_energy_per_inference_mj':29922.185719125897,'count':10,'flag':False}
    saved=dict(actual,power_w=10.435630546916565,dynamic_energy_per_inference_mj=29922.185719125904)
    assert len(power.compare_power_summaries(actual,saved))==2
    for bad in [dict(saved,power_w=10.43563054692),dict(saved,count=11),dict(saved,flag=0),dict(saved,power_w=float('nan'))]:
        with pytest.raises(ValueError):power.compare_power_summaries(actual,bad)
    with pytest.raises(ValueError):power.compare_power_summaries({'start_monotonic_s':1.},{'start_monotonic_s':math.nextafter(1.,2.)})


def test_returned_power_recomputed_from_raw_csv(iio,tmp_path):
    import csv
    folder=tmp_path/"capture";folder.mkdir()
    samples=[dict(monotonic_s=i/10,read_duration_s=.001,power_raw=960,voltage_raw=9600,current_raw=2000,
                  power_w=12.,voltage_v=12.,current_a=1.) for i in range(181)]
    calls=[dict(index=i,start_monotonic_s=6.+i,end_monotonic_s=6.8+i) for i in range(11)]
    with (folder/"power_samples.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(samples[0]));writer.writeheader();writer.writerows(samples)
    binding={"graph_sha256":"b"*64};rpc=[dict(call_index=i,input_graph_capture_seconds=.79) for i in range(11)]
    value=dict(schema_version="zcu104-iio-power-v1",status="MEASURED",scope=power.POWER_SCOPE,
        clock="CLOCK_MONOTONIC",test_evaluated=False,sensor_configuration_modified=False,
        source_sha256="a"*64,binding=binding,boot_id="fixture_boot",sensor=power.validate_iio_sensor(iio),
        samples_sha256=power.sha(folder/"power_samples.csv"),sample_count=181,sample_rate_hz=10,
        idle_window=[0,5],raw_calls=calls,summary=power.summarize_iio_windows(samples,[0,5],calls,100))
    (folder/"power.json").write_text(json.dumps(value))
    assert power.verify_iio_evidence(folder,binding,"a"*64,rpc)["status"]=="MEASURED"
    value["summary"]["power_w"]+=1;(folder/"power.json").write_text(json.dumps(value))
    with pytest.raises(ValueError,match="integration"):
        power.verify_iio_evidence(folder,binding,"a"*64,rpc)


def test_monitor_lifecycle_with_readonly_synthetic_sysfs(iio,tmp_path,monkeypatch):
    import time
    original=Path.read_text
    monkeypatch.setattr(Path,"read_text",lambda p,*a,**k:"test-boot-id" if p.as_posix()=="/proc/sys/kernel/random/boot_id" else original(p,*a,**k))
    binding={k:"a"*64 for k in ("deployment_binding_sha256","package_manifest_sha256","board_manifest_sha256","graph_sha256","benchmark_manifest_sha256")}
    monitor=power.IioPowerMonitor(tmp_path/"power_capture",binding,sensor_path=iio)
    assert monitor.start()["status"]=="SAMPLING"
    try:
        for index in range(11):
            monitor.begin_inference(index);time.sleep(.31);monitor.end_inference(index)
    finally:
        result=monitor.finish()
    assert result["status"]=="MEASURED"
    assert result["summary"]["measured_inferences"]==10
    rpc=[dict(call_index=i,input_graph_capture_seconds=.30) for i in range(11)]
    assert power.verify_iio_evidence(tmp_path/"power_capture",binding,power.sha(power.__file__),rpc)["status"]=="MEASURED"
