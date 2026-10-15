"""Board verifier contract fixtures, never substitutes actual FPGA evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.sonar_deploy import board_result as b


def save(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def board(tmp_path):
    graph = dict(outputs={p: "head."+p for p in ("P3", "P4", "P5")}, edges={}, descriptors=[])
    for p, node in graph["outputs"].items():
        graph["edges"][node] = dict(shape=[21, 2, 2], exp=-2)
        graph["descriptors"].append(dict(name=p, channels=21, height=2, width=2, real_bytes=84, padded_bytes=96))
    expected, actual = [], []
    source = dict(layers=[dict(node=node, output_sha256="e"*64) for node in graph["outputs"].values()])
    for i in range(107):
        split = "train" if i < 8 else "valid"
        identity = f"{split}/images/{i:03d}.jpg"
        data = bytes([i]) * 84; digest = hashlib.sha256(data).hexdigest()
        heads = []
        for p in graph["outputs"]:
            folder = tmp_path / "frames" / str(i); folder.mkdir(parents=True, exist_ok=True)
            (folder / (p+".bin")).write_bytes(data)
            (folder / (p+".padded.bin")).write_bytes(data + bytes(12))
            heads.append(dict(level=p, bytes=84, sha256=digest, padding_zero=True))
        expected.append(dict(sample_id=identity, split=split,
                             expected_heads={p: digest for p in graph["outputs"]},
                             expected_layers={row["node"]: "e"*64 for row in source["layers"]}))
        actual.append(dict(index=i, sample_id=identity, split=split, heads=heads, mismatch_count=0,
                           layer_trace=[dict(node=row["node"], sha256="e"*64, mismatch_count=0) for row in source["layers"]] if i<8 else []))
    frames = dict(frames=expected, calibration_first8_sample_ids=[r["sample_id"] for r in expected[:8]])
    save(tmp_path / "frames.json", actual)
    save(tmp_path / "layers.json", [dict(index=i, node=row["node"], mismatch_count=0, output_sha256="e"*64) for i,row in enumerate(source["layers"])])
    save(tmp_path / "completion.json", dict(status="FPGA_GRAPH_PARITY_PASS", real_weights_bound=True, test_evaluated=False,
        nodes=3, board_clock_mhz=200., frame_status=dict(train_frames=8, valid_frames=99, mismatch_count=0, test_evaluated=False)))
    return tmp_path, graph, frames, source, actual


def test_raw_frames_and_cpu_replay_contract(board):
    path, graph, frames, source, actual = board
    assert len(b.validate_actual_frames(path, graph, frames, source)) == 107
    model = b.ReplayBoardModel(graph, actual, frames["frames"], path, SimpleNamespace(loss_config=lambda c: c))
    output = model(torch.zeros(4, 1, 640, 640))["predictions"]
    assert len(output) == 3 and all(t.device.type == "cpu" and t.dtype == torch.float32 for t in output)
    assert output[0].shape == (4, 21, 2, 2)
    assert torch.equal(output[0][:,0,0,0], torch.tensor([2., 2.25, 2.5, 2.75]))
    assert model.loss_config({"protocol": "frozen"}) == {"protocol": "frozen"}


@pytest.mark.parametrize("mutation", ["bytes", "padding", "missing_frame", "reorder", "duplicate", "layer", "test", "shape"])
def test_rejects_corrupt_or_incomplete_actual_evidence(board, mutation):
    path, graph, frames, source, actual = board
    if mutation == "bytes": (path / "frames/8/P3.bin").write_bytes(bytes([3])*84)
    elif mutation == "padding": (path / "frames/8/P3.padded.bin").write_bytes(bytes([8])*84+bytes([1])*12)
    elif mutation == "missing_frame": actual.pop(); save(path/"frames.json", actual)
    elif mutation == "reorder": actual[8],actual[9]=actual[9],actual[8]; save(path/"frames.json", actual)
    elif mutation == "duplicate": actual[9]["sample_id"]=actual[8]["sample_id"]; save(path/"frames.json", actual)
    elif mutation == "layer": actual[0]["layer_trace"].pop(); save(path/"frames.json", actual)
    elif mutation == "test":
        completion=json.loads((path/"completion.json").read_text());completion["test_evaluated"]=True;save(path/"completion.json",completion)
    else: graph["descriptors"][0]["height"]=3
    with pytest.raises(ValueError): b.validate_actual_frames(path, graph, frames, source)


def test_missing_actual_board_results_waits(tmp_path):
    value=b.verify_board_result(tmp_path,"D")
    assert value["status"]=="WAITING_FOR_LOCAL_BOARD_RESULT" and not value["allowed"]
    assert not (tmp_path/"D/seed_42/board.json").exists()


def test_replay_cannot_consume_non_cpu_or_wrong_input_metadata(board):
    path, graph, frames, _, actual = board
    model=b.ReplayBoardModel(graph,actual,frames["frames"],path,None)
    with pytest.raises(ValueError,match="CPU decode"):
        model(torch.zeros(1,3,640,640))
    assert model.cursor==0
    model.cursor=98
    with pytest.raises(ValueError,match="too many"):
        model(torch.zeros(2,1,640,640))


@pytest.fixture
def timing(tmp_path):
    save(tmp_path/"board_manifest.json",{"identity":"fixture"})
    row=dict(graph_seconds=.8,end_to_end_seconds=1.,board_clock_mhz=200.)
    rows=[dict(**row,call_index=i,warmup=i==0) for i in range(11)]
    data=dict(schema_version="real-int8-e2e-v1",status="PASS",precision="INT8",quantization_policy="W8A8",
        test_evaluated=False,batch_size=1,input_size=640,warmup=1,repetitions=10,weights_resident=True,
        timing_scope="host_preprocess_rpc_board_graph_receive_cpu_decode",graph_sha256="a"*64,
        board_manifest_sha256=b.gate.sha256(tmp_path/"board_manifest.json"),rows=rows,samples=rows[1:],
        timing=dict(graph_seconds=[.8]*10,end_to_end_seconds=[1.]*10,graph_median_ms=800.,graph_p95_ms=800.,
                    end_to_end_median_ms=1000.,end_to_end_p95_ms=1000.))
    save(tmp_path/"e2e.json",data)
    return tmp_path,data


def test_e2e_uses_actual_samples(timing):
    path,_=timing
    values,clock,_=b._timing_from_e2e(path/"e2e.json",dict(implemented_clock_mhz=200.),"a"*64,dict(board_clock_mhz=200.))
    assert values["end_to_end_median_ms"]==1000 and clock==200.


@pytest.mark.parametrize("mutation",["missing","clock","nan_clock","summary","order","warmup_order","test","scope"])
def test_e2e_rejects_unbound_measurements(timing,mutation):
    path,data=timing
    if mutation=="missing": data["samples"].pop()
    elif mutation=="clock": data["samples"][0]["board_clock_mhz"]=199.
    elif mutation=="nan_clock": data["samples"][0]["board_clock_mhz"]=float("nan")
    elif mutation=="summary": data["timing"]["graph_median_ms"]=1.
    elif mutation=="order": data["samples"][0]["call_index"]=2
    elif mutation=="warmup_order": data["rows"][0]["call_index"]=1
    elif mutation=="test": data["test_evaluated"]=True
    else: data["timing_scope"]="estimated sum"
    save(path/"e2e.json",data)
    with pytest.raises(ValueError):
        b._timing_from_e2e(path/"e2e.json",dict(implemented_clock_mhz=200.),"a"*64,dict(board_clock_mhz=200.))


@pytest.fixture
def rpc_session(tmp_path):
    identity=dict(graph_sha256="a"*64,package_manifest_sha256="b"*64,board_manifest_sha256="c"*64)
    board_rows=[dict(call_index=i,graph_seconds=.8,input_graph_capture_seconds=.9,
                     board_clock_mhz=200.,test_evaluated=False) for i in range(11)]
    host_rows=[dict(call_index=i,graph_seconds=.8,input_graph_capture_seconds=.9,
                    board_clock_mhz=200.,end_to_end_seconds=1.,warmup=i==0) for i in range(11)]
    rpc=dict(status="PASS",successful_calls=11,weights_resident=True,test_evaluated=False,
             rows=board_rows,**identity)
    e2e=dict(rows=host_rows,**identity)
    path=tmp_path/"e2e_rpc_session.json";save(path,rpc)
    return path,rpc,e2e


def test_rpc_completion_matches_all_eleven_host_responses(rpc_session):
    path,rpc,e2e=rpc_session
    assert b._validate_rpc_session(path,e2e)==rpc


@pytest.mark.parametrize("mutation",["quarantined","count","missing_row","identity","order","graph_time","capture_time","clock","nan","test"])
def test_rpc_rejects_failed_unbound_or_mismatched_session(rpc_session,mutation):
    path,rpc,e2e=rpc_session
    if mutation=="quarantined":rpc["status"]="QUARANTINED"
    elif mutation=="count":rpc["successful_calls"]=10
    elif mutation=="missing_row":rpc["rows"].pop()
    elif mutation=="identity":rpc["package_manifest_sha256"]="d"*64
    elif mutation=="order":rpc["rows"][0]["call_index"]=1
    elif mutation=="graph_time":rpc["rows"][1]["graph_seconds"]=.7
    elif mutation=="capture_time":rpc["rows"][0]["input_graph_capture_seconds"]=.7
    elif mutation=="clock":rpc["rows"][1]["board_clock_mhz"]=199.
    elif mutation=="nan":rpc["rows"][1]["graph_seconds"]=float("nan")
    else:rpc["rows"][1]["test_evaluated"]=True
    save(path,rpc)
    with pytest.raises(ValueError):b._validate_rpc_session(path,e2e)


def test_host_pass_without_board_rpc_completion_cannot_publish(tmp_path):
    folder=tmp_path/"D/seed_42/hardware/board_evidence"
    save(folder/"completion.json",dict(status="FPGA_GRAPH_PARITY_PASS"))
    save(folder/"e2e.json",dict(status="PASS"))
    report=b.verify_board_result(tmp_path,"D")
    assert report["status"]=="WAITING_FOR_LOCAL_BOARD_RESULT"
    assert "buffer release" in report["reason"]
    assert not (tmp_path/"D/seed_42/board.json").exists()


@pytest.mark.parametrize("version",["serial_ddr_graph_v2_requant","serial_ddr_graph_v2_requant_timing02"])
def test_known_executor_schedule_must_match_bound_source(version):
    implementation=dict(executor_version=version,requested_clock_mhz=200,arena_capacity_bytes=16777216)
    if version.endswith("timing02"):implementation["requant_pipeline"]="off"
    b._validate_executor(dict(executor=implementation),dict(implementation=implementation.copy()))
    changed=dict(implementation,requested_clock_mhz=100)
    with pytest.raises(ValueError,match="bound build"):
        b._validate_executor(dict(executor=changed),dict(implementation=implementation))


def test_timing02_cannot_claim_unbound_or_pipelined_requant():
    for implementation in ({},dict(executor_version="unverified-v3"),
                           dict(executor_version="serial_ddr_graph_v2_requant_timing02",requant_pipeline="II=1")):
        with pytest.raises(ValueError):
            b._validate_executor(dict(executor=implementation),dict(implementation=implementation))
