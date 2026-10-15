"""Small software fixtures; these are not measured nine-model diagnostics."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.sonar_deploy import diagnostic_layers as diagnostic


@pytest.fixture
def traces():
    graph=dict(nodes=[dict(id="first",op="requant"),dict(id="second",op="conv")],
               edges={name:dict(shape=[1,1,3],exp=-2) for name in ("first","second")})
    integer={"first":np.array([[[[0,1,-2]]]],dtype=np.int8),
             "second":torch.tensor([[[[1,-1,127]]]],dtype=torch.int8)}
    floating={name:diagnostic._numpy(value).astype(np.float32)*.25 for name,value in integer.items()}
    return graph,floating,integer


def test_exact_dequantized_comparison_and_first_difference(traces):
    graph,floating,integer=traces
    rows=diagnostic.compare_layers(graph,floating,integer)
    assert diagnostic.summarize(rows)["first_difference"] is None
    floating["first"][0,0,0,1]=np.nextafter(np.float32(.25),np.float32(1.))
    floating["second"][0,0,0,2]-=.5
    rows=diagnostic.compare_layers(graph,floating,integer)
    summary=diagnostic.summarize(rows)
    assert summary["first_difference"]["node"]=="first"
    assert summary["mismatch_count"]==2 and summary["max_abs_error"]==.5
    assert rows[0]["max_abs_error"]>0 and rows[0]["max_abs_error"]<1e-6
    assert summary["compared_values"]==6


@pytest.mark.parametrize("mutation",["missing","shape","float_integer","nonfinite","empty"])
def test_invalid_trace_never_produces_diagnostic(traces,mutation):
    graph,floating,integer=traces
    if mutation=="missing":floating.pop("second")
    elif mutation=="shape":floating["first"]=floating["first"][0]
    elif mutation=="float_integer":integer["first"]=integer["first"].astype(np.float32)
    elif mutation=="nonfinite":floating["first"][0,0,0,0]=np.nan
    else:graph["nodes"]=[]
    with pytest.raises(ValueError):diagnostic.compare_layers(graph,floating,integer)


@pytest.mark.parametrize("device",["cpu","cuda"])
def test_original_folding_never_falls_back_to_cpu(monkeypatch,tmp_path,device):
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    monkeypatch.setattr(diagnostic.gate,"load_binding",lambda root:pytest.fail("no data should be opened"))
    with pytest.raises(RuntimeError,match="CUDA_ENVIRONMENT_NOT_READY"):
        diagnostic.run(tmp_path,tmp_path/"out",device=device)
    assert not (tmp_path/"out").exists()


def test_nine_run_dispatch_consumes_only_first_frozen_train_image(monkeypatch,tmp_path):
    binding_path=tmp_path/"DEPLOYMENT_BINDING.json";binding_path.write_text("{}")
    calibration_path=tmp_path/"calibration.json";calibration_path.write_text("{}")
    entry=dict(image="train/images/first.png")
    rows=[dict(network=n,seed=s,calibration_sha256=diagnostic.gate.sha256(calibration_path))
          for n in diagnostic.gate.NETWORKS for s in diagnostic.gate.SEEDS]
    binding=dict(runs=rows,source_files=dict(calibration_manifest=dict(path=str(calibration_path))))
    monkeypatch.setattr(torch.cuda,"is_available",lambda:True)
    monkeypatch.setattr(diagnostic.gate,"load_binding",lambda root:binding)
    real_read=diagnostic.gate.read_json
    monkeypatch.setattr(diagnostic.gate,"read_json",lambda path:dict(entries=[entry]) if Path(path)==calibration_path else real_read(path))
    monkeypatch.setattr(diagnostic.gate,"load_result",lambda root,row:row)
    closed=[]
    def prefix(calibration):
        try:
            yield 0,entry,torch.zeros(1,1,640,640)
            pytest.fail("second train image must not be consumed")
        finally:closed.append(True)
    monkeypatch.setattr(diagnostic.parity,"_train_prefix",prefix)
    calls=[]
    def run_one(root,output,row,result,entry,image,device,binding_hash,sources):
        calls.append((row["network"],row["seed"]))
        report=dict(network=row["network"],seed=row["seed"],first_difference=None,
                    mismatch_count=0,max_abs_error=0.,mean_abs_error=0.,
                    graph_sha256="a"*64,checkpoint_sha256="b"*64)
        diagnostic.gate.atomic_json(output/row["network"]/f"seed_{row['seed']}"/"HISTORICAL_VS_INTEGER.json",report)
        return report
    monkeypatch.setattr(diagnostic,"_diagnose_one",run_one)
    out=tmp_path/"out";report=diagnostic.run(tmp_path,out)
    assert closed==[True] and len(calls)==9 and len(set(calls))==9
    assert report["all_nine_complete"] and report["completed"]==9
    assert report["test_evaluated"] is False
    assert (out/"HISTORICAL_VS_INTEGER_SUMMARY.csv").is_file()


def test_completed_diagnostic_reuse_cannot_ignore_bound_source_change(monkeypatch,tmp_path):
    # The reuse branch checks provenance before any CUDA tensor evaluation.
    report=tmp_path/"out/D/seed_42/HISTORICAL_VS_INTEGER.json"
    diagnostic.gate.atomic_json(report,dict(status="COMPLETE",schema_version=diagnostic.SCHEMA))
    package=tmp_path/"package";package.mkdir();(package/"graph.json").write_text("{}")
    result={key:"a"*64 for key in diagnostic.gate.IDENTITY}
    result.update(network="D",seed=42,package_manifest_path=str(package/"manifest.json"),package_manifest_sha256="b"*64)
    with pytest.raises(ValueError,match="identity/source/CSV changed"):
        diagnostic._diagnose_one(tmp_path,tmp_path/"out",dict(network="D",seed=42),result,
                                 {},None,"cuda","c"*64,{"new.py":"d"*64})
