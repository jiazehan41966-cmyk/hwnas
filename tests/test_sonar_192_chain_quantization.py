"""Numeric behavior tests; tiny fixtures never become research/board evidence."""
from copy import deepcopy
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from tools.sonar_192_chain import quantization as q
from tools.sonar_192_chain.contract import digest


@pytest.mark.parametrize("groups", [1, 2, 4, 8])
def test_grouped_integer_conv_has_exact_wrap_semantics(groups):
    rng = np.random.default_rng(611)
    x = rng.integers(-128,128,(8,5,5),dtype=np.int8)
    weight = rng.integers(-128,128,(8,8//groups,3,3),dtype=np.int8)
    bias = np.array([2**31-1,-2**31,0,1,-1,77,999,-999],dtype=np.int32)
    actual = q.grouped_conv_acc(x,weight,bias,stride=1,padding=1,groups=groups)
    # Independent FP64 convolution, then exact modulo wrap.
    expected = F.conv2d(torch.from_numpy(x).double()[None], torch.from_numpy(weight).double(),
                        torch.from_numpy(bias).double(),padding=1,groups=groups)[0]
    expected = ((expected+2**31) % 2**32 - 2**31).numpy().astype(np.int64)
    assert np.array_equal(actual,expected)
    assert actual.dtype == np.int64


@pytest.mark.parametrize("input_exp,output_exp", [(-8,-9),(-3,-4),(0,0),(4,3),(-24,-24),(16,16)])
def test_silu_table_entire_int8_domain_matches_independent_formula(input_exp,output_exp):
    table = q.silu_lut(input_exp,output_exp)
    x = torch.arange(-128,128,dtype=torch.float64)*(2.0**input_exp)
    reference = (F.silu(x)/(2.0**output_exp)).round().clamp(-128,127).numpy().astype(np.int8)
    assert table.shape == (256,)
    assert np.array_equal(table,reference)
    assert table[128] == 0


def mixed_primitive_fixture():
    rng = np.random.default_rng(413)
    nodes, parameters, luts = [], {}, {}
    shapes, exps = {"image":[4,8,8]}, {"image":-4}
    def node(name,op,inputs,shape,eo,**kwargs):
        row = dict(id=name,op=op,inputs=inputs,input_shapes=[shapes[x] for x in inputs],output_shape=shape,
                   input_exps=[exps[x] for x in inputs],output_exp=eo,**kwargs)
        nodes.append(row); shapes[name]=shape; exps[name]=eo
        return name
    def conv(name,source,co,k=1,groups=1,eo=-3,activation="linear"):
        ci,h,w=shapes[source]
        we=-4
        m,s=q.h2.ratio(exps[source]+we,eo)
        node(name,"conv",[source],[co,h,w],eo,kernel=k,stride=1,padding=k//2,groups=groups,
             activation=activation,weight_exp=we,multiplier=m,shift=s)
        parameters[name]=dict(weight=rng.integers(-16,17,(co,ci//groups,k,k),dtype=np.int8),
                              bias=rng.integers(-32,33,co,dtype=np.int32))
        return name
    conv("dense","image",8,3,activation="relu6")
    node("silu","silu",["dense"],[8,8,8],-4)
    luts["silu"]=q.silu_lut(-3,-4)
    conv("group4","silu",8,3,groups=4)
    node("pool","maxpool",["group4"],[8,4,4],-3)
    node("up","nearest",["pool"],[8,8,8],-3)
    node("sum","residual_add",["dense","up"],[8,8,8],-2)
    node("blur","blur",["sum"],[8,4,4],-2,fixed_kernel=[[1,2,1],[2,4,2],[1,2,1]])
    conv("reg","blur",4,groups=4,eo=-4)
    node("relu","relu",["reg"],[4,4,4],-4)
    conv("cls","blur",2,eo=-3)
    node("concat","concat",["relu","cls"],[6,4,4],-3)
    graph=dict(input_shape=[4,8,8],input_exp=-4,nodes=nodes,outputs={"P2":"concat"})
    graph["numeric_graph_sha256"]=digest(graph)
    return graph,parameters,luts


def test_complete_micrograph_integer_fake_all_node_parity():
    graph,parameters,luts=mixed_primitive_fixture()
    integer=q.IntegerReference(graph,parameters,luts)
    fake=q.FakeReference(graph,parameters).eval()
    images=torch.linspace(-2,2,256,dtype=torch.float32).reshape(1,4,8,8)
    actual=integer.forward(integer.quantize_input(images))
    expected=fake.forward_all(images)
    assert set(actual)==set(expected)
    for name in actual:
        assert np.array_equal(actual[name],expected[name].numpy()),name
    assert actual["concat"].shape==(1,6,4,4)
    assert np.all(actual["relu"]>=0)


def test_negative_and_positive_halfway_requant_uses_ties_even():
    values=np.array([-7,-5,-3,-1,1,3,5,7],dtype=np.int32)
    expected=np.array([-4,-2,-2,0,0,2,2,4],dtype=np.int8)
    assert np.array_equal(q._int_requant(values,0,1),expected)
    assert np.array_equal(q._float_requant(torch.from_numpy(values).double(),0,1).numpy(),expected)


def test_integer_runtime_rejects_float_inputs_and_fake_weight_dtype():
    graph,parameters,luts=mixed_primitive_fixture()
    integer=q.IntegerReference(graph,parameters,luts)
    with pytest.raises(ValueError,match="already-quantized"):
        integer.forward(np.zeros((1,4,8,8),dtype=np.float32))
    with pytest.raises(ValueError,match="INT8/INT16 input/weight"):
        q.grouped_conv_acc(np.zeros((4,2,2),dtype=np.float32),np.zeros((4,1,1,1),dtype=np.int8),
                           np.zeros(4,dtype=np.int32),stride=1,padding=0,groups=4)


@pytest.mark.parametrize("profile",["W4A8","W16A16","bogus"])
def test_unsupported_profiles_fail_before_model_or_calibration_access(profile):
    with pytest.raises(NotImplementedError,match="unsupported 192v2"):
        q.prepare_ptq(None,None,profile=profile)


@pytest.mark.parametrize("weight_bits",[8,16])
def test_int16_grouped_accumulation_and_full_domain_SiLU(weight_bits):
    rng=np.random.default_rng(881)
    x=rng.integers(-32768,32768,(8,5,5),dtype=np.int16)
    lo,hi=q._bounds(weight_bits)
    weights=rng.integers(lo,hi+1,(8,2,3,3),dtype=q._dtype(weight_bits))
    biases=np.full(8,2**31-1,dtype=np.int32)
    actual=q.grouped_conv_acc(x,weights,biases,stride=1,padding=1,groups=4)
    expected=F.conv2d(torch.from_numpy(x).double()[None],torch.from_numpy(weights).double(),
                      torch.from_numpy(biases).double(),padding=1,groups=4)[0]
    expected=((expected+2**31)%2**32-2**31).numpy().astype(np.int64)
    assert np.array_equal(actual,expected)
    for ei,eo in [(-10,-11),(-4,-5),(0,0)]:
        table=q.silu_lut(ei,eo,input_bits=16,output_bits=16)
        formula=(F.silu(torch.arange(-32768,32768,dtype=torch.float64)*(2.0**ei))/(2.0**eo)).round().clamp(-32768,32767)
        assert np.array_equal(table,formula.numpy())
        assert table.dtype==np.int16 and len(table)==65536


@pytest.mark.parametrize("weight_bits",[8,16])
def test_explicit_body8_head16_boundary_group4_relu_concat_parity(weight_bits):
    rng=np.random.default_rng(552)
    nodes=[dict(id='boundary',op='requantize',inputs=['image'],input_shapes=[[8,4,4]],output_shape=[8,4,4],
                input_exps=[-4],output_exp=-12,input_bits=8,output_bits=16)]
    parameters,luts={},{}
    def conv(name,source,co,groups,ei,eo,k=1):
        ci=8
        we=-4 if weight_bits==8 else -12
        lo,hi=q._bounds(weight_bits)
        parameters[name]=dict(weight=rng.integers(lo,hi+1,(co,ci//groups,k,k),dtype=q._dtype(weight_bits)),
                              bias=rng.integers(-2**31,2**31,co,dtype=np.int32))
        m,s=q.h2.ratio(ei+we,eo)
        nodes.append(dict(id=name,op='conv',inputs=[source],input_shapes=[[ci,4,4]],output_shape=[co,4,4],
                          input_exps=[ei],output_exp=eo,input_bits=16,output_bits=16,weight_bits=weight_bits,
                          kernel=k,stride=1,padding=k//2,groups=groups,activation='linear',weight_exp=we,multiplier=m,shift=s))
    conv('group4','boundary',8,4,-12,-9,k=3)
    nodes.append(dict(id='silu',op='silu',inputs=['group4'],input_shapes=[[8,4,4]],output_shape=[8,4,4],
                      input_exps=[-9],output_exp=-10,input_bits=16,output_bits=16))
    luts['silu']=q.silu_lut(-9,-10,input_bits=16,output_bits=16)
    conv('reg','silu',4,4,-10,-8)
    nodes.append(dict(id='relu',op='relu',inputs=['reg'],input_shapes=[[4,4,4]],output_shape=[4,4,4],
                      input_exps=[-8],output_exp=-8,input_bits=16,output_bits=16))
    conv('cls','silu',2,1,-10,-7)
    nodes.append(dict(id='concat',op='concat',inputs=['relu','cls'],input_shapes=[[4,4,4],[2,4,4]],output_shape=[6,4,4],
                      input_exps=[-8,-7],output_exp=-7,input_bits=16,output_bits=16))
    graph=dict(input_shape=[8,4,4],input_exp=-4,nodes=nodes,outputs={'P2':'concat'},
               precision_profile='HEAD_W16A16' if weight_bits==16 else 'HEAD_W8A16')
    integer=q.IntegerReference(graph,parameters,luts)
    fake=q.FakeReference(graph,parameters).eval()
    x=torch.linspace(-8,8,128,dtype=torch.float32).reshape(1,8,4,4)
    actual=integer.forward(integer.quantize_input(x))
    expected=fake.forward_all(x)
    for name in actual:
        assert np.array_equal(actual[name],expected[name].numpy()),name
    assert actual['image'].dtype==np.int8
    assert all(value.dtype==np.int16 for name,value in actual.items() if name!='image')


def test_lut_tamper_is_detected_by_parameter_binding():
    graph,parameters,luts=mixed_primitive_fixture()
    integer=q.IntegerReference(graph,parameters,luts)
    before=integer.parameter_hash()
    integer.luts["silu"][127] += 1
    assert integer.parameter_hash()!=before


def test_export_roundtrip_keeps_all_parameters_luts_and_reference_nodes(tmp_path):
    graph,parameters,luts=mixed_primitive_fixture()
    integer=q.IntegerReference(graph,parameters,luts)
    fake=q.FakeReference(graph,parameters).eval()
    profile=dict(fake_model_state_sha256=q.model_state_hash(fake),integer_parameters_sha256=integer.parameter_hash(),
                 calibration_sha256="a"*64,original_model_state_sha256="b"*64,identity_sha256=None,precision_profile="W8A8")
    prepared=q.PreparedPTQ(torch.nn.Identity(),fake,integer,{},profile)
    witness=q.verify_parity(prepared,torch.ones((1,4,8,8)))
    folder=q.save_artifacts(tmp_path/"export",prepared,witness=witness)
    import json
    manifest=json.loads((folder/"artifact_manifest.json").read_text())
    assert all(q.sha256_file(folder/name)==sha for name,sha in manifest["files"].items())
    index=json.loads((folder/"parameter_index.json").read_text())
    with np.load(folder/"integer_parameters.npz",allow_pickle=False) as saved:
        for row in index["parameters"]:
            assert q.array_sha(saved[row["key"]])==row["sha256"]
            assert saved[row["key"]].dtype.str==row["dtype"]
    assert len(json.loads((folder/"reference_index.json").read_text())["nodes"])==len(graph["nodes"])+1
    assert any(row["kind"]=="lut" for row in index["parameters"])
    with pytest.raises(FileExistsError):
        q.save_artifacts(folder,prepared,witness=witness)


def test_reference_tampering_cannot_be_exported_under_old_witness(tmp_path):
    graph,parameters,luts=mixed_primitive_fixture()
    integer=q.IntegerReference(graph,parameters,luts)
    fake=q.FakeReference(graph,parameters).eval()
    profile=dict(fake_model_state_sha256=q.model_state_hash(fake),integer_parameters_sha256=integer.parameter_hash(),
                 calibration_sha256="a"*64,original_model_state_sha256="b"*64,identity_sha256=None,precision_profile="W8A8")
    prepared=q.PreparedPTQ(torch.nn.Identity(),fake,integer,{},profile)
    witness=q.verify_parity(prepared,torch.ones((1,4,8,8)))
    prepared.reference_outputs["image"][0,0,0,0]+=1
    with pytest.raises(ValueError,match="reference bytes differ"):
        q.save_artifacts(tmp_path/"export",prepared,witness=witness)
