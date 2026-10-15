"""Versioned measured-operator precision profiles; INT8 body stays on H2 V1.

HEAD_W8A16 and HEAD_W16A16 keep the backbone and neck at W8A8 and change only
the detection head. W8A16 / W16A16 are aliases of those head contracts. They
hash separately from each other and from uniform W8A8.
"""
import hashlib
import json

PROFILES = {'W8A8':(8,8), 'HEAD_W8A16':(8,16), 'HEAD_W16A16':(16,16)}
PROFILE_ALIASES = {
    'W8A8':'W8A8',
    'HEAD_W8A8':'W8A8',
    'NUMERIC_CONTRACT_V1':'W8A8',
    'numeric_contract_v1':'W8A8',
    'W8A16':'HEAD_W8A16',
    'HEAD_W8A16':'HEAD_W8A16',
    'W16A16':'HEAD_W16A16',
    'HEAD_W16A16':'HEAD_W16A16',
}

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def canonical_profile(name):
    return PROFILE_ALIASES.get(name)

def operator_contract(weight_bits, activation_bits, body_hashes):
    if (weight_bits,activation_bits)==(8,8):
        return dict(body_hashes)
    if (weight_bits,activation_bits) not in ((8,16),(16,16)):
        raise ValueError('Unsupported measured precision')
    numeric=dict(parent=body_hashes['numeric_contract_hash'],version='head_mixed_v1',
        input_bits=activation_bits,output_bits=activation_bits,weight_bits=weight_bits,
        accumulator_bits=32,overflow='twos_complement_modulo_2pow32',rounding='nearest_even',
        zero_point=0,requant_product_bits=64,scale='positive_power_of_two_exponent_int32')
    interface=dict(parent=body_hashes['interface_contract_hash'],version='head_mixed_axis_v1',
        activation_bits=activation_bits,max_axis_bits=256,layout='pixel_major_channel_segments',
        tail_padding='zero',last='final_pixel_final_segment')
    return dict(numeric_contract_hash=digest(numeric),interface_contract_hash=digest(interface))

def quantization_regions(profile):
    """Backbone, neck, and head bit-widths. Only the head changes across profiles."""
    profile=canonical_profile(profile) or profile
    if profile not in PROFILES:
        raise ValueError('Unsupported quantization profile')
    head_w,head_a=PROFILES[profile]
    body=dict(weight_bitwidth=8,activation_bitwidth=8)
    return dict(profile=profile,backbone=dict(body),neck=dict(body),
        head=dict(weight_bitwidth=head_w,activation_bitwidth=head_a))

def quant_contract(profile, body_hashes):
    """Full candidate quantization contract. Head W8A16 and W16A16 do not share a hash."""
    regions=quantization_regions(profile)
    head=regions['head']
    head_op=operator_contract(head['weight_bitwidth'],head['activation_bitwidth'],body_hashes)
    payload=dict(version='hw_quant_contract_v1',regions=regions,
        body_numeric_contract_hash=body_hashes['numeric_contract_hash'],
        body_interface_contract_hash=body_hashes['interface_contract_hash'],
        head_numeric_contract_hash=head_op['numeric_contract_hash'],
        head_interface_contract_hash=head_op['interface_contract_hash'],
        accumulator_bitwidth=32)
    return dict(payload,quant_contract_hash=digest(payload))

def node_precision(profile, node):
    regions=quantization_regions(profile)
    # Head nodes use the head contract. Backbone, neck, and other body nodes stay W8A8.
    picked=regions['head'] if str(node).startswith('head.') else regions['backbone']
    return dict(input_bitwidth=picked['activation_bitwidth'],weight_bitwidth=picked['weight_bitwidth'],
        accumulator_bitwidth=32,output_bitwidth=picked['activation_bitwidth'])
