"""NUMERIC_CONTRACT_V1: integer-only runtime arithmetic, symmetric zero point."""
import numpy as np

CONTRACT = dict(version=1,activation_bits=8,weight_bits=8,accumulator_bits=32,
    range=[-128,127],zero_point=0,rounding='nearest_even',overflow='twos_complement_modulo_2pow32',
    scale='positive_power_of_two_exponent_int32',scale_exponent_range=[-24,16],
    multiplier_range=[1,2147483647],shift_range=[0,31],bias='signed_int32_little_endian_at_input_times_weight_scale',
    add='align_each_to_common_INT8_then_INT32_sum_then_requant_to_output',common_scale='max(input_scales)',
    activation='linear_relu_scaled_relu6_after_requant',requant_product_bits=64)

def wrap32(a):
    a=np.asarray(a,dtype=np.int64)
    return ((a+2**31)%2**32-2**31).astype(np.int64)

def round_shift(a,shift):
    if not 0<=int(shift)<=31:raise ValueError('shift outside V1')
    a=np.asarray(a,dtype=np.int64);absolute=np.abs(a)
    q=absolute>>shift
    if shift:
        remainder=absolute&((1<<shift)-1);half=1<<(shift-1)
        q=q+((remainder>half)|((remainder==half)&((q&1)==1)))
    return np.where(a<0,-q,q)

def requantize_int32_to_int8(accumulator,multiplier,shift,rounding_mode='nearest_even',saturation_range=(-128,127)):
    if rounding_mode!='nearest_even' or tuple(saturation_range)!=(-128,127):raise ValueError('V1 has one rounding/range policy')
    if not 1<=int(multiplier)<=2**31-1:raise ValueError('multiplier outside V1')
    a=np.asarray(accumulator,dtype=np.int64)
    if np.any(a<-(2**31)) or np.any(a>2**31-1):raise ValueError('requant input must be INT32; use wrap32 for accumulators')
    return np.clip(round_shift(a*int(multiplier),int(shift)),-128,127).astype(np.int8)

def ratio(from_exp,to_exp):
    delta=int(from_exp)-int(to_exp)
    if not -31<=delta<=30:raise ValueError('scale ratio not representable in V1')
    return (1<<delta,0) if delta>=0 else (1,-delta)

def q6(exponent):
    # Exact powers of two, ties-to-even. Bound before conversion.
    return int(np.clip(np.rint(6.0/(2.0**exponent)),0,127))

def activation(a,kind,exponent):
    a=np.asarray(a,dtype=np.int8)
    if kind=='linear':return a
    if kind=='relu':return np.maximum(a,0).astype(np.int8)
    if kind=='relu6':return np.clip(a,0,q6(exponent)).astype(np.int8)
    raise ValueError(kind)

def quantize_real(a,exponent,bias=False):
    a=np.asarray(a,dtype=np.float64)
    if not np.isfinite(a).all():raise ValueError('nonfinite parameter')
    values=np.rint(a/(2.0**exponent))
    bounds=(-2**31,2**31-1) if bias else (-128,127)
    return np.clip(values,*bounds).astype('<i4' if bias else 'i1')

def aligned_add(a,b,ea,eb,eo):
    common=max(ea,eb)
    qa=requantize_int32_to_int8(np.asarray(a,dtype=np.int32),*ratio(ea,common))
    qb=requantize_int32_to_int8(np.asarray(b,dtype=np.int32),*ratio(eb,common))
    total=qa.astype(np.int32)+qb.astype(np.int32)
    return requantize_int32_to_int8(total,*ratio(common,eo))

def conv_acc(x,w,b,stride=1,padding=0,groups=1):
    """Independent NumPy INT64 sums then exact INT32 modulo; tensors are CHW/OIHW."""
    ci,h,wi=x.shape;co,_,k,_=w.shape
    oh=(h+2*padding-k)//stride+1;ow=(wi+2*padding-k)//stride+1
    p=np.pad(x.astype(np.int64),((0,0),(padding,padding),(padding,padding)))
    out=np.zeros((oh,ow,co),dtype=np.int64)
    if groups==ci==co:
        for kh in range(k):
            for kw in range(k):out+=p[:,kh:kh+oh*stride:stride,kw:kw+ow*stride:stride].transpose(1,2,0)*w[:,0,kh,kw].astype(np.int64)
    elif groups==1:
        for kh in range(k):
            for kw in range(k):
                sample=p[:,kh:kh+oh*stride:stride,kw:kw+ow*stride:stride].transpose(1,2,0)
                out+=sample@w[:,:,kh,kw].astype(np.int64).T
    else:raise ValueError('V1 supports dense or depthwise only')
    return wrap32(out.transpose(2,0,1)+np.asarray(b,dtype=np.int64)[:,None,None])

def conv(x,w,b,meta):
    acc=conv_acc(x,w,b,meta['stride'],meta['padding'],meta['groups'])
    return activation(requantize_int32_to_int8(acc,meta['multiplier'],meta['shift']),meta['activation'],meta['output_exp'])
