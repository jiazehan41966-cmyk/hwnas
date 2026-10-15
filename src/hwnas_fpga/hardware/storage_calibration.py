"""Small, auditable coupled BRAM/LUTRAM calibration for one FPGA mapping."""
import math

def upper_line(points):
    """Nonnegative least-squares slope, then an intercept covering training data."""
    if len(points)<2: raise ValueError('At least two independent training points are required')
    points=[(float(x),float(y)) for x,y in points]
    if any(not math.isfinite(x) or not math.isfinite(y) or x<0 for x,y in points):
        raise ValueError('Invalid calibration measurements')
    mx=sum(x for x,_ in points)/len(points);my=sum(y for _,y in points)/len(points)
    variance=sum((x-mx)**2 for x,_ in points)
    slope=max(0,sum((x-mx)*(y-my) for x,y in points)/variance) if variance else 0
    intercept=max(0,max(y-slope*x for x,y in points))
    return dict(slope=slope,intercept=intercept)

def line_value(line,x):
    slope=float(line['slope']);intercept=float(line['intercept'])
    if any(not math.isfinite(v) or v<0 for v in [slope,intercept,x]):
        raise ValueError('Invalid coupled calibration coefficients/input')
    return slope*x+intercept

def fit_coupled_storage(training):
    identities=[r['architecture_sha256'] for r in training]
    if len(set(identities))!=len(identities): raise ValueError('Duplicate training architecture')
    request=upper_line([(r['bram18_hls_sum'],r['requested_bram18']) for r in training])
    excess=[max(0,line_value(request,r['bram18_hls_sum'])-624) for r in training]
    residual={key:upper_line([(x,r['actual'][key]-r['operator_sum'][key]) for x,r in zip(excess,training)])
              for key in ['lut_as_logic','lut_as_memory','ff','dsp']}
    hls=[r['bram18_hls_sum'] for r in training]
    return dict(model_kind='coupled_bram_lutram_v1',bram_request_model=request,overflow_residual_models=residual,
                max_calibrated_excess_bram18=max(excess),hls_bram18_domain=[min(hls),max(hls)])

def predict_coupled_storage(raw,hls_sum,model):
    requested=line_value(model['bram_request_model'],float(hls_sum))
    excess=max(0,requested-624)
    upper=float(model['max_calibrated_excess_bram18'])
    lo,hi=map(float,model['hls_bram18_domain'])
    if any(not math.isfinite(v) or v<0 for v in [upper,lo,hi]) or lo>hi:
        raise ValueError('Invalid coupled calibration domain')
    unknown=excess>upper+1e-8 or hls_sum<lo-1e-8 or hls_sum>hi+1e-8
    resources=dict(raw)
    # The physical allocation bound is meaningful only with the corresponding
    # overflow cost. Unknown overflow regimes are rejected by the caller.
    resources['bram36']=min(312,requested/2)
    for key in ['lut_as_logic','lut_as_memory','ff','dsp']:
        resources[key]+=line_value(model['overflow_residual_models'][key],excess)
    resources['lut']=resources['lut_as_logic']+resources['lut_as_memory']
    return resources,dict(requested_bram18=requested,excess_bram18=excess,domain_unknown=unknown)
