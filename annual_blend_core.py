"""Convex team-error blending; selectors receive calibration arrays only."""
import numpy as np

MODELS=('p4','tabm','tft')
DENOMINATOR=20
MIN_BIN_ROWS=200
CONDITIONAL_MARGIN=.0025
RULES=('global','peak','cloud','peak_cloud')


def candidates():
    # P4-heavy solutions win exact ties, then less TFT, then less TabM.
    rows=[(1-(t+f)/DENOMINATOR,t/DENOMINATOR,f/DENOMINATOR)
          for t in range(DENOMINATOR+1) for f in range(DENOMINATOR+1-t)]
    return np.asarray(sorted(rows,key=lambda w:(1-w[0],w[2],w[1])),dtype=float)


def validate(errors,weights):
    errors=np.asarray(errors,dtype=float);weights=np.asarray(weights,dtype=float)
    if errors.ndim!=2 or errors.shape[1]!=3 or weights.shape!=(len(errors),):raise ValueError('Invalid blend shape')
    if not np.isfinite(errors).all() or not np.isfinite(weights).all() or (weights<0).any() or weights.sum()<=0:
        raise ValueError('Invalid errors/price weights')
    return errors,weights


def grid_fit(errors,weights):
    errors,weights=validate(errors,weights)
    grid=candidates()
    # Three explicit components avoid a matmul floating-point exception in the
    # deployed NumPy/macOS combination even for bounded, entirely finite arrays.
    scores=np.array([np.sum(np.abs(errors[:,0]*w[0]+errors[:,1]*w[1]+errors[:,2]*w[2])*weights) for w in grid])
    if not np.isfinite(scores).all():raise ValueError('Nonfinite candidate W')
    best=float(scores.min())
    chosen=int(np.flatnonzero(scores<=best+1e-6)[0])
    return dict(weights=grid[chosen].tolist(),W=float(scores[chosen]),
        candidates=[dict(weights=w.tolist(),W=float(s)) for w,s in zip(grid,scores)])


def gate(rule,hours,cloud):
    hours=np.asarray(hours);cloud=np.asarray(cloud,dtype=float)
    peak=((hours>=11)&(hours<=15)).astype(int)
    cloudy=(cloud>=50).astype(int)
    if rule=='global':return np.zeros(len(hours),dtype=int)
    if rule=='peak':return peak
    if rule=='cloud':return np.where(np.isfinite(cloud),cloudy,-1)
    if rule=='peak_cloud':return np.where(np.isfinite(cloud),peak*2+cloudy,-1)
    raise ValueError('Unknown gate rule')


def fit_policy(errors,weights,hours,cloud,rule):
    errors,weights=validate(errors,weights)
    fit=grid_fit(errors,weights)
    policy=dict(rule=rule,fallback=fit['weights'],bins={},minimum_rows=MIN_BIN_ROWS)
    if rule=='global':return policy
    labels=gate(rule,hours,cloud)
    for label in sorted(set(labels)-{-1}):
        keep=(labels==label)&(weights>0)&np.any(errors!=0,axis=1)
        if int(keep.sum())<MIN_BIN_ROWS:continue
        policy['bins'][str(label)]=grid_fit(errors[keep],weights[keep])['weights']
    return policy


def coefficients(policy,hours,cloud):
    result=np.tile(np.asarray(policy['fallback'],dtype=float),(len(hours),1))
    labels=gate(policy['rule'],hours,cloud)
    for label,weights in policy['bins'].items():result[labels==int(label)]=weights
    if (result<0).any() or not np.allclose(result.sum(axis=1),1.,rtol=0,atol=1e-12):raise ValueError('Nonconvex coefficients')
    return result


def apply(policy,errors,hours,cloud):
    return np.sum(np.asarray(errors)*coefficients(policy,hours,cloud),axis=1)


def select_conditional(errors,weights,hours,cloud,calibration,validation):
    calibration=np.asarray(calibration,dtype=bool);validation=np.asarray(validation,dtype=bool)
    if np.any(calibration&validation) or not calibration.any() or not validation.any():raise ValueError('Disjoint nonempty calibration/validation required')
    rows=[]
    for rule in RULES:
        policy=fit_policy(errors[calibration],weights[calibration],np.asarray(hours)[calibration],np.asarray(cloud)[calibration],rule)
        pred=apply(policy,errors[validation],np.asarray(hours)[validation],np.asarray(cloud)[validation])
        rows.append(dict(rule=rule,validation_W=float(np.sum(np.abs(pred)*weights[validation])),calibration_policy=policy))
    baseline=rows[0]['validation_W']
    best=min(rows[1:],key=lambda r:(r['validation_W'],RULES.index(r['rule'])))
    chosen=best['rule'] if best['validation_W']<baseline*(1-CONDITIONAL_MARGIN) else 'global'
    return dict(selected_rule=chosen,best_conditional_rule=best['rule'],margin=CONDITIONAL_MARGIN,
        validation_global_W=baseline,validation_candidates=rows)
