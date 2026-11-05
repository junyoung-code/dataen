"""Fixed-candidate diagnostics; no training or weight selection."""
import math
import numpy as np

WEIGHTS=np.array([.5,.2,.3])
REPLICATES=5000
BLOCKS=(7,28)
SEED=20261105
SUBSETS=('all','seen','new')
SCOPES=('total','G1','G2')


def mixed(errors,weights=WEIGHTS):
    e=np.asarray(errors,dtype=float);w=np.asarray(weights,dtype=float)
    if e.shape[-1]!=3 or w.shape!=(3,) or not np.isfinite(e).all():raise ValueError('Invalid errors')
    if (w<0).any() or not np.isclose(w.sum(),1.,rtol=0,atol=1e-12):raise ValueError('Nonconvex weights')
    return e[...,0]*w[0]+e[...,1]*w[1]+e[...,2]*w[2]


def daily_amounts(error,weights):
    error=np.asarray(error,dtype=float);weights=np.asarray(weights,dtype=float)
    if error.shape!=weights.shape or error.ndim!=1 or len(error)%48:raise ValueError('Complete two-team days required')
    if not np.isfinite(error).all() or not np.isfinite(weights).all() or (weights<0).any():raise ValueError('Invalid values')
    return (np.abs(error)*weights).reshape(2,-1,24).sum(axis=2)


def improvement(base,candidate):
    b=float(np.sum(base));c=float(np.sum(candidate))
    return 100*(1-c/b) if b>0 else None


def block_bootstrap(base,candidate,block,seed,reps=REPLICATES):
    b=np.asarray(base,dtype=float);c=np.asarray(candidate,dtype=float)
    if b.ndim!=1 or b.shape!=c.shape or len(b)<block or block<1:raise ValueError('Invalid block/series')
    if not np.isfinite(b).all() or not np.isfinite(c).all() or (b<0).any() or (c<0).any():raise ValueError('Invalid daily amounts')
    result=dict(block_days=block,replicates=reps,seed=seed,point_improvement_percent=improvement(b,c))
    if b.sum()<=0:return dict(**result,valid_replicates=0,ci95_percent=None,fraction_improved=None)
    rng=np.random.default_rng(seed);samples=[];n=len(b);blocks=math.ceil(n/block)
    for start in range(0,reps,250):
        count=min(250,reps-start)
        offsets=rng.integers(0,n,size=(count,blocks))
        indices=((offsets[:,:,None]+np.arange(block))%n).reshape(count,-1)[:,:n]
        bs=b[indices].sum(axis=1);cs=c[indices].sum(axis=1)
        valid=bs>0
        samples.extend((100*(1-cs[valid]/bs[valid])).tolist())
    if len(samples)<.99*reps:return dict(**result,valid_replicates=len(samples),ci95_percent=None,fraction_improved=None)
    return dict(**result,valid_replicates=len(samples),ci95_percent=np.quantile(samples,[.025,.975]).tolist(),
        fraction_improved=float(np.mean(np.asarray(samples)>0)))


def perturbations():
    result={}
    for i in range(3):
        for j in range(3):
            if i==j:continue
            w=WEIGHTS.copy();w[i]-=.05;w[j]+=.05
            result[f'move_05_{i}_to_{j}']=w.tolist()
    for i in range(3):
        w=WEIGHTS.copy();w[i]=0;w/=w.sum()
        result[f'exclude_{i}']=w.tolist()
    return result


def support(f3):
    reasons=[]
    total=f3['all']['total']
    if total['point_improvement_percent'] is None or total['point_improvement_percent']<=0:
        return dict(classification='no_historical_support',reasons=['F3 overall not improved'])
    for block in BLOCKS:
        ci=total['bootstrap'][str(block)]['ci95_percent']
        if ci is None or ci[0]<=0:reasons.append(f'F3 overall {block}-day interval includes non-improvement')
    seen=f3['seen']['total']['point_improvement_percent']
    if seen is None or seen<0:reasons.append('F3 previously observed subset not improved')
    for group in ['G1','G2']:
        p=f3['all'][group]['point_improvement_percent']
        if p is None or p<0:reasons.append(f'F3 {group} not improved')
    return dict(classification='mixed_historical_support' if reasons else 'strong_historical_support',reasons=reasons,
        independent_final_test=False,automatic_model_replacement=False)
