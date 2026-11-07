"""Deterministic feasible single-move / pair-swap search on frozen tune residuals."""
import numpy as np


def validate_arrays(errors, capacities, weights, group1, minimum):
    errors=np.asarray(errors,dtype=float);capacities=np.asarray(capacities,dtype=float)
    weights=np.asarray(weights,dtype=float);group1=np.asarray(group1)
    if errors.ndim!=2 or capacities.shape!=(errors.shape[1],) or weights.shape!=(errors.shape[0],):
        raise ValueError('Expected hours x sites residuals and aligned capacities/weights')
    if group1.shape!=capacities.shape or group1.dtype!=bool:
        raise ValueError('Boolean assignment must contain every site exactly once')
    if not all(np.isfinite(x).all() for x in [errors,capacities,weights]) or (capacities<=0).any():
        raise ValueError('Finite inputs and positive capacities required')
    if (weights<0).any() or weights.sum()<=0 or not np.isfinite(minimum) or minimum<=0:
        raise ValueError('Valid weights and positive minimum capacity required')
    if capacities[group1].sum()<minimum or capacities[~group1].sum()<minimum:
        raise ValueError('Both teams must satisfy minimum capacity')
    return errors,capacities,weights,group1.copy()


def score(errors,weights,group1):
    first=errors[:,group1].sum(axis=1);second=errors[:,~group1].sum(axis=1)
    return float(np.sum(weights*(np.abs(first)+np.abs(second)))/(2*weights.sum()))


def optimize(errors,capacities,weights,group1,minimum=20000.,max_steps=50,callback=None):
    errors,capacities,weights,group1=validate_arrays(errors,capacities,weights,group1,minimum)
    if not isinstance(max_steps,int) or max_steps<0:raise ValueError('Nonnegative integer search limit required')
    total=errors.sum(axis=1);denominator=2*weights.sum()
    initial=score(errors,weights,group1)
    history=[];stop='step_limit';evaluated=0
    for step in range(max_steps):
        first=errors[:,group1].sum(axis=1);current=score(errors,weights,group1)
        cap=capacities[group1].sum();upper=capacities.sum()-minimum
        # Stable site order: all single moves, then lexicographic G1/G2 swaps.
        operations=[]
        for i in range(len(capacities)):
            next_cap=cap-capacities[i] if group1[i] else cap+capacities[i]
            if minimum<=next_cap<=upper:operations.append((i,-1))
        for i in np.flatnonzero(group1):
            for j in np.flatnonzero(~group1):
                next_cap=cap-capacities[i]+capacities[j]
                if minimum<=next_cap<=upper:operations.append((int(i),int(j)))
        best=current;action=None
        for offset in range(0,len(operations),128):
            chunk=operations[offset:offset+128]
            proposed=np.repeat(first[:,None],len(chunk),axis=1)
            for k,(i,j) in enumerate(chunk):
                proposed[:,k]+=(-errors[:,i] if group1[i] else errors[:,i])
                if j>=0:proposed[:,k]+=errors[:,j]
            values=np.sum(weights[:,None]*(np.abs(proposed)+np.abs(total[:,None]-proposed)),axis=0)/denominator
            # First operation wins ties within 1e-9, avoiding floating-point churn.
            for k,value in enumerate(values):
                if value<best-1e-9:best=float(value);action=chunk[k]
        evaluated+=len(operations)
        if action is None:stop='no_improving_feasible_move_or_swap';break
        i,j=action;group1[i]=not group1[i]
        if j>=0:group1[j]=not group1[j]
        validate_arrays(errors,capacities,weights,group1,minimum)
        actual=score(errors,weights,group1)
        np.testing.assert_allclose(actual,best,rtol=1e-12,atol=1e-9)
        entry=dict(step=step+1,kind='move' if j<0 else 'swap',site_indices=[i] if j<0 else [i,j],
                   weighted_group_mae=actual,capacity_g1=float(capacities[group1].sum()),
                   capacity_g2=float(capacities[~group1].sum()),neighbors_evaluated=len(operations))
        history.append(entry)
        if callback:callback(entry)
    lower_bound=float(np.sum(weights*np.abs(total))/denominator)
    return group1,dict(initial_score=initial,final_score=score(errors,weights,group1),
                       mathematical_lower_bound=lower_bound,steps=len(history),stop_reason=stop,
                       candidates_evaluated=evaluated,history=history)
