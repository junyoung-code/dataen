"""Three fixed G2 corrections and exact two-player loss attribution."""
import numpy as np
from candidate_validation import mixed

POLICIES=('g2_cold_p4','g2_no_tabm','g2_cold_no_tabm')


def apply_policy(all_errors,known_errors,cold_errors,policy):
    all_errors=np.asarray(all_errors,dtype=float)
    known_errors=np.asarray(known_errors,dtype=float);cold_errors=np.asarray(cold_errors,dtype=float)
    if all_errors.ndim!=2 or all_errors.shape[1]!=3 or len(all_errors)%2:raise ValueError('Two complete groups required')
    np.testing.assert_allclose(all_errors,known_errors+cold_errors,rtol=1e-12,atol=1e-8)
    output=mixed(all_errors).copy();g2=slice(len(output)//2,None)
    if policy=='original':return output
    if policy=='g2_cold_p4':output[g2]=mixed(known_errors[g2])+cold_errors[g2,0]
    elif policy=='g2_no_tabm':output[g2]=.625*all_errors[g2,0]+.375*all_errors[g2,2]
    elif policy=='g2_cold_no_tabm':output[g2]=mixed(known_errors[g2])+.625*cold_errors[g2,0]+.375*cold_errors[g2,2]
    else:raise ValueError('Unknown fixed policy')
    return output


def attribution(base,change_a,change_b,weight):
    base=np.asarray(base);a=np.asarray(change_a);b=np.asarray(change_b);w=np.asarray(weight)
    if not(base.shape==a.shape==b.shape==w.shape):raise ValueError('Attribution shape mismatch')
    first=.5*((np.abs(base+a)-np.abs(base))+(np.abs(base+a+b)-np.abs(base+b)))*w
    second=.5*((np.abs(base+b)-np.abs(base))+(np.abs(base+a+b)-np.abs(base+a)))*w
    np.testing.assert_allclose(first+second,(np.abs(base+a+b)-np.abs(base))*w,rtol=1e-12,atol=1e-7)
    return first,second


def select_policy(calibration_metrics):
    # Candidate definition/order are fixed before A6 confirmation evaluation.
    eligible=[p for p in POLICIES if calibration_metrics[p]['G2_W']<=calibration_metrics['p4']['G2_W']+1e-5]
    return min(eligible,key=lambda p:(calibration_metrics[p]['W'],POLICIES.index(p))) if eligible else 'original'


def support(metrics):
    reasons=[]
    if metrics['F2']['selected']['G2_W']>metrics['F2']['p4']['G2_W']+1e-5:reasons.append('F2 full G2 worse than P4')
    if metrics['F2_validation']['selected']['G2_W']>metrics['F2_validation']['p4']['G2_W']+1e-5:reasons.append('F2 later-half G2 worse than P4')
    if metrics['F3']['selected']['W']>metrics['F3']['p4']['W']+1e-5:reasons.append('F3 total worse than P4')
    if metrics['F3']['selected']['G2_W']>metrics['F3']['p4']['G2_W']+1e-5:reasons.append('F3 G2 worse than P4')
    if metrics['F3']['selected']['W']>metrics['F3']['original']['W']*1.0025:reasons.append('F3 total costs more than 0.25% versus original mixture')
    return dict(classification='supported_historical_correction' if not reasons else 'hold',reasons=reasons,
        independent_final_test=False,automatic_model_replacement=False)
