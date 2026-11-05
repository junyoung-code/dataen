import unittest
import numpy as np
from g2_correction import POLICIES,apply_policy,attribution,select_policy,support
from candidate_validation import mixed


class G2Tests(unittest.TestCase):
    def test_known_group_unchanged_and_cold_fallback(self):
        known=np.array([[1.,2.,3.],[4.,5.,6.],[1.,2.,3.],[4.,5.,6.]])
        cold=np.array([[10.,-10.,20.]]*4);all_errors=known+cold
        for policy in POLICIES:
            got=apply_policy(all_errors,known,cold,policy)
            np.testing.assert_array_equal(got[:2],mixed(all_errors)[:2])
        got=apply_policy(all_errors,known,cold,'g2_cold_p4')
        np.testing.assert_allclose(got[2:],mixed(known)[2:]+cold[2:,0])

    def test_cohort_policy_equals_site_mix_before_group_sum(self):
        p=np.array([[10.,-5.,8.],[3.,4.,2.],[7.,-1.,9.]])
        known=p[:1].sum(axis=0);cold=p[1:].sum(axis=0)
        all_errors=np.tile(known+cold,(4,1));a=np.tile(known,(4,1));b=np.tile(cold,(4,1))
        got=apply_policy(all_errors,a,b,'g2_cold_no_tabm')
        expected=float(mixed(p[:1]).sum()+np.sum(.625*p[1:,0]+.375*p[1:,2]))
        self.assertAlmostEqual(got[-1],expected)

    def test_attribution_exact_sum_and_symmetry(self):
        rng=np.random.default_rng(1);p,a,b=rng.normal(size=(3,100));w=np.arange(1,101.)
        x,y=attribution(p,a,b,w);u,v=attribution(p,b,a,w)
        np.testing.assert_allclose(x+y,(np.abs(p+a+b)-np.abs(p))*w,atol=1e-12)
        np.testing.assert_allclose(x,v,atol=1e-12);np.testing.assert_allclose(y,u,atol=1e-12)

    def test_selection_uses_calibration_metrics_and_p4_guard(self):
        rows={'p4':dict(W=100,G2_W=30)}
        rows.update({p:dict(W=80+i,G2_W=31 if i==0 else 29) for i,p in enumerate(POLICIES)})
        self.assertEqual(select_policy(rows),POLICIES[1])
        for p in POLICIES:rows[p]['G2_W']=40
        self.assertEqual(select_policy(rows),'original')

    def test_zero_cold_equals_original_and_unknown_rule_fails(self):
        e=np.ones((8,3));z=np.zeros_like(e)
        for p in ['g2_cold_p4','g2_cold_no_tabm']:np.testing.assert_array_equal(apply_policy(e,e,z,p),mixed(e))
        with self.assertRaises(ValueError):apply_policy(e,e,z,'bad')

    def test_support_checks_recent_cost_and_group_risk(self):
        rows={fold:{'p4':dict(W=100.,G2_W=30.),'original':dict(W=90.,G2_W=32.),'selected':dict(W=90.1,G2_W=29.)} for fold in ['F2','F2_validation','F3']}
        self.assertEqual(support(rows)['classification'],'supported_historical_correction')
        rows['F3']['selected']['W']=91.
        self.assertEqual(support(rows)['classification'],'hold')


if __name__=='__main__':unittest.main()
