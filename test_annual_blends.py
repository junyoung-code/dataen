import unittest
import numpy as np
from annual_blend_core import candidates,grid_fit,fit_policy,apply,gate,select_conditional


class BlendTests(unittest.TestCase):
    def test_simplex_and_ties(self):
        grid=candidates()
        self.assertEqual(len(grid),231)
        self.assertTrue((grid>=0).all())
        np.testing.assert_allclose(grid.sum(axis=1),1,rtol=0,atol=1e-12)
        self.assertEqual(grid_fit(np.ones((4,3)),np.ones(4))['weights'],[1.,0.,0.])

    def test_large_finite_grid_under_strict_float_checks(self):
        errors=(np.arange(60000,dtype=float).reshape(20000,3)%1000)-500
        with np.errstate(all='raise'):
            fitted=grid_fit(errors,np.ones(len(errors)))
        self.assertTrue(np.isfinite([r['W'] for r in fitted['candidates']]).all())

    def test_cancellation_not_average_absolute_error(self):
        fit=grid_fit(np.array([[10,-10,20],[20,-20,40.]]),np.ones(2))
        self.assertEqual(fit['weights'],[.5,.5,0.])
        self.assertAlmostEqual(fit['W'],0.)

    def test_same_mask_site_mix_equals_team_error_mix(self):
        truth=np.array([1.,2.,3.,4.]);pred=np.array([[2.,4.,3.],[5.,1.,4.],[9.,2.,1.],[3.,8.,2.]])
        keep=np.array([True,False,True,True]);weight=np.array([.7,.2,.1])
        team=np.sum(pred[keep]-truth[keep,None],axis=0)
        self.assertAlmostEqual(float(team@weight),float(np.sum(pred[keep]@weight-truth[keep])))

    def test_gate_boundaries_missing_and_fallback(self):
        hours=np.array([10,11,15,16]);cloud=np.array([49.,50.,np.nan,80.])
        np.testing.assert_array_equal(gate('peak',hours,cloud),[0,1,1,0])
        np.testing.assert_array_equal(gate('cloud',hours,cloud),[0,1,-1,1])
        policy=dict(rule='cloud',fallback=[1,0,0],bins={'1':[0,1,0]})
        np.testing.assert_array_equal(apply(policy,np.tile([1,2,3],(4,1)),hours,cloud),[1,2,1,2])

    def test_sparse_bin_falls_back_and_invalid_shape_rejected(self):
        p=fit_policy(np.ones((10,3)),np.ones(10),np.arange(10),np.zeros(10),'cloud')
        self.assertEqual(p['bins'],{})
        with self.assertRaises(ValueError):grid_fit(np.ones((3,2)),np.ones(3))

    def test_future_arrays_not_available_to_calibration_selector(self):
        errors=np.tile([10.,-10.,30.],(800,1));weights=np.ones(800);hours=np.tile(np.arange(20),40);cloud=np.zeros(800)
        train=np.arange(800)<400;valid=~train
        before=select_conditional(errors,weights,hours,cloud,train,valid)
        self.assertEqual(before['selected_rule'],'global')
        with self.assertRaises(ValueError):select_conditional(errors,weights,hours,cloud,train,train)

    def test_conditional_calibration_selects_simpler_correct_gate(self):
        n=1600;hours=np.tile([10,12],n//2);cloud=np.zeros(n)
        errors=np.where((hours==12)[:,None],np.array([10,20,-10]),np.array([10,-10,20])).astype(float)
        weights=np.ones(n);train=np.arange(n)<800;valid=~train
        result=select_conditional(errors,weights,hours,cloud,train,valid)
        self.assertEqual(result['selected_rule'],'peak')
        policy=fit_policy(errors,weights,hours,cloud,'peak')
        self.assertAlmostEqual(float(np.abs(apply(policy,errors,hours,cloud)).sum()),0.)


if __name__=='__main__':unittest.main()
