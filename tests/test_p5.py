import unittest
import numpy as np
import pandas as pd
from team_assignment import optimize,score
from run_p5 import check_assignment


class AssignmentTests(unittest.TestCase):
    def test_swap_cancels_same_hour_errors_and_is_deterministic(self):
        errors=np.array([[4.,4.,-4.,-4.],[2.,2.,-2.,-2.]])
        caps=np.ones(4);weights=np.array([3.,1.]);initial=np.array([True,True,False,False])
        chosen,report=optimize(errors,caps,weights,initial,minimum=2.)
        np.testing.assert_array_equal(chosen,[False,True,True,False])
        np.testing.assert_array_equal(initial,[True,True,False,False])
        self.assertEqual(report['steps'],1);self.assertEqual(report['final_score'],0.)
        self.assertEqual(report['stop_reason'],'no_improving_feasible_move_or_swap')

    def test_first_step_matches_brute_force_with_capacity_constraints(self):
        errors=np.array([[6.,4.,-7.,-3.,1.],[-2.,1.,4.,-3.,2.]])
        caps=np.array([1.,1.,1.,1.,1.]);weights=np.array([1.,3.]);initial=np.array([True,True,False,False,False])
        scores=[score(errors,weights,initial)]
        for i in range(5):
            for j in [-1]+list(range(i+1,5)):
                if j>=0 and initial[i]==initial[j]:continue
                proposed=initial.copy();proposed[i]=not proposed[i]
                if j>=0:proposed[j]=not proposed[j]
                if caps[proposed].sum()>=2 and caps[~proposed].sum()>=2:scores.append(score(errors,weights,proposed))
        chosen,report=optimize(errors,caps,weights,initial,minimum=2.,max_steps=1)
        self.assertAlmostEqual(report['final_score'],min(scores))
        self.assertGreaterEqual(caps[chosen].sum(),2);self.assertGreaterEqual(caps[~chosen].sum(),2)

    def test_lower_bound_and_no_improvement_do_not_invent_gain(self):
        errors=np.ones((3,4));caps=np.ones(4);weights=np.ones(3);initial=np.array([True,True,False,False])
        chosen,report=optimize(errors,caps,weights,initial,minimum=2.)
        np.testing.assert_array_equal(chosen,initial)
        self.assertEqual(report['steps'],0);self.assertEqual(report['final_score'],report['mathematical_lower_bound'])
        _,zero=optimize(errors,caps,weights,initial,minimum=2.,max_steps=0)
        self.assertEqual(zero['stop_reason'],'step_limit')

    def test_invalid_search_inputs_fail(self):
        e=np.ones((2,4));c=np.ones(4);w=np.ones(2);g=np.array([True,True,False,False])
        for args in [(e,c,w,g,3.),(e,c,w,[1,1,0,0],2.),(e,c,np.zeros(2),g,2.),(e*np.nan,c,w,g,2.)]:
            with self.assertRaises(ValueError):optimize(*args)

    def test_assignment_requires_all_sites_and_capacity(self):
        sites=pd.DataFrame(dict(site_id=['a','b','c','d'],capacity_kw=[10000.]*4))
        assignment=pd.DataFrame(dict(site_id=['a','b','c','d'],group=['G1','G1','G2','G2']))
        self.assertEqual(check_assignment(assignment,sites)['G1']['capacity_kw'],20000.)
        for bad in [assignment.iloc[:-1],pd.concat([assignment,assignment.iloc[:1]]),assignment.assign(group='G1')]:
            with self.assertRaises(ValueError):check_assignment(bad,sites)


if __name__=='__main__':unittest.main()
