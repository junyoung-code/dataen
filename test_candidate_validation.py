import unittest
import numpy as np
from candidate_validation import block_bootstrap,daily_amounts,mixed,perturbations


class CandidateTests(unittest.TestCase):
    def test_constant_paired_improvement_and_seed(self):
        base=np.arange(1,61,dtype=float)
        first=block_bootstrap(base,.9*base,7,1,reps=200)
        self.assertEqual(first,block_bootstrap(base,.9*base,7,1,reps=200))
        np.testing.assert_allclose(first['ci95_percent'],[10,10],atol=1e-12)
        self.assertEqual(first['fraction_improved'],1.)

    def test_identical_and_worse(self):
        b=np.ones(60)
        self.assertEqual(block_bootstrap(b,b,28,1,200)['ci95_percent'],[0.,0.])
        self.assertEqual(block_bootstrap(b,2*b,7,1,200)['fraction_improved'],0.)

    def test_right_closed_daily_alignment(self):
        e=np.arange(2*48,dtype=float);w=np.ones_like(e)
        got=daily_amounts(e,w)
        self.assertEqual(got.shape,(2,2))
        self.assertEqual(got[0,0],sum(range(24)))
        self.assertEqual(got[1,1],sum(range(72,96)))
        with self.assertRaises(ValueError):daily_amounts(e[:-1],w[:-1])

    def test_signed_error_mix_and_nonnegative_weights(self):
        self.assertAlmostEqual(float(mixed(np.array([10.,-10.,0.]))),3.)
        with self.assertRaises(ValueError):mixed(np.ones((2,3)),[1,1,-1])

    def test_fixed_perturbations_not_search(self):
        policies=perturbations();self.assertEqual(len(policies),9)
        for w in policies.values():self.assertGreaterEqual(min(w),0);self.assertAlmostEqual(sum(w),1.)

    def test_zero_information_and_invalid_blocks(self):
        r=block_bootstrap(np.zeros(60),np.zeros(60),7,1,200)
        self.assertIsNone(r['ci95_percent'])
        with self.assertRaises(ValueError):block_bootstrap(np.ones(5),np.ones(5),7,1)


if __name__=='__main__':unittest.main()
