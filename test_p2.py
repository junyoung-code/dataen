import unittest
import numpy as np
import run_p2 as p


class BlendTests(unittest.TestCase):
    def test_blending_before_absolute_error_preserves_cancellation(self):
        # Opposite errors cancel; averaging their absolute errors would be wrong.
        truth=np.array([10.,20.]); early=np.array([6.,28.]); late=np.array([14.,12.])
        np.testing.assert_array_equal(p.blend(early,late,.5),truth)
        np.testing.assert_array_equal(p.blend(early,late,.75),[12.,16.])
        np.testing.assert_array_equal(p.blend(early,late,1.),late)

    def test_invalid_inputs_are_rejected(self):
        for a,b,w in [([1.],[1.,2.],.5),([np.nan],[1.],.5),([1.],[np.inf],.5),([1.],[2.],-1),([1.],[2.],np.nan)]:
            with self.assertRaises(ValueError):p.blend(a,b,w)

    def test_selection_requires_full_grid_and_breaks_ties(self):
        rows=[dict(weight_17=w,weighted_group_mae=100.) for w in p.WEIGHTS]
        self.assertEqual(p.choose(rows)['weight_17'],1.)
        rows[0]['weighted_group_mae']-=0.5e-9
        self.assertEqual(p.choose(rows)['weight_17'],1.)
        rows[1]['weighted_group_mae']=90.
        self.assertEqual(p.choose(rows)['weight_17'],.75)
        for bad in [rows[:-1],rows[:-1]+[rows[0]],rows[:-1]+[dict(weight_17=1.,weighted_group_mae=np.nan)]]:
            with self.assertRaises(ValueError):p.choose(bad)


if __name__=='__main__':unittest.main()
