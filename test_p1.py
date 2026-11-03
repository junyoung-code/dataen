import unittest
import numpy as np
import run_p1 as p


class SelectionTests(unittest.TestCase):
    def rows(self):
        return [dict(trees=n,learning_rate=lr,weighted_group_mae=100.) for n,lr in p.CONFIGS]

    def test_full_grid_and_tie_break(self):
        rows=self.rows()
        self.assertEqual(len(rows),6)
        self.assertEqual(p.choose(rows)['trees'],50)
        self.assertEqual(p.choose(rows)['learning_rate'],0.06)
        rows[0]['weighted_group_mae']-=0.5e-9
        self.assertEqual(p.choose(rows)['learning_rate'],0.06)

    def test_best_score_can_choose_larger_model(self):
        rows=self.rows();rows[-1]['weighted_group_mae']=90.
        self.assertEqual(p.choose(rows)['trees'],600)

    def test_incomplete_duplicate_nonfinite_grid_rejected(self):
        rows=self.rows()
        for bad in [rows[:-1],rows[:-1]+[rows[0]]]:
            with self.assertRaises(ValueError):p.choose(bad)
        rows[0]['weighted_group_mae']=np.nan
        with self.assertRaises(ValueError):p.choose(rows)


if __name__=='__main__':unittest.main()
