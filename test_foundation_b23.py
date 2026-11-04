import unittest
import subprocess
import sys
import numpy as np
import pandas as pd
from foundation_b23 import select_context,train_medians,fill,key_hash,check_frozen

class FoundationChecks(unittest.TestCase):
 def test_runner_command_line_starts_before_loading_data(self):
  for script in ["run_foundation_b23.py","verify_foundation_b23.py"]:
   result=subprocess.run([sys.executable,script,"--help"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
   self.assertEqual(result.returncode,0,script)

 def test_capacity_context_fixed_order_and_weighted_selection(self):
  c=np.r_[np.ones(900),np.full(100,100.)]
  a=select_context(c,100,0);b=select_context(c,100,0)
  np.testing.assert_array_equal(a,b)
  self.assertEqual(len(np.unique(a)),100)
  self.assertTrue((np.diff(a)>0).all())
  self.assertGreater((a>=900).sum(),50)
  with self.assertRaises(ValueError):select_context([0,1],1,0)
 def test_imputation_fitted_only_training(self):
  train=np.array([[1,np.nan],[3,4],[5,6]],dtype='float32')
  med=train_medians(train);np.testing.assert_array_equal(med,[3,5])
  eval_data=np.array([[np.nan,1e9]],dtype='float32')
  np.testing.assert_array_equal(fill(eval_data,med),[[3,1e9]])
  np.testing.assert_array_equal(med,[3,5])
 def test_mask_hash_canonical_order_and_frozen_source(self):
  x=pd.DataFrame({'site_id':['x','y'],'ts':pd.date_range('2023-06-01',periods=2,tz='Asia/Seoul'),'has_peer':[True,False]})
  self.assertEqual(key_hash(x),key_hash(x.iloc[::-1]))
  y=x.copy();y.loc[1,'has_peer']=True
  self.assertNotEqual(key_hash(x),key_hash(y))
  self.assertEqual(check_frozen()['contract_id'],'ab-peer-annual-w-v1-20261104')

if __name__=='__main__':unittest.main()
