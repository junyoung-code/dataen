"""Tests of leakage boundaries, loss masking, and neural model persistence."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from neural_models import Scaler, make_model, predictions, weighted_loss
from neural_data import daily_starts, sequence_indices, row_indices, Batches
from run_neural import choose


class NeuralTests(unittest.TestCase):
    def test_preprocessing_does_not_refit_on_future(self):
        train=np.array([[1.,np.nan],[3.,4.],[5.,6.]],dtype='float32')
        scaler=Scaler().fit(train)
        before=[v.copy() for v in (scaler.median,scaler.mean,scaler.std)]
        future=np.array([[10000.,np.nan]],dtype='float32')
        self.assertTrue(np.isfinite(scaler.transform(future)).all())
        for actual,expected in zip((scaler.median,scaler.mean,scaler.std),before):
            np.testing.assert_array_equal(actual,expected)
        self.assertEqual(scaler.median[1],5.)

    def test_sequence_never_crosses_sites_or_target_day(self):
        starts=daily_starts(3,24*5,24,24*5)
        ix=sequence_indices(starts)
        np.testing.assert_array_equal(ix[:,0]//120,ix[:,-1]//120)
        np.testing.assert_array_equal(ix[:,24],starts)
        np.testing.assert_array_equal(ix[:,-1],starts+23)
        self.assertTrue((starts%24==0).all())
        with self.assertRaises(ValueError):daily_starts(3,120,0,120)

    def test_raw_row_alignment_keeps_missing_labels_missing(self):
        axis=pd.Timestamp('2025-01-01 01:00',tz='Asia/Seoul')
        rows=pd.DataFrame({'site_id':['B','A','B'], 'ts':[axis+pd.Timedelta(hours=h) for h in (25,24,47)]})
        idx=row_indices(rows,['A','B'],axis,72)
        b=object.__new__(Batches);b.family='lstm';b.eval_starts=np.array([24,48,96,120]);b.eval_rows=idx
        pred=np.concatenate([np.arange(s,s+24) for s in b.eval_starts]).reshape(-1,24)
        np.testing.assert_array_equal(b.align(pred),idx)
        self.assertEqual(len(b.align(pred)),len(rows))

    def test_targets_not_inputs(self):
        b=object.__new__(Batches);b.family='lstm'
        b.x=np.arange(72*2,dtype='float32').reshape(72,2)
        b.y=np.zeros(72,dtype='float32');b.w=np.ones(72,dtype='float32');b.train_starts=np.array([24])
        first=b.training([0])[0]
        b.y[:]=1e6
        np.testing.assert_array_equal(b.training([0])[0],first)

    def test_masked_labels_have_no_gradient(self):
        pred=torch.tensor([[.1,.2]],requires_grad=True)
        y=torch.tensor([[.9,1e6]]);w=torch.tensor([[3.,0.]])
        weighted_loss(pred,y,w,'lstm').backward()
        np.testing.assert_allclose(pred.grad.numpy(),[[-1.,0.]])

    def test_tabm_optimizes_members_before_averaging(self):
        p=torch.tensor([[0.,2.]],requires_grad=True)
        loss=weighted_loss(p,torch.tensor([1.]),torch.tensor([1.]),'tabm')
        self.assertEqual(float(loss.detach()),1.)

    def test_weighted_loss_capacity_effect(self):
        p=torch.tensor([0.,0.],requires_grad=True)
        weighted_loss(p,torch.tensor([1.,-1.]),torch.tensor([9.,1.]),'lstm').backward()
        np.testing.assert_allclose(p.grad.numpy(),[-.9,.1])

    def test_selection_uses_tune_only(self):
        rows=[dict(window='tune',epochs=e,weighted_group_mae=s) for e,s in [(5,4),(10,2),(20,3)]]
        self.assertEqual(choose(rows)['epochs'],10)
        rows[1]['window']='confirmation'
        with self.assertRaises(ValueError):choose(rows)

    def test_architectures_learn_and_reload(self):
        torch.set_num_threads(2)
        columns=['weather','hour','cap','lat','lon','wa_dist','wb_dist']
        for family in ('tabm','lstm','tft'):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as tmp:
                torch.manual_seed(1)
                model=make_model(family,columns).to('cpu')
                x=torch.randn((8,len(columns)) if family=='tabm' else (8,48,len(columns)))
                target=torch.full((8,) if family=='tabm' else (8,24),.5)
                opt=torch.optim.AdamW(model.parameters(),lr=.01)
                for _ in range(4):
                    opt.zero_grad();loss=weighted_loss(predictions(model,x,family),target,torch.ones_like(target),family)
                    self.assertTrue(torch.isfinite(loss));loss.backward();opt.step()
                model.eval()
                expected=predictions(model,x,family).detach()
                path=Path(tmp)/'weights.pt';torch.save(model.state_dict(),path)
                restored=make_model(family,columns).to('cpu')
                restored.load_state_dict(torch.load(path,weights_only=True));restored.eval()
                torch.testing.assert_close(predictions(restored,x,family),expected)


if __name__=='__main__':unittest.main()
