"""Annual sequence boundary and forecast-only GPU capability regressions."""
import unittest
import numpy as np
import pandas as pd
import torch

from annual_neural_data import bounded_starts
from neural_data import daily_starts, sequence_indices
from neural_models import Scaler, make_model, predictions, weighted_loss
from run_peer_w import TZ, bounds


class AnnualBoundaries(unittest.TestCase):
    def test_partial_last_day_keeps_17_masks_18_through_midnight(self):
        # 01:00..next 00:00: final decoder starts at index 48, 17:00 is 64.
        n_hours, n_sites, cutoff = 120, 2, 64
        w = np.zeros(n_sites*n_hours, dtype='float32')
        for site in range(n_sites):
            w[site*n_hours+24:site*n_hours+cutoff+1] = 1
        starts = bounded_starts(w, n_sites, n_hours, 24, cutoff)
        np.testing.assert_array_equal(starts, [24,48,144,168])
        decoders = starts[:,None]+np.arange(24)
        self.assertEqual(int((w[decoders] > 0).sum()), 82)
        for site in range(n_sites):
            last_day = site*n_hours+48+np.arange(24)
            self.assertEqual(float(w[last_day[16]]), 1.)
            self.assertFalse(w[last_day[17:]].any())
        context = sequence_indices(starts)
        self.assertEqual(context.shape, (4,48))
        self.assertTrue(np.all(context//n_hours == starts[:,None]//n_hours))

    def test_whole_day_helper_rejects_partial_without_explicit_rounding(self):
        with self.assertRaises(ValueError):
            daily_starts(2,120,24,65)

    def test_kst_fold_bounds_and_decoder_clock(self):
        for fold,year in [('F2',2023),('F3',2024)]:
            a,cutoff,c,d = bounds(fold)
            self.assertEqual(cutoff, pd.Timestamp(f'{year}-05-31 17:00',tz=TZ))
            axis_start = a-pd.Timedelta(hours=24)
            last = int((cutoff-axis_start).total_seconds()/3600)
            self.assertEqual(last%24,16)
            decoder = pd.date_range(axis_start+pd.Timedelta(hours=(last//24)*24),periods=24,freq='h')
            self.assertEqual(decoder[0].hour,1)
            self.assertEqual(decoder[-1].hour,0)
            self.assertEqual(decoder[16],cutoff)
            self.assertEqual((d-c).total_seconds()/3600,8784 if fold=='F2' else 8760)

    def test_masked_future_labels_cannot_change_loss_or_training_scaler(self):
        pred=torch.zeros((2,24));target=torch.ones((2,24));w=torch.ones((2,24))
        w[:,17:]=0
        expected=weighted_loss(pred,target,w,'tft')
        target[:,17:]=1e6
        torch.testing.assert_close(weighted_loss(pred,target,w,'tft'),expected)
        grid=np.arange(144,dtype='float32').reshape(48,3)
        train=np.array([1,3,16,24,40])
        first=Scaler().fit(grid[train])
        grid[41:]=1e9
        second=Scaler().fit(grid[train])
        np.testing.assert_array_equal(first.mean,second.mean)


def mps_smoke():
    assert torch.backends.mps.is_available()
    torch.set_num_threads(2)
    columns=['weather','hour','cap','lat','lon','wa_dist','wb_dist']
    for family in ('tabm','tft'):
        torch.manual_seed(0)
        model=make_model(family,columns).to('mps')
        x=torch.randn((8,len(columns)) if family=='tabm' else (8,48,len(columns)),device='mps')
        target=torch.ones((8,) if family=='tabm' else (8,24),device='mps')*.5
        w=torch.ones_like(target)
        if family=='tft':w[:,17:]=0
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        opt.zero_grad();loss=weighted_loss(predictions(model,x,family),target,w,family)
        assert torch.isfinite(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step();torch.mps.synchronize()
        model.eval()
        with torch.inference_mode():
            assert torch.isfinite(predictions(model,x,family)).all()
        print(f'{family}: real MPS forward/backward passed',flush=True)


if __name__=='__main__':
    import sys
    if '--mps-smoke' in sys.argv:mps_smoke()
    else:unittest.main()
