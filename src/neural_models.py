"""Untrained official TabM/TFT and a forecast-only encoder-decoder LSTM."""
import numpy as np
import torch
from torch import nn

CONFIGS = {
    'tabm': dict(n_blocks=3, d_block=128, k=8, dropout=0.1),
    'lstm': dict(hidden_size=64, layers=1),
    'tft': dict(hidden_size=16, hidden_continuous_size=8,
                attention_head_size=4, lstm_layers=1, dropout=0.1),
}
EPOCHS = (5, 10, 20)


class Scaler:
    """Fit only on observed daytime training rows; finite float32 output."""
    def fit(self, x):
        x = np.asarray(x, dtype=np.float32)
        self.median = np.nanmedian(x, axis=0)
        self.median = np.nan_to_num(self.median, nan=0.)
        filled = np.where(np.isfinite(x), x, self.median)
        self.mean = filled.mean(axis=0, dtype=np.float64).astype('float32')
        self.std = filled.std(axis=0, dtype=np.float64).astype('float32')
        self.std[self.std < 1e-6] = 1
        return self

    def transform(self, x):
        filled = np.where(np.isfinite(x), x, self.median)
        return np.clip((filled-self.mean)/self.std, -10, 10).astype('float32')


class ForecastLSTM(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        config = CONFIGS['lstm']
        self.encoder = nn.LSTM(n_features, config['hidden_size'], config['layers'], batch_first=True)
        self.decoder = nn.LSTM(n_features, config['hidden_size'], config['layers'], batch_first=True)
        self.head = nn.Linear(config['hidden_size'], 1)

    def forward(self, x):
        _, state = self.encoder(x[:, :24])
        hidden, _ = self.decoder(x[:, 24:], state)
        return self.head(hidden).squeeze(-1)


class ForecastTFT(nn.Module):
    def __init__(self, columns):
        super().__init__()
        from pytorch_forecasting import TemporalFusionTransformer
        from pytorch_forecasting.metrics import MAE
        from pytorch_forecasting.data import TorchNormalizer
        static = [c for c in columns if c in ('cap', 'lat', 'lon', 'wa_dist', 'wb_dist')]
        dynamic = [c for c in columns if c not in static]
        self.net = TemporalFusionTransformer(
            **CONFIGS['tft'], max_encoder_length=24, output_size=1, loss=MAE(),
            output_transformer=TorchNormalizer(method='identity'),
            x_reals=list(columns), static_reals=static,
            time_varying_reals_encoder=dynamic, time_varying_reals_decoder=dynamic,
            learning_rate=0.001, causal_attention=True, log_interval=-1,
        )

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # LightningModule tracks its device separately from tensor placement.
        # Moving only the enclosing nn.Module leaves attention masks on CPU.
        self.net.to(*args, **kwargs)
        return self

    def forward(self, x):
        n = len(x)
        lengths = torch.full((n,), 24, dtype=torch.long, device=x.device)
        empty = torch.empty((n, 24, 0), dtype=torch.long, device=x.device)
        scale = torch.zeros((n, 2), device=x.device)
        scale[:, 1] = 1
        return self.net(dict(encoder_cont=x[:, :24], decoder_cont=x[:, 24:],
            encoder_cat=empty, decoder_cat=empty, encoder_lengths=lengths,
            decoder_lengths=lengths, target_scale=scale))['prediction'].squeeze(-1)


def make_model(family, columns):
    if family == 'tabm':
        from tabm import TabM
        return TabM.make(n_num_features=len(columns), d_out=1, **CONFIGS[family])
    if family == 'lstm':
        return ForecastLSTM(len(columns))
    if family == 'tft':
        return ForecastTFT(columns)
    raise ValueError(f'Unknown family: {family}')


def predictions(model, x, family):
    out = model(x)
    return out.squeeze(-1) if family == 'tabm' else out


def weighted_loss(pred, target, weight, family):
    # TabM learns each member separately, then averages members at inference.
    if family == 'tabm':
        error = (pred-target[:, None]).abs().mean(dim=1)
    else:
        error = (pred-target).abs()
    # Missing/night targets carry zero weight and never contribute a label.
    return (error*weight).sum()/weight.sum().clamp_min(1e-12)
