"""A1: fixed TabM10 then TFT20, annual F2/F3 W under frozen AB contract."""
import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import train_smoke as original
from annual_neural_data import CONTRACT_SHA256, check_contract, key_mask_digest, prepare
from neural_models import CONFIGS
from run_neural import fit_round
from run_peer_w import TZ, bounds, score, write_json

ROOT = Path(__file__).resolve().parent
FIXED_EPOCHS = {'tabm':10, 'tft':20}


def evaluate(folder, cache, family, fold, training):
    contract = check_contract()
    expected = contract['folds'][fold]
    output = pd.read_parquet(cache/fold/'r10/evaluation.parquet')
    other = pd.read_parquet(cache/fold/'r17/evaluation.parquet')
    pd.testing.assert_frame_equal(output, other)
    del other
    assert key_mask_digest(output) == expected['evaluation_key_mask_sha256']
    for r in (10, 17):
        raw = np.load(folder/f'prediction_round_{r}_e{FIXED_EPOCHS[family]:03}.npy')
        # P4 uses float64 model output times float32 cap. Promote neural outputs
        # before this multiplication to preserve the shared kWh restoration rule.
        output[f'pred_{r}'] = np.maximum(np.where(output.daylight, raw.astype('float64'), 0.), 0.)*output.capacity_kw.to_numpy(dtype='float32')
    output['ensemble'] = output.pred_10*.25+output.pred_17*.75
    sites = pd.read_csv(original.DATA/'sites.csv')
    prices = pd.read_csv(original.DATA/'train/smp.csv')
    prices.ts = pd.to_datetime(prices.ts).dt.tz_convert(TZ)
    _, _, c, d = bounds(fold)
    scores = {}
    for prediction in ('pred_10', 'pred_17', 'ensemble'):
        scores[prediction], agg = score(output, sites, prices, c, d, prediction=prediction)
        if prediction == 'ensemble':
            agg.to_csv(folder/'group_errors.csv', index=False)
    assert key_mask_digest(output) == expected['evaluation_key_mask_sha256']
    output.to_parquet(folder/'site_predictions.parquet', index=False, compression='zstd')
    W = scores['ensemble']['W']
    return dict(status='complete', fold=fold, family=family, epochs=FIXED_EPOCHS[family],
        config=CONFIGS[family], metrics=scores, baseline_W=expected['baseline_W'],
        change_percent=(W/expected['baseline_W']-1)*100,
        evaluation_key_mask_sha256=key_mask_digest(output), training=training,
        capacity_restoration='float64(normalized_prediction) * float32(capacity), matching P4',
        peak_process_memory_mib=original.peak_memory_mib())


def worker(args):
    started = time.perf_counter()
    training = {}
    for r in (10, 17):
        print(f'{args.family}/{args.fold}: annual cache r{r}', flush=True)
        meta = prepare(args.cache_dir/args.fold/f'r{r}', args.fold, r)
        print(f'cache verified; daylight_labels={meta["train_daylight_rows"]}; post_cutoff_labels=0', flush=True)
        training[str(r)] = fit_round(args.output_dir, args.cache_dir/args.fold/f'r{r}',
            args.family, r, [FIXED_EPOCHS[args.family]], args.device, args.threads)
    report = evaluate(args.output_dir, args.cache_dir, args.family, args.fold, training)
    report['total_seconds'] = time.perf_counter()-started
    write_json(args.output_dir/'summary.json', report)
    print(f'{args.family}/{args.fold} W={report["metrics"]["ensemble"]["W"]:.6f}; change={report["change_percent"]:+.3f}%', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--family', choices=FIXED_EPOCHS)
    parser.add_argument('--fold', choices=['F2', 'F3'])
    args = parser.parse_args()
    for path in (args.output_dir, args.cache_dir):
        path.resolve().relative_to((ROOT/'outputs').resolve())
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; no silent CPU fallback')
    check_contract()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.fold:
        assert args.family
        worker(args)
        return
    started = time.perf_counter()
    source = args.output_dir/'source'
    source.mkdir()
    names = ['annual_neural_data.py', 'run_annual_neural.py', 'verify_annual_neural.py',
             'test_annual_neural.py', 'neural_data.py', 'neural_models.py', 'run_neural.py',
             'run_peer_w.py', 'features_p3.py', 'training_weights.py', 'train_smoke.py',
             'AB_EVALUATION_CONTRACT.json', 'data-manifest.json']
    hashes = {name:original.sha256(ROOT/name) for name in names}
    for name in names:
        (source/name).write_bytes((ROOT/name).read_bytes())
    report = dict(status='running', protocol='peer-annual-w-causal-v1', epochs=FIXED_EPOCHS,
        contract_sha256=CONTRACT_SHA256, source_sha256=hashes,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        git_status=subprocess.check_output(['git','status','--short'], cwd=ROOT, text=True),
        versions={p:importlib.metadata.version(p) for p in ['torch','tabm','pytorch-forecasting','lightning','numpy','pandas','lightgbm','pyarrow']},
        python=platform.python_version(), device=args.device, threads=args.threads,
        machine=dict(chip=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'], text=True).strip(),
            memory_bytes=int(subprocess.check_output(['sysctl','-n','hw.memsize'], text=True)), macos=platform.mac_ver()[0]),
        historical_comparison_not_independent_final_test=True, results={})
    write_json(args.output_dir/'summary.json', report)
    try:
        original.self_test()
        original.check_data()
        report['data_manifest_verified'] = True
        for family in (['tabm', 'tft'] if args.family is None else [args.family]):
            report['results'][family] = {}
            for fold in ('F2', 'F3'):
                destination = args.output_dir/family/fold
                subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()),
                    '--family', family, '--fold', fold, '--device', args.device,
                    '--threads', str(args.threads), '--output-dir', str(destination),
                    '--cache-dir', str(args.cache_dir)], check=True, cwd=ROOT)
                report['results'][family][fold] = json.loads((destination/'summary.json').read_text())
                write_json(args.output_dir/'summary.json', report)
        assert all(original.sha256(ROOT/name) == digest for name, digest in hashes.items())
        report['status'] = 'trained_pending_independent_verification'
    except BaseException as error:
        report['status'] = 'failed'
        report['error'] = str(error)
        raise
    finally:
        report['total_seconds'] = time.perf_counter()-started
        write_json(args.output_dir/'summary.json', report)


if __name__ == '__main__':
    main()
