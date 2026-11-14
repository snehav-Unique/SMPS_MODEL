"""Verify saved blending, then optionally reproduce predictions from real weights.

Never overwrites predictions.csv. CPU timing is never presented as T4 timing.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs/reproduction/verification'


def compare(reference, actual, tolerance):
    for frame in (reference, actual):
        assert list(frame.columns) == ['image_id', 'label', 'confidence']
        assert len(frame) == 3000 and frame.image_id.nunique() == 3000
        assert not frame.isna().any().any()
        assert frame.confidence.between(0, 1).all() and frame.label.isin([0, 1]).all()
    assert set(reference.image_id) == set(actual.image_id), 'Image IDs differ'
    a = reference.set_index('image_id').sort_index()
    b = actual.set_index('image_id').sort_index()
    delta = np.abs(a.confidence.to_numpy() - b.confidence.to_numpy())
    labels = int((a.label != b.label).sum())
    return dict(passed=bool(delta.max() <= tolerance and labels == 0),
                max_probability_difference=float(delta.max()),
                mean_probability_difference=float(delta.mean()),
                label_mismatches=labels, tolerance=tolerance, images=3000)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--tolerance', type=float, default=1e-5)
    parser.add_argument('--weights-dir', type=Path, default=ROOT/'weights')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    OUT = args.output_dir.resolve()
    if OUT == (ROOT/'outputs/final_verification').resolve():
        parser.error('Recorded evidence is immutable; choose a new output directory.')
    if (OUT/'verification.json').exists():
        parser.error('Output already contains a run; choose a new --output-dir.')
    assert args.runs > 0 and args.batch_size > 0 and args.tolerance >= 0
    OUT.mkdir(parents=True, exist_ok=True)
    weights_dir=args.weights_dir.resolve()
    cfg = json.loads((weights_dir/'ensemble_config.json').read_text(encoding='utf-8'))
    sources = {
        'efficientnetv2_s': 'outputs/models/efficientnetv2_s/predictions_efficientnetv2_s.csv',
        'convnext_tiny': 'outputs/predictions/predictions_convnext.csv',
        'swin_tiny': 'outputs/models/swin_tiny/predictions_swin.csv',
        'maxvit_tiny': 'outputs/predictions/predictions_maxvit.csv',
        'densenet121': 'outputs/models/densenet121/predictions_densenet121.csv',
    }
    reference = pd.read_csv(ROOT/'predictions.csv')
    blended = reference[['image_id']].copy().set_index('image_id')
    blended['confidence'] = 0.0
    assert abs(sum(m['weight'] for m in cfg['models']) - 1) < 1e-8
    for m in cfg['models']:
        frame = pd.read_csv(ROOT/sources[m['name']]).set_index('image_id')
        assert frame.index.is_unique and set(frame.index) == set(blended.index)
        blended['confidence'] += m['weight'] * frame.loc[blended.index, 'confidence']
    blended['label'] = (blended.confidence >= .5).astype(int)
    blended['confidence'] = blended.confidence.round(6)
    blended = blended.reset_index()[['image_id', 'label', 'confidence']]
    audit = compare(pd.read_csv(ROOT/'outputs/predictions/predictions_original.csv') if (ROOT/'outputs/predictions/predictions_original.csv').exists() else reference, blended, 0)
    print('\n[1/3] SAVED FIVE-MODEL BLEND:', 'PASS' if audit['passed'] else 'FAIL', flush=True)
    print(json.dumps(audit, indent=2), flush=True)
    missing = []
    for m in cfg['models']:
        path = weights_dir/m['file']
        if not path.exists() or path.stat().st_size < 1024:
            missing.append(str(path))
    result = dict(saved_blend=audit, weights_to_csv_status='not_run',
                  missing_weights=missing, runs=[],
                  reference_sha256=hashlib.sha256((ROOT/'predictions.csv').read_bytes()).hexdigest(),
                  inference_sha256=hashlib.sha256((ROOT/'inference_script.py').read_bytes()).hexdigest())
    (OUT/'verification.json').write_text(json.dumps(result, indent=2))
    if missing or args.test_dir is None:
        result['weights_to_csv_status'] = 'blocked'
        result['reason'] = 'Download LFS weights and provide --test-dir with all 3000 official images.'
        print('\n[2/3] WEIGHTS-TO-CSV: BLOCKED (not verified)', flush=True)
        print(result['reason'], flush=True)
    else:
        result['device'] = 'recorded by inference worker'
        result['weight_sha256'] = {m['file']: hashlib.sha256((weights_dir/m['file']).read_bytes()).hexdigest() for m in cfg['models']}
        result['weights_to_csv_status'] = 'running'
        (OUT/'verification.json').write_text(json.dumps(result, indent=2))
        try:
            for i in range(args.runs):
                target = OUT/f'predictions_run_{i+1:02}.csv'
                metrics = OUT/f'metrics_run_{i+1:02}.json'
                command = [sys.executable, '-u', str(ROOT/'inference_script.py'),
                           '--test_dir', str(args.test_dir.resolve()), '--weights', str(weights_dir/'ensemble_config.json'),
                           '--output', str(target), '--batch_size', str(args.batch_size),
                           '--progress_json', str(OUT/'progress.json'), '--metrics_json', str(metrics), '--threads', str(args.threads)]
                print(f'\n[2/3] REAL INFERENCE RUN {i+1}/{args.runs}', flush=True)
                started = time.perf_counter()
                with (OUT/f'console_run_{i+1:02}.log').open('w', encoding='utf-8') as log:
                    proc = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    for line in proc.stdout:
                        print(line, end='', flush=True)
                        log.write(line)
                    if proc.wait():
                        raise RuntimeError('Inference failed; see saved console log')
                measured = json.loads(metrics.read_text(encoding='utf-8'))
                result['device'] = measured['device_name']
                check = compare(reference, pd.read_csv(target), args.tolerance)
                result['runs'].append(dict(run=i+1, process_wall_seconds=time.perf_counter()-started,
                                           reproduction=check, metrics=measured))
                result['weights_to_csv_status'] = 'passed' if all(r['reproduction']['passed'] for r in result['runs']) else 'failed'
                (OUT/'verification.json').write_text(json.dumps(result, indent=2))
        except Exception as error:
            result['weights_to_csv_status'] = 'failed'
            result['error'] = str(error)
    (OUT/'verification.json').write_text(json.dumps(result, indent=2))
    # Fresh evidence stays separate; the committed dashboard describes the recorded run.
    print('\n[3/3] Fresh evidence:', OUT, flush=True)
    print('Weights-to-CSV status:', result['weights_to_csv_status'].upper(), flush=True)
    return 0 if result['weights_to_csv_status'] == 'passed' and audit['passed'] else 2


if __name__ == '__main__':
    sys.exit(main())
