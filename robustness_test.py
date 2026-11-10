"""Reproducible, resumable 200-image paired robustness evaluation.
Run: python SMPS/robustness_test.py (from workspace root).
"""
import hashlib
import io
import json
import platform
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import timm
from PIL import Image
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs' / 'robustness'
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
CONDITIONS = ['clean', 'jpeg_q50', 'jpeg_q70', 'blur_sigma1.5', 'noise_sigma15']
MODELS = {
    'efficientnetv2_s': ('tf_efficientnetv2_s', 'efficientnetv2_s_results/efficientnetv2_s_results/efficientnetv2_s_best.pth', .35),
    'convnext_tiny': ('convnext_tiny', 'weights/convnext_tiny_best.pth', .30),
    'swin_tiny': ('swin_tiny_patch4_window7_224', 'swin_results/swin_results/swin_tiny_best.pth', .20),
    'densenet121': ('densenet121', 'densenet121_results/densenet121_results/densenet121_best.pth', .15),
}

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def atomic_json(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2), encoding='utf-8')
    tmp.replace(path)

def checkpoint(step, details):
    event = {'utc': datetime.now(timezone.utc).isoformat(), 'step': step, 'details': details}
    with (OUT / 'progress.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(event) + '\n')
    atomic_json(OUT / 'checkpoint.json', event)
    print(step, details, flush=True)

def prepare():
    archive = ROOT.parent / 'Deepfake ML Hackathon - Participant Package' / 'hackathon_dataset.zip'
    dataset = archive.parent / 'extracted' / 'hackathon_dataset'
    existing = list(archive.parent.rglob('train_metadata.csv'))
    if existing and not archive.exists():
        dataset = existing[0].parent
    else:
        with zipfile.ZipFile(archive) as z:
            for entry in z.infolist():
                target = (archive.parent / 'extracted' / entry.filename).resolve()
                assert target.is_relative_to((archive.parent / 'extracted').resolve())
                if not entry.is_dir() and (not target.exists() or target.stat().st_size != entry.file_size):
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(z.read(entry))
    val = pd.read_csv(ROOT / 'data/val_split.csv')
    train = pd.read_csv(ROOT / 'data/train_split.csv')
    assert val.image_id.is_unique and train.image_id.is_unique
    assert set(val.label) == {0, 1}
    assert not set(val.image_id) & set(train.image_id)
    metadata = pd.read_csv(dataset / 'train_metadata.csv').set_index('image_id')
    for row in val.itertuples():
        assert int(metadata.loc[row.image_id, 'label']) == row.label
        image_path = dataset / 'train' / ('fake' if row.label else 'real') / (row.image_id + '.jpg')
        assert image_path.exists(), f'Missing validation image: {image_path}'
    sample = pd.concat([val[val.label == c].sort_values('image_id').sample(n=100, random_state=SEED)
                        for c in [0, 1]]).sort_values('image_id').reset_index(drop=True)
    sample.to_csv(OUT / 'sample_manifest.csv', index=False)
    checkpoint('01_dataset_and_sample_verified', {'validation': len(val), 'train': len(train),
               'sample_real': 100, 'sample_fake': 100, 'train_val_overlap': 0})
    return dataset, sample

def perturb(a, condition, image_id):
    if condition.startswith('jpeg'):
        b = io.BytesIO()
        Image.fromarray(a).save(b, format='JPEG', quality=int(condition[-2:]), subsampling=2)
        return np.asarray(Image.open(io.BytesIO(b.getvalue())).convert('RGB'))
    if condition.startswith('blur'):
        return cv2.GaussianBlur(a, (0, 0), sigmaX=1.5, sigmaY=1.5, borderType=cv2.BORDER_REFLECT_101)
    if condition.startswith('noise'):
        seed = int(hashlib.sha256(f'{SEED}:{image_id}'.encode()).hexdigest()[:16], 16)
        return np.clip(a.astype(np.float32) + np.random.default_rng(seed).normal(0, 15, a.shape), 0, 255).astype(np.uint8)
    return a

def inputs(dataset, sample):
    arrays = []
    for row in sample.itertuples():
        p = dataset / 'train' / ('fake' if row.label else 'real') / (row.image_id + '.jpg')
        a = np.asarray(Image.open(p).convert('RGB'))
        assert a.shape == (224, 224, 3), (p, a.shape)
        arrays.append(a)
    result = {}
    for c in CONDITIONS:
        a = np.stack([perturb(a, c, row.image_id) for a, row in zip(arrays, sample.itertuples())])
        # Images are already 224x224; ImageNet normalization matches notebook validation.
        result[c] = ((torch.from_numpy(a.copy()).permute(0, 3, 1, 2).float() / 255
                      - torch.tensor([.485, .456, .406])[None, :, None, None])
                     / torch.tensor([.229, .224, .225])[None, :, None, None])
        Image.fromarray(a[0]).save(OUT / f'example_{c}.png')
    checkpoint('02_perturbations_prepared', {'conditions': CONDITIONS, 'seed': SEED})
    return result

def infer(sample, tensors, device):
    for name, (architecture, weights, _) in MODELS.items():
        model = timm.create_model(architecture, pretrained=False, num_classes=1)
        state = torch.load(ROOT / weights, map_location='cpu', weights_only=True)
        if 'state_dict' in state: state = state['state_dict']
        elif 'model' in state: state = state['model']
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        checkpoint('03_model_loaded', {'model': name, 'strict_load': True})
        for c in CONDITIONS:
            path = OUT / f'predictions_{name}_{c}.csv'
            previous = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=['image_id', 'true_label', 'confidence'])
            assert previous.image_id.tolist() == sample.image_id.tolist()[:len(previous)]
            rows = previous.to_dict('records')
            started = time.perf_counter()
            with torch.inference_mode():
                for start in range(len(rows), len(sample), 8):
                    x = tensors[c][start:start+8].to(device)
                    # Two-view original + horizontal-flip TTA, as used in saved test predictions.
                    p = (torch.sigmoid(model(x)).flatten() + torch.sigmoid(model(x.flip(-1))).flatten()) / 2
                    probabilities = p.cpu().numpy()
                    assert np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()
                    for row, prob in zip(sample.iloc[start:start+8].itertuples(), probabilities):
                        rows.append({'image_id': row.image_id, 'true_label': row.label, 'confidence': float(prob)})
                    tmp = path.with_suffix('.tmp')
                    pd.DataFrame(rows).to_csv(tmp, index=False)
                    tmp.replace(path)
                    checkpoint('04_batch_saved', {'model': name, 'condition': c, 'images': len(rows), 'total': len(sample)})
            checkpoint('04_condition_complete', {'model': name, 'condition': c, 'seconds_this_run': time.perf_counter()-started})
        del model

def summarize(sample):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    y = sample.label.to_numpy()
    scores = {}
    for c in CONDITIONS:
        scores[c] = np.zeros(len(y))
        for name, (_, _, weight) in MODELS.items():
            df = pd.read_csv(OUT / f'predictions_{name}_{c}.csv')
            assert df.image_id.tolist() == sample.image_id.tolist() and df.true_label.tolist() == y.tolist()
            scores[c] += weight * df.confidence.to_numpy()
        df = sample.rename(columns={'label': 'true_label'}).copy()
        df['confidence'] = scores[c]
        df['predicted_label'] = (scores[c] >= .5).astype(int)
        df.to_csv(OUT / f'predictions_ensemble_{c}.csv', index=False)
    rng = np.random.default_rng(SEED)
    boot = np.stack([np.concatenate([rng.choice(np.where(y == k)[0], 100, replace=True) for k in [0, 1]]) for _ in range(2000)])
    clean_boot = np.array([roc_auc_score(y[i], scores['clean'][i]) for i in boot])
    clean_auc = roc_auc_score(y, scores['clean'])
    rows = []
    for c, p in scores.items():
        pred = p >= .5
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        auc = roc_auc_score(y, p)
        auc_boot = np.array([roc_auc_score(y[i], p[i]) for i in boot])
        ci = np.quantile(auc_boot, [.025, .975])
        dci = np.quantile(clean_boot - auc_boot, [.025, .975])
        rows.append(dict(condition=c, n=len(y), auc=auc, auc_ci_low=ci[0], auc_ci_high=ci[1],
            auc_drop=clean_auc-auc, drop_ci_low=dci[0], drop_ci_high=dci[1],
            relative_auc_drop_pct=100*(clean_auc-auc)/clean_auc,
            accuracy=accuracy_score(y,pred), precision=precision_score(y,pred,zero_division=0),
            recall=recall_score(y,pred), f1=f1_score(y,pred), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
            observed_drop_below_002=bool(clean_auc-auc < .02)))
    stats = pd.DataFrame(rows)
    stats.to_csv(OUT / 'statistics.csv', index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(stats.condition, stats.auc, color=['#2563eb']+['#0891b2']*4)
    ax.errorbar(range(5), stats.auc, yerr=[stats.auc-stats.auc_ci_low, stats.auc_ci_high-stats.auc], fmt='none', color='black', capsize=4)
    ax.axhline(clean_auc-.02, color='#dc2626', linestyle='--', label='Clean AUC minus 0.02')
    ax.set(ylim=(0,1), ylabel='ROC AUC', title='Paired robustness test: 200 validation images (95% bootstrap CI)')
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / 'auc_comparison.png', dpi=180); plt.close(fig)
    table = '| Condition | AUC (95% CI) | AUC drop | Accuracy | Precision | Recall | F1 | TN / FP / FN / TP | <0.02 drop? |\n|---|---|---|---|---|---|---|---|---|\n'
    for r in stats.itertuples():
        table += f'| {r.condition} | {r.auc:.4f} ({r.auc_ci_low:.4f}–{r.auc_ci_high:.4f}) | {r.auc_drop:.4f} | {r.accuracy:.3f} | {r.precision:.3f} | {r.recall:.3f} | {r.f1:.3f} | {r.tn} / {r.fp} / {r.fn} / {r.tp} | {"Yes" if r.observed_drop_below_002 else "No"} |\n'
    worst = stats.iloc[1:].sort_values('auc_drop', ascending=False).iloc[0]
    report = f'''# Deepfake detector robustness report

## Findings

Clean ensemble AUC: **{clean_auc:.4f}**. Largest observed degradation: **{worst['condition']}**, with AUC **{worst['auc']:.4f}** and absolute drop **{worst['auc_drop']:.4f}**.
{int(stats.iloc[1:].observed_drop_below_002.sum())} of 4 corruption conditions meet the observed absolute AUC-drop target of less than 0.02.

{table}

## Method

200 images (100 real, 100 fake), sampled without replacement from the existing 1,400-image validation split; seed 42. All validation IDs were checked against archive metadata and checked for no ID overlap with the 5,600-image training split. The same images are paired across all conditions. No models were trained and no thresholds or ensemble weights were tuned on these results.

Four-model probability ensemble: EfficientNetV2-S 0.35, ConvNeXt-Tiny 0.30, Swin-Tiny 0.20, DenseNet-121 0.15. Strict checkpoint loading; evaluation mode; FP32; original and horizontal-flip probability averaging; RGB 224x224; ImageNet normalization. Classification threshold 0.5; positive class is fake. Environment and SHA-256 weight hashes are in run_manifest.json.

Each corruption is independently applied to the decoded original: JPEG quality 50/70 with 4:2:0 subsampling; OpenCV Gaussian blur sigma=1.5 with automatic kernel and reflect-101 borders; independent Gaussian pixel noise sigma=15 on the 0–255 scale, clipped and cast to uint8. Noise is deterministic per image. Corruptions are not stacked.

95% percentile intervals use 2,000 stratified bootstrap resamples, paired between clean and degraded images. statistics.csv includes confidence intervals for the AUC drops and relative percentage drops. The target is interpreted as an **absolute drop below 0.02**, not a relative 2% decrease.

## Interpretation and limitations

These are measured validation-sample results, not hidden-test guarantees. A 200-image sample has sampling uncertainty; meeting the observed target does not establish that the true drop is below 0.02. Models were selected using this validation split, so these are not independent final-test estimates. ID separation does not establish identity/source separation. Noise uses one fixed realization per image. No T4 speed or VRAM score is claimed from this CPU run. The handover's projected ensemble AUC and robustness claims were not treated as measured results.

## Files and resuming

- statistics.csv: full metrics and uncertainty estimates.
- predictions_*.csv: all model and ensemble per-image probabilities.
- sample_manifest.csv: exact selected IDs and ground truth.
- auc_comparison.png: AUC comparison with confidence intervals.
- example_*.png: one image under each condition.
- progress.jsonl and checkpoint.json: append-only progress history and latest checkpoint.
- run_manifest.json: reproducibility settings and hashes.

Resume or regenerate the report by running `python SMPS/robustness_test.py` from the workspace root. Completed prediction batches are reused only when the run signature matches.
'''
    (OUT / 'REPORT.md').write_text(report, encoding='utf-8')
    checkpoint('05_statistics_and_report_complete', {'clean_auc': clean_auc, 'worst_condition': worst['condition'], 'worst_auc_drop': float(worst['auc_drop'])})
    print(stats.to_string(index=False), flush=True)

def main():
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    dataset, sample = prepare()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    manifest = {'protocol_version': 1, 'seed': SEED, 'device': device, 'precision': 'float32', 'tta': 'original+horizontal_flip',
        'batch_size': 8, 'threads': 4, 'conditions': CONDITIONS, 'threshold': .5,
        'python': platform.python_version(), 'torch': torch.__version__, 'timm': timm.__version__,
        'numpy': np.__version__, 'opencv': cv2.__version__, 'pillow': Image.__version__,
        'sample_sha256': digest(OUT / 'sample_manifest.csv'), 'script_sha256': digest(Path(__file__)),
        'dataset_path': str(dataset), 'sample_image_sha256': {r.image_id: digest(dataset / 'train' / ('fake' if r.label else 'real') / (r.image_id+'.jpg')) for r in sample.itertuples()},
        'models': {name: {'architecture': a, 'path': p, 'weight': w, 'sha256': digest(ROOT / p)} for name, (a,p,w) in MODELS.items()}}
    manifest_path = OUT / 'run_manifest.json'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, 'Run signature changed; use a new output folder.'
    atomic_json(manifest_path, manifest)
    checkpoint('00_environment_recorded', {'device': device, 'torch': torch.__version__, 'timm': timm.__version__})
    tensors = inputs(dataset, sample)
    infer(sample, tensors, device)
    summarize(sample)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        checkpoint('ERROR', {'type': type(e).__name__, 'message': str(e)})
        raise
