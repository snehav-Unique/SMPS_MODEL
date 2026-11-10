"""T4-only, fresh-process benchmark of the four-model, two-view ensemble.

Example: python benchmark_t4.py --test-dir datasets/extracted/hackathon_dataset/test/images
Never substitutes CPU timing. Setup/download/extraction are outside the timed runs.
"""
import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = [
    ('efficientnetv2_s', 'tf_efficientnetv2_s', 'efficientnetv2_s_results/efficientnetv2_s_results/efficientnetv2_s_best.pth', .35),
    ('convnext_tiny', 'convnext_tiny', 'weights/convnext_tiny_best.pth', .30),
    ('swin_tiny', 'swin_tiny_patch4_window7_224', 'swin_results/swin_results/swin_tiny_best.pth', .20),
    ('densenet121', 'densenet121', 'densenet121_results/densenet121_results/densenet121_best.pth', .15),
]

def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    tmp.replace(path)

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def check_gpu(torch):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Select a T4 GPU runtime; CPU timings are not accepted.')
    name = torch.cuda.get_device_name(0)
    if 'T4' not in name.upper():
        raise RuntimeError(f'T4 required, received {name}. Change the Colab runtime to T4.')
    return name

def worker(args):
    # Imports are intentionally in the child process and inside its parent wall timer.
    import numpy as np
    import torch
    import timm
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    gpu = check_gpu(torch)
    torch.manual_seed(42)
    torch.set_num_threads(2)
    torch.set_flush_denormal(True)
    torch.backends.cudnn.benchmark = True
    paths = sorted(p for p in args.test_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    if len(paths) != 3000 or len({p.stem for p in paths}) != 3000:
        raise ValueError(f'Expected exactly 3,000 unique test image IDs, found {len(paths)} images.')
    transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

    class Images(Dataset):
        def __len__(self): return len(paths)
        def __getitem__(self, index):
            with Image.open(paths[index]) as image:
                return transform(image.convert('RGB'))

    # Colab/Linux workers fork; on other platforms require --workers 0.
    loader_options = {'multiprocessing_context':'fork'} if args.workers else {}
    if args.workers and sys.platform=='win32':
        raise RuntimeError('Use --workers 0 on Windows; the prepared Colab notebook uses Linux workers.')
    loader = DataLoader(Images(), batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers>0, **loader_options)
    confidence = np.zeros(3000, dtype=np.float64)
    batch_rows, model_rows = [], []
    prediction_start = time.perf_counter()
    for name, architecture, relative, weight in MODELS:
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        model = timm.create_model(architecture, pretrained=False, num_classes=1)
        state = torch.load(ROOT/relative, map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
        del state
        model.eval().to('cuda').half()
        torch.cuda.synchronize()
        offset = 0
        with torch.inference_mode():
            for batch_index, images in enumerate(loader):
                batch_start = time.perf_counter()
                images = images.to('cuda', non_blocking=True).half()
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                p = (model(images).sigmoid().float().flatten() + model(images.flip(-1)).sigmoid().float().flatten())/2
                end.record()
                probabilities = p.cpu().numpy()
                torch.cuda.synchronize()
                if not np.isfinite(probabilities).all() or not ((probabilities>=0)&(probabilities<=1)).all():
                    raise ValueError(f'Invalid FP16 probabilities from {name}')
                confidence[offset:offset+len(p)] += weight*probabilities
                offset += len(p)
                batch_rows.append({'model':name,'batch':batch_index+1,'images':len(p),
                    'gpu_compute_ms':begin.elapsed_time(end),
                    'batch_processing_ms_excludes_loader_wait':1000*(time.perf_counter()-batch_start)})
                if (batch_index+1)%10 == 0 or offset==3000:
                    print(f'{name}: {offset}/3000', flush=True)
            if offset != 3000: raise ValueError('Incomplete model inference')
        model_rows.append({'model':name,'seconds_including_load_and_loader_wait':time.perf_counter()-started,
            'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
            'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20})
        save_json(args.output/'checkpoint.json', {'stage':'model_complete','model':name,'models_finished':len(model_rows)})
        del model, images, p
        torch.cuda.empty_cache()
    prediction_path = args.output/'predictions.csv'
    with prediction_path.open('w',newline='') as stream:
        writer=csv.writer(stream); writer.writerow(['image_id','label','confidence'])
        writer.writerows((p.stem,int(prob>=.5),f'{prob:.8f}') for p,prob in zip(paths,confidence))
    for filename, rows in [('batch_latency.csv',batch_rows),('model_profile.csv',model_rows)]:
        with (args.output/filename).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    gpu_ms=[row['gpu_compute_ms'] for row in batch_rows]
    save_json(args.output/'worker_metrics.json', {'gpu':gpu,'test_images':3000,'precision':'FP16',
        'tta':'original and horizontal flip, probability average','batch_size':args.batch_size,
        'workers':args.workers,'inference_seconds_including_model_loads':time.perf_counter()-prediction_start,
        'gpu_batch_ms_mean':float(np.mean(gpu_ms)),'gpu_batch_ms_p50':float(np.median(gpu_ms)),
        'gpu_batch_ms_p95':float(np.percentile(gpu_ms,95)),
        'peak_allocated_mib':max(row['peak_allocated_mib'] for row in model_rows),
        'peak_reserved_mib':max(row['peak_reserved_mib'] for row in model_rows),
        'torch':torch.__version__,'timm':timm.__version__,'cuda':torch.version.cuda,
        'cudnn':torch.backends.cudnn.version(),'python':sys.version,'utc':datetime.now(timezone.utc).isoformat()})

def benchmark(args):
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'summary.csv').exists():
        raise ValueError('Output already contains a completed benchmark. Choose a fresh --output directory.')
    import torch
    gpu=check_gpu(torch)
    manifest={'status':'running','gpu':gpu,'runs':args.runs,'weights':[
        {'model':name,'weight':w,'path':p,'sha256':sha256(ROOT/p)} for name,_,p,w in MODELS],
        'script_sha256':sha256(Path(__file__)),'utc':datetime.now(timezone.utc).isoformat()}
    try: manifest['git_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    except (OSError,subprocess.CalledProcessError): manifest['git_commit']=None
    save_json(args.output/'manifest.json',manifest)
    rows=[]
    for run in range(1,args.runs+1):
        folder=args.output/f'run_{run:02d}'; folder.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,str(Path(__file__).resolve()),'--worker','--test-dir',str(args.test_dir.resolve()),
            '--output',str(folder.resolve()),'--batch-size',str(args.batch_size),'--workers',str(args.workers)]
        start=time.perf_counter()
        with (folder/'console.log').open('w',encoding='utf-8') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        elapsed=time.perf_counter()-start
        if result.returncode:
            save_json(args.output/'checkpoint.json',{'status':'failed','run':run,'returncode':result.returncode})
            raise RuntimeError(f'Run {run} failed; inspect {folder}/console.log. No passing score is claimed.')
        metrics=json.loads((folder/'worker_metrics.json').read_text())
        metrics.update({'run':run,'wall_seconds':elapsed,'images_per_second_end_to_end':3000/elapsed,
                        'under_5_minutes':elapsed<300})
        save_json(folder/'result.json',metrics)
        rows.append({key:metrics[key] for key in ['run','wall_seconds','images_per_second_end_to_end',
            'gpu_batch_ms_mean','gpu_batch_ms_p50','gpu_batch_ms_p95','peak_allocated_mib','peak_reserved_mib','under_5_minutes']})
        with (args.output/'summary.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        save_json(args.output/'checkpoint.json',{'status':'run_complete','runs_finished':run,'runs_requested':args.runs})
        print(f'Run {run}: {elapsed:.2f}s, {3000/elapsed:.2f} images/s',flush=True)
    times=[r['wall_seconds'] for r in rows]
    table='| Run | End-to-end seconds | Images/sec | GPU batch mean ms | GPU batch p95 ms | Peak allocated MiB | Peak reserved MiB | <300s |\n|---|---|---|---|---|---|---|---|\n'
    for r in rows:
        table+=f"| {r['run']} | {r['wall_seconds']:.2f} | {r['images_per_second_end_to_end']:.2f} | {r['gpu_batch_ms_mean']:.2f} | {r['gpu_batch_ms_p95']:.2f} | {r['peak_allocated_mib']:.1f} | {r['peak_reserved_mib']:.1f} | {r['under_5_minutes']} |\n"
    report=f'''# Measured NVIDIA T4 efficiency benchmark

GPU: {gpu}. Images per run: 3,000. Runs: {args.runs}. Median runtime: **{statistics.median(times):.2f} seconds**; range {min(times):.2f}–{max(times):.2f} seconds. All measured runs under 300 seconds: **{all(t<300 for t in times)}**.

{table}

## Measurement protocol

Each run starts a new Python process and performs all four model loads, FP16 inference, original+horizontal-flip TTA, fixed probability blending, and CSV output. Batch size {args.batch_size}; loader workers {args.workers}. No artificial warmup is excluded: first-batch initialization and cuDNN algorithm selection count in wall time. Downloads, ZIP extraction, hash checks, and parent setup are outside the timer. Filesystem caches are not flushed; later runs can benefit from warm caches. Profiling synchronization and log writing are included in runtime.

GPU batch times measure both TTA forward passes and probability averaging, excluding loading/preprocessing and host-to-device transfer. They are per model batch, not the latency of a complete four-model ensemble batch. Full-run throughput counts each of the 3,000 unique images once, not once per model or TTA view. PyTorch peak allocated/reserved VRAM excludes non-PyTorch allocations such as CUDA contexts; this is not a total-board memory measurement.

This instrumented ensemble runner is a reference pipeline. The repository's original inference_script.py remains a separate single-model CLI; final submission inference must be aligned with the selected pipeline and benchmarked before claiming the official efficiency score. Timing on Colab does not guarantee timing on the judges' system. No labels or accuracy claims are inferred from the unlabeled test set. FP16 predictions have not been assumed bitwise equivalent to the CPU FP32 robustness predictions.

## Evidence

summary.csv; manifest.json; checkpoint.json; per-run console.log, result.json, model_profile.csv, batch_latency.csv, and predictions.csv. Failed runs stop the benchmark and do not receive a passing claim. Re-running a measurement requires a fresh output directory; completed measurements are not silently reused.
'''
    (args.output/'REPORT.md').write_text(report,encoding='utf-8')
    save_json(args.output/'checkpoint.json',{'status':'complete','runs_finished':len(rows),'all_runs_under_300_seconds':all(t<300 for t in times)})
    manifest['status']='complete'; save_json(args.output/'manifest.json',manifest)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-dir',required=True,type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs'/'efficiency_t4')
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--runs',type=int,default=3)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.runs<1 or args.batch_size<1 or args.workers<0: parser.error('Invalid run/batch/worker count')
    args.output.mkdir(parents=True,exist_ok=True)
    (worker if args.worker else benchmark)(args)

if __name__=='__main__':
    main()
