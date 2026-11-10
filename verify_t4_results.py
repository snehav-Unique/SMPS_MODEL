"""Verify and import Colab evidence: python verify_t4_results.py /path/to/results."""
import argparse
import csv
import hashlib
import json
import shutil
import statistics
import subprocess
import zipfile
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('results',type=Path)
args=parser.parse_args()
source=args.results.resolve()
repo=Path(__file__).resolve().parent
manifest=json.loads((source/'manifest.json').read_text())
script=subprocess.check_output(['git','-C',str(repo),'-c',f'safe.directory={repo.as_posix()}',
    'show',manifest['git_commit']+':benchmark_t4.py'])
assert hashlib.sha256(script).hexdigest()==manifest['script_sha256'], 'Benchmark script hash mismatch'
assert manifest['status']=='complete' and 'T4' in manifest['gpu']
for item in manifest['weights']:
    h=hashlib.sha256()
    with (repo/item['path']).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    assert h.hexdigest()==item['sha256'], item['path']
with zipfile.ZipFile(repo/'datasets/hackathon_dataset.zip') as archive:
    expected={Path(n).stem for n in archive.namelist() if n.startswith('hackathon_dataset/test/images/') and n.endswith('.jpg')}
assert len(expected)==3000
with (source/'summary.csv').open() as stream: summary=list(csv.DictReader(stream))
assert len(summary)==manifest['runs']==3
checks=[]
probabilities=[]
for row in summary:
    folder=source/f"run_{int(row['run']):02d}"
    result=json.loads((folder/'result.json').read_text())
    with (folder/'predictions.csv').open() as stream: predictions=list(csv.DictReader(stream))
    with (folder/'batch_latency.csv').open() as stream: batches=list(csv.DictReader(stream))
    with (folder/'model_profile.csv').open() as stream: profiles=list(csv.DictReader(stream))
    assert len(predictions)==3000 and {p['image_id'] for p in predictions}==expected
    assert all(0<=float(p['confidence'])<=1 and int(p['label'])==int(float(p['confidence'])>=.5) for p in predictions)
    assert result['gpu']=='Tesla T4' and result['test_images']==3000 and result['precision']=='FP16'
    assert result['batch_size']==64 and result['workers']==2
    assert len(batches)==188 and len(profiles)==4
    assert abs(statistics.mean(float(b['gpu_compute_ms']) for b in batches)-result['gpu_batch_ms_mean'])<1e-6
    assert abs(3000/result['wall_seconds']-result['images_per_second_end_to_end'])<1e-9
    assert abs(float(row['wall_seconds'])-result['wall_seconds'])<1e-9
    assert all(sum(int(b['images']) for b in batches if b['model']==p['model'])==3000 for p in profiles)
    assert abs(max(float(p['peak_allocated_mib']) for p in profiles)-result['peak_allocated_mib'])<1e-6
    checkpoint=json.loads((folder/'checkpoint.json').read_text())
    assert checkpoint['models_finished']==4
    probabilities.append({p['image_id']:float(p['confidence']) for p in predictions})
    checks.append({'run':result['run'],'predictions':3000,'profiled_batches':188,
        'wall_seconds':result['wall_seconds'],'max_gpu_batch_ms':max(float(b['gpu_compute_ms']) for b in batches)})
max_diff=max(abs(p[k]-probabilities[0][k]) for p in probabilities[1:] for k in expected)
verification={'status':'passed','script_matches_committed_source':True,'all_four_weight_hashes_match':True,
    'all_test_ids_match_original_dataset':True,'runs':checks,'max_probability_difference_between_runs':max_diff,
    'median_wall_seconds':statistics.median(float(r['wall_seconds']) for r in summary),
    'all_runs_under_300_seconds':all(float(r['wall_seconds'])<300 for r in summary),
    'limitation':'Verification checks internal consistency of supplied evidence; it is not an independent GPU rerun.'}
destination=repo/'outputs'/'efficiency_t4'
destination.mkdir(parents=True,exist_ok=True)
for filename in ['REPORT.md','summary.csv','manifest.json','checkpoint.json','nvidia_smi.txt','benchmark_console.log']:
    shutil.copy2(source/filename,destination/filename)
for run in range(1,4): shutil.copytree(source/f'run_{run:02d}',destination/f'run_{run:02d}',dirs_exist_ok=True)
(destination/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
with (destination/'REPORT.md').open('a',encoding='utf-8') as stream:
    stream.write(f'\n## Evidence verification\n\nAll three runs contain 3,000 matching test IDs, finite probabilities, four completed models, and 188 profiled batches. Script and all four checkpoint hashes match the committed project. Timings and throughput were recomputed from the supplied logs. Maximum prediction difference between runs: {max_diff:.8g}. See verification.json. This verifies consistency of supplied Colab evidence; it is not an independent GPU rerun.\n')
print(json.dumps(verification,indent=2))
