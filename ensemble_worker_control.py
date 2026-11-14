"""Stdlib-only ensemble controller: isolate model memory in short-lived workers."""
import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def run_if_ensemble(argv):
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--weights')
    known,_=parser.parse_known_args(argv)
    if not known.weights:return None
    config_path=Path(known.weights)
    if config_path.is_dir():config_path=config_path/'ensemble_config.json'
    if config_path.suffix.lower()!='.json':return None
    parser.add_argument('--test_dir',required=True)
    parser.add_argument('--output',default='predictions.csv')
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--threshold',type=float,default=.5)
    parser.add_argument('--progress_json')
    parser.add_argument('--metrics_json')
    parser.add_argument('--fp16',action='store_true')
    parser.add_argument('--tta',action='store_true')
    parser.add_argument('--backbone')
    args=parser.parse_args(argv)
    if args.tta:parser.error('JSON ensemble already uses deterministic original+horizontal-flip TTA; omit --tta.')
    started=time.perf_counter()
    config=json.loads(config_path.read_text())
    assert abs(sum(m['weight'] for m in config['models'])-1)<1e-8
    assert all(m['weight']>=0 for m in config['models'])
    output=Path(args.output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    combined={};order=None;measurements=[]
    # Use output drive for scratch files; avoids the nearly full system TEMP drive.
    with tempfile.TemporaryDirectory(prefix='smps_workers_',dir=output.parent) as scratch:
      for index,info in enumerate(config['models']):
        target=Path(scratch)/(info['name']+'.csv');metrics=target.with_suffix('.json')
        print(f'\n[ENSEMBLE {index+1}/{len(config["models"])}] {info["name"]} | weight={info["weight"]}',flush=True)
        command=[sys.executable,'-u',str(Path(__file__).with_name('inference_script.py')),
                 '--ensemble_worker','--flip_tta','--test_dir',args.test_dir,
                 '--weights',str(config_path.parent/info['file']),'--backbone',info['arch'],
                 '--output',str(target),'--batch_size',str(args.batch_size),'--threads',str(args.threads),
                 '--metrics_json',str(metrics)]
        if args.progress_json:command+=['--progress_json',args.progress_json]
        subprocess.run(command,check=True)
        measurements.append(json.loads(metrics.read_text()))
        with target.open(newline='') as stream:rows=list(csv.DictReader(stream))
        ids=[r['image_id'] for r in rows]
        assert len(ids)==3000 and len(set(ids))==3000
        if order is None:
            order=ids;combined=dict.fromkeys(ids,0.)
        assert ids==order
        for row in rows:
            value=float(row['confidence']);assert math.isfinite(value) and 0<=value<=1
            combined[row['image_id']]+=info['weight']*value
    with output.open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['image_id','label','confidence'])
        for key in order:
            value=combined[key]
            writer.writerow([key,int(value>=args.threshold),f'{value:.6f}'])
    result=dict(stage='complete',processed=3000,total=3000,models=len(config['models']),
                device=measurements[0]['device'],device_name=measurements[0]['device_name'],
                seconds=time.perf_counter()-started,
                peak_allocated_mib=max((m['peak_allocated_mib'] or 0) for m in measurements) or None,
                per_model=measurements,execution='isolated sequential workers; original+horizontal_flip')
    if args.metrics_json:Path(args.metrics_json).write_text(json.dumps(result,indent=2))
    if args.progress_json:Path(args.progress_json).write_text(json.dumps(result))
    print(f'[ENSEMBLE COMPLETE] {len(order)} images; {result["seconds"]:.2f} seconds; {result["device_name"]}',flush=True)
    return 0
