"""Five-model validation/robustness on the existing fixed 200-image sample."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
import gc
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, roc_curve
from inference_script import load_model, get_inference_transforms, EnsembleRunner
from robustness_test import perturb

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root',type=Path,required=True)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/reproduction/validation')
    parser.add_argument('--weights-dir',type=Path,default=ROOT/'weights')
    args=parser.parse_args()
    out=args.output_dir.resolve()
    if out == (ROOT/'outputs/final_validation').resolve():
        parser.error('Recorded evidence is immutable; choose a new output directory.')
    if (out/'protocol.json').exists():
        parser.error('Output already contains a run; choose a new --output-dir.')
    out.mkdir(parents=True,exist_ok=True)
    sample=pd.read_csv(ROOT/'outputs/robustness/sample_manifest.csv')
    val=pd.read_csv(ROOT/'data/val_split.csv');train=pd.read_csv(ROOT/'data/train_split.csv')
    assert not set(train.image_id)&set(val.image_id)
    assert set(sample.image_id)<=set(val.image_id) and sample.image_id.is_unique
    metadata=pd.read_csv(args.dataset_root/'train_metadata.csv').set_index('image_id')
    assert all(int(metadata.loc[r.image_id,'label'])==r.label for r in sample.itertuples())
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(4)
    config=json.loads((args.weights_dir/'ensemble_config.json').read_text(encoding='utf-8'))
    tf=get_inference_transforms(); rows=[]
    y=sample.label.to_numpy()
    rng=np.random.default_rng(42)
    indices=[np.concatenate([rng.choice(np.flatnonzero(y==c),int((y==c).sum()),replace=True) for c in (0,1)]) for _ in range(2000)]
    clean=None
    conditions=['clean','jpeg_q50','jpeg_q70','blur_sigma1.5','noise_sigma15']
    blended={condition:np.zeros(len(sample),dtype=np.float32) for condition in conditions}
    with torch.inference_mode():
      for info in config['models']:
        single=load_model(str(args.weights_dir/info['file']),info['arch'],device)
        model=EnsembleRunner([(single,1.0)],device)
        for condition in conditions:
            probs=[]
            for start in range(0,len(sample),args.batch_size):
                tensors=[]
                for row in sample.iloc[start:start+args.batch_size].itertuples():
                    path=args.dataset_root/'train'/('fake' if row.label else 'real')/(row.image_id+'.jpg')
                    array=np.asarray(Image.open(path).convert('RGB'))
                    tensors.append(tf(Image.fromarray(perturb(array,condition,row.image_id))))
                probs.extend(model(torch.stack(tensors).to(device)).cpu().tolist())
                print(f'[FINAL VALIDATION] {info["name"]} / {condition}: {min(start+args.batch_size,len(sample))}/{len(sample)}',flush=True)
            blended[condition]+=info['weight']*np.asarray(probs,dtype=np.float32)
        del model,single
        gc.collect()
        if device=='cuda':torch.cuda.empty_cache()
      for condition in conditions:
            p=blended[condition]
            if condition=='clean':clean=p.copy()
            auc=roc_auc_score(y,p)
            boot=np.array([roc_auc_score(y[ix],p[ix]) for ix in indices])
            drops=np.array([roc_auc_score(y[ix],clean[ix])-roc_auc_score(y[ix],p[ix]) for ix in indices])
            tn,fp,fn,tp=confusion_matrix(y,p>=.5,labels=[0,1]).ravel()
            lo,hi=np.quantile(boot,[.025,.975]);dl,dh=np.quantile(drops,[.025,.975])
            rows.append(dict(condition=condition,n=len(y),auc=auc,auc_ci_low=lo,auc_ci_high=hi,
                             auc_drop=roc_auc_score(y,clean)-auc,drop_ci_low=dl,drop_ci_high=dh,
                             f1=f1_score(y,p>=.5),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp)))
            frame=sample[['image_id','label']].copy();frame['confidence']=p
            frame.to_csv(out/f'predictions_{condition}.csv',index=False)
            pd.DataFrame(rows).to_csv(out/'statistics.csv',index=False)
    (out/'protocol.json').write_text(json.dumps(dict(models=5,sample=200,seed=42,bootstrap=2000,
        device=device,precision='float32',tta='original+horizontal_flip',
        limitation='Fixed validation subset, not full validation or hidden test. FP32 validation; submission default inference uses FP32.'),indent=2))
    import subprocess,sys
    print('Fresh validation evidence:',out)


if __name__=='__main__':main()
