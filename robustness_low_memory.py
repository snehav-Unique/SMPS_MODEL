"""Resume DenseNet using batch size 2 and on-demand image preprocessing."""
import json
import time
import numpy as np
import pandas as pd
import torch
from PIL import Image
import robustness_test as r

torch.set_num_threads(2)
torch.set_flush_denormal(True)
torch.backends.mkldnn.enabled = True
manifest = json.loads((r.OUT / 'run_manifest.json').read_text())
sample = pd.read_csv(r.OUT / 'sample_manifest.csv')
assert r.digest(r.OUT / 'sample_manifest.csv') == manifest['sample_sha256']
dataset = r.Path(manifest['dataset_path'])
for row in sample.itertuples():
    p = dataset / 'train' / ('fake' if row.label else 'real') / (row.image_id+'.jpg')
    assert r.digest(p) == manifest['sample_image_sha256'][row.image_id]
name = 'densenet121'
architecture, weights, _ = r.MODELS[name]
assert r.digest(r.ROOT / weights) == manifest['models'][name]['sha256']
model = r.timm.create_model(architecture, pretrained=False, num_classes=1)
model.load_state_dict(torch.load(r.ROOT / weights, map_location='cpu', weights_only=True), strict=True)
model.eval()
r.atomic_json(r.OUT / 'low_memory_resume.json', {'model':name,'batch_size':2,'threads':2,'layout':'contiguous_nchw',
    'device':'cpu','precision':'float32','mkldnn':True,'flush_denormal':True,'script_sha256':r.digest(r.Path(__file__)),
    'note':'Earlier DenseNet batches retained; same preprocessing and original+flip TTA.'})
r.checkpoint('low_memory_resume', {'model':name,'batch_size':2,'threads':2,'mkldnn':True,'flush_denormal':True,'layout':'contiguous_nchw'})
mean = torch.tensor([.485,.456,.406])[None,:,None,None]
std = torch.tensor([.229,.224,.225])[None,:,None,None]
for condition in r.CONDITIONS:
    path = r.OUT / f'predictions_{name}_{condition}.csv'
    prev = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=['image_id','true_label','confidence'])
    assert prev.image_id.tolist() == sample.image_id.tolist()[:len(prev)]
    rows = prev.to_dict('records')
    start_time = time.perf_counter()
    with torch.inference_mode():
        for start in range(len(rows),len(sample),2):
            arrays=[]
            for row in sample.iloc[start:start+2].itertuples():
                p=dataset/'train'/('fake' if row.label else 'real')/(row.image_id+'.jpg')
                a=np.asarray(Image.open(p).convert('RGB'))
                arrays.append(r.perturb(a,condition,row.image_id))
            x=((torch.from_numpy(np.stack(arrays)).permute(0,3,1,2).float()/255-mean)/std).contiguous()
            probs=((torch.sigmoid(model(x)).flatten()+torch.sigmoid(model(x.flip(-1))).flatten())/2).numpy()
            assert np.isfinite(probs).all() and ((probs>=0)&(probs<=1)).all()
            for row,prob in zip(sample.iloc[start:start+2].itertuples(),probs):
                rows.append({'image_id':row.image_id,'true_label':row.label,'confidence':float(prob)})
            tmp=path.with_suffix('.tmp'); pd.DataFrame(rows).to_csv(tmp,index=False); tmp.replace(path)
            if len(rows)%8==0 or len(rows)==200:
                r.checkpoint('04_batch_saved',{'model':name,'condition':condition,'images':len(rows),'total':200,'batch_size':2})
    r.checkpoint('04_condition_complete',{'model':name,'condition':condition,'seconds_this_run':time.perf_counter()-start_time})
r.summarize(sample)
report=r.OUT/'REPORT.md'
report.write_text(report.read_text(encoding='utf-8')+'\n## Runtime adjustment\n\nDenseNet resumed after unusually slow CPU execution with batch size 2 and two threads, with images prepared on demand, contiguous NCHW layout, MKLDNN enabled, and denormal flushing enabled. Diagnostic runs also tried MKLDNN disabled and one thread; earlier completed batches were retained. The other models used batch size 8 and four threads. All inference used CPU FP32 with identical preprocessing and TTA; batch/backend changes can introduce small floating-point differences. low_memory_resume.json and progress.jsonl record these adjustments. Resume DenseNet and regenerate the report using `python SMPS/robustness_low_memory.py`, then run the model summary script.\n',encoding='utf-8')
