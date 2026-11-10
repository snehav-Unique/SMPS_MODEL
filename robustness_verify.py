"""Verify completed artifacts and DenseNet CPU-runtime consistency."""
import json
import numpy as np
import pandas as pd
import torch
from PIL import Image
import robustness_test as r
import time

deadline = time.monotonic() + 1800
while not (r.OUT/'REPORT.md').exists():
    assert time.monotonic() < deadline, 'Timed out waiting for evaluation report'
    time.sleep(5)

torch.set_num_threads(2)
torch.set_flush_denormal(True)
manifest=json.loads((r.OUT/'run_manifest.json').read_text())
sample=pd.read_csv(r.OUT/'sample_manifest.csv')
dataset=r.Path(manifest['dataset_path'])
name='densenet121'
a,p,_=r.MODELS[name]
model=r.timm.create_model(a,pretrained=False,num_classes=1)
model.load_state_dict(torch.load(r.ROOT/p,map_location='cpu',weights_only=True),strict=True)
model.eval()
original=pd.read_csv(r.OUT/f'predictions_{name}_clean.csv')
diffs=[]
for row in sample.iloc[:2].itertuples():
    image=np.asarray(Image.open(dataset/'train'/('fake' if row.label else 'real')/(row.image_id+'.jpg')).convert('RGB'))
    x=torch.from_numpy(image.copy()).permute(2,0,1).float().unsqueeze(0)/255
    x=((x-torch.tensor([.485,.456,.406])[None,:,None,None])/torch.tensor([.229,.224,.225])[None,:,None,None]).contiguous()
    with torch.inference_mode():
        prob=float(((torch.sigmoid(model(x))+torch.sigmoid(model(x.flip(-1))))/2).item())
    prior=float(original.set_index('image_id').loc[row.image_id,'confidence'])
    diffs.append(abs(prob-prior))
assert max(diffs)<1e-5, diffs
results={'dense_runtime_consistency_max_probability_difference':max(diffs),'tolerance':1e-5,'checked_original_batch_images':2}
for condition in r.CONDITIONS:
    ensemble=np.zeros(200)
    for name,(_,_,weight) in r.MODELS.items():
        df=pd.read_csv(r.OUT/f'predictions_{name}_{condition}.csv')
        assert len(df)==200 and df.image_id.is_unique
        assert df.image_id.tolist()==sample.image_id.tolist()
        assert df.true_label.tolist()==sample.label.tolist()
        assert df.confidence.between(0,1).all()
        ensemble+=weight*df.confidence.to_numpy()
    final=pd.read_csv(r.OUT/f'predictions_ensemble_{condition}.csv')
    assert final.image_id.tolist()==sample.image_id.tolist()
    assert np.allclose(ensemble,final.confidence.to_numpy(),atol=1e-12,rtol=0)
    assert np.array_equal(ensemble>=.5,final.predicted_label)
results.update({'model_prediction_files_verified':20,'ensemble_prediction_files_verified':5,'images_per_file':200,'ensemble_weights_verified':True})
r.atomic_json(r.OUT/'verification.json',results)
r.checkpoint('06_final_verification_complete',results)
report=r.OUT/'REPORT.md'
report.write_text(report.read_text(encoding='utf-8')+f'\n## Final verification\n\nVerified 20 model files and 5 ensemble files, each with exactly 200 matching IDs, labels, and finite probabilities. Independently recomputed the weighted ensemble. For two images from the original DenseNet batch, adjusted-runtime probabilities differed by at most {max(diffs):.3g} (tolerance 1e-5). These spot checks do not establish bitwise equivalence for every image. See verification.json.\n',encoding='utf-8')
print(results)
