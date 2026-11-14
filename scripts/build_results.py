"""Build an offline HTML evidence page and PNG charts from committed results."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import base64
import html
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT/'outputs/final_verification'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans', 'axes.spines.top':False,
                     'axes.spines.right':False, 'figure.facecolor':'#ffffff', 'font.size':11})
sections = []


def chart(name, title, explanation, narration):
    path = OUT/(name+'.png')
    plt.savefig(path, dpi=135, bbox_inches='tight')
    plt.close()
    encoded = base64.b64encode(path.read_bytes()).decode()
    sections.append(f'<section><span class="eyebrow">EVIDENCE / {len(sections)+1:02}</span><h2>{title}</h2><p>{explanation}</p><img alt="{title}" src="data:image/png;base64,{encoded}"><details><summary>What to say in the video</summary><p>{narration}</p></details></section>')


cfg = json.loads((ROOT/'weights/ensemble_config.json').read_text(encoding='utf-8'))
models = cfg['models']
fig, ax = plt.subplots(figsize=(9,3.5))
ax.barh([m['name'] for m in models], [m['val_auc'] for m in models], color='#167d8d')
ax.set(xlim=(0.5,1), xlabel='Saved peak validation AUC (individual models)')
for i,m in enumerate(models): ax.text(m['val_auc']+.006,i,f"{m['val_auc']:.4f}",va='center')
chart('model_auc','Different architectures, one detector',
      'Individual peak validation values recorded in the ensemble configuration. These are not a measured ensemble AUC or hidden-test scores.',
      'We compared several architectures on validation data. EfficientNetV2-S has the highest recorded individual AUC, 0.9025. We combine five models using fixed probability weights. Model diversity motivates this design, but an improvement must be measured rather than assumed.')

fig, ax = plt.subplots(figsize=(9,3.5))
for label,path in [('MaxViT','outputs/train_log_maxvit.csv'),('ConvNeXt','outputs/train_log_convnext.csv')]:
    df=pd.read_csv(ROOT/path)
    ax.plot(df.epoch,df.val_auc,label=label+' validation',linewidth=2)
    ax.plot(df.epoch,df.train_auc,label=label+' training',linestyle='--',alpha=.65)
ax.set(xlabel='Epoch',ylabel='AUC',ylim=(.5,1.01)); ax.legend(ncol=2)
chart('learning','Why checkpoint selection matters',
      'Training and validation histories from saved logs. Peak validation checkpoints are preferable to selecting the last epoch automatically.',
      'Training AUC continues rising while validation performance levels off. This gap indicates overfitting. We select checkpoints using validation performance, and keep the official test images out of training.')

stats=pd.read_csv(ROOT/'outputs/robustness/statistics.csv')
fig, ax=plt.subplots(figsize=(9,3.8))
ax.errorbar(range(len(stats)),stats.auc,yerr=[stats.auc-stats.auc_ci_low,stats.auc_ci_high-stats.auc],fmt='o',capsize=5,color='#167d8d')
ax.set(xticks=range(len(stats)),xticklabels=['Clean','JPEG 50','JPEG 70','Blur','Noise'],ylabel='AUC with saved 95% bootstrap intervals',ylim=(.75,1))
chart('robustness','Robustness, with uncertainty visible',
      'Existing FOUR-model ensemble; 200 balanced validation images. Clean AUC 0.8813; largest observed drop 0.0129. Intervals do not prove the population drop stays below 0.02. This does not validate the final five-model ensemble.',
      'We tested the earlier four-model ensemble on 200 balanced validation images. Noise produced the largest observed AUC drop, 0.0129. The uncertainty intervals matter: this sample does not guarantee the same robustness on unseen data or for our final five-model configuration.')

timing=pd.read_csv(ROOT/'outputs/efficiency_t4/summary.csv')
fig, ax=plt.subplots(figsize=(9,3.8))
ax.errorbar(range(len(stats)),stats.auc_drop,yerr=[stats.auc_drop-stats.drop_ci_low,stats.drop_ci_high-stats.auc_drop],fmt='o',capsize=5,color='#167d8d')
ax.axhline(.02,color='#bc633e',linestyle='--',label='0.02 analysis target (not official scoring formula)')
ax.axhline(0,color='gray',linewidth=.7)
ax.set(xticks=range(len(stats)),xticklabels=['Clean','JPEG 50','JPEG 70','Blur','Noise'],ylabel='Paired AUC drop with 95% bootstrap intervals');ax.legend(fontsize=9)
chart('auc_drop','Observed drops versus statistical uncertainty',
      'Earlier FOUR-model sample. Negative drops indicate higher measured AUC than clean. Noise and blur intervals extend above 0.02; a small observed drop does not prove a strict upper bound.',
      'The dots are observed changes relative to clean images. The intervals capture sample uncertainty. Noise and blur extend beyond the analysis target, so we report encouraging observed stability without claiming a guarantee.')

from sklearn.metrics import roc_curve,confusion_matrix,roc_auc_score
old=pd.read_csv(ROOT/'outputs/robustness/predictions_ensemble_clean.csv')
truth='label' if 'label' in old else 'true_label'
prob='confidence' if 'confidence' in old else 'probability'
if truth in old and prob in old:
    assert abs(roc_auc_score(old[truth],old[prob])-stats.iloc[0].auc)<1e-6
    fig,(ax,bx)=plt.subplots(1,2,figsize=(10,4))
    fpr,tpr,_=roc_curve(old[truth],old[prob]);ax.plot(fpr,tpr,color='#167d8d');ax.plot([0,1],[0,1],'--',color='gray')
    ax.set(xlabel='False positive rate',ylabel='True positive rate',title='Earlier four-model ROC')
    cm=confusion_matrix(old[truth],old[prob]>=.5,labels=[0,1]);bx.imshow(cm,cmap='Blues')
    for i in range(2):
        for j in range(2):bx.text(j,i,str(cm[i,j]),ha='center',va='center')
    bx.set(xticks=[0,1],yticks=[0,1],xticklabels=['Real','Fake'],yticklabels=['Real','Fake'],xlabel='Predicted',ylabel='True',title='Clean sample: threshold 0.5')
    chart('reference_roc','Ranking performance and classification errors',
          'Existing FOUR-model balanced validation sample, 200 images. ROC and confusion counts are recomputed from the saved predictions, with AUC checked against statistics.csv.',
          'The ROC curve evaluates ranking across thresholds. The confusion matrix shows actual correct and incorrect decisions at threshold 0.5. These are earlier four-model validation results, not private test results.')

fig, ax=plt.subplots(figsize=(9,3.3))
ax.bar(['Run 1','Run 2','Run 3'],timing.wall_seconds,color='#167d8d')
ax.axhline(300,color='#bc633e',linestyle='--',label='5-minute scoring boundary')
ax.set(ylabel='End-to-end seconds (Tesla T4)',ylim=(0,330));ax.legend()
chart('timing','Measured speed of the earlier pipeline',
      'Supplied FOUR-model T4 measurements: 77.66, 60.36, 56.74 seconds. Median 60.36 seconds. Includes process startup and CSV writing; excludes downloads. Final five-model timing remains separate.',
      'The earlier four-model pipeline processed all 3,000 images in a median of 60.36 seconds on a T4. We do not transfer that timing to the five-model pipeline. The final runner records its own runtime and memory when executed on the target hardware.')

pred=pd.read_csv(ROOT/'predictions.csv')
fig, ax=plt.subplots(figsize=(9,3.3))
ax.hist(pred.confidence,bins=25,color='#167d8d',edgecolor='white');ax.axvline(.5,color='#bc633e',linestyle='--')
ax.set(xlabel='Predicted probability of Fake',ylabel='Number of test images')
chart('confidence','What the submitted predictions contain',
      f'{len(pred):,} unlabelled test predictions. This distribution describes model outputs; it cannot measure test accuracy, AUC, or calibration without labels.',
      'Each image receives a probability of being fake, and the label uses a threshold of 0.5. The histogram shows how those probabilities are distributed. Because test labels are private, this graph does not establish test accuracy or calibration.')

verification=json.loads((OUT/'verification.json').read_text(encoding='utf-8')) if (OUT/'verification.json').exists() else {}
if (ROOT/'outputs/final_validation/protocol.json').exists():
    final=pd.read_csv(ROOT/'outputs/final_validation/statistics.csv')
    fig,ax=plt.subplots(figsize=(9,3.5))
    ax.errorbar(range(len(final)),final.auc,yerr=[final.auc-final.auc_ci_low,final.auc_ci_high-final.auc],fmt='o',capsize=5,color='#167d8d')
    ax.set(xticks=range(len(final)),xticklabels=final.condition, ylabel='Five-model AUC with 95% bootstrap intervals')
    ax.tick_params(axis='x',labelrotation=15)
    chart('final_robustness','Final five-model validation and robustness',
          'Measured on the existing fixed 200-image balanced validation sample; 2,000 stratified bootstrap resamples. Not hidden-test performance.',
          'This graph evaluates the final five-model pipeline on the same fixed validation sample. We report uncertainty alongside AUC, and keep this separate from hidden-test performance.')
    clean=pd.read_csv(ROOT/'outputs/final_validation/predictions_clean.csv')
    from sklearn.metrics import roc_curve,confusion_matrix
    fig,(ax,bx)=plt.subplots(1,2,figsize=(10,4))
    fpr,tpr,_=roc_curve(clean.label,clean.confidence)
    ax.plot(fpr,tpr,color='#167d8d');ax.plot([0,1],[0,1],'--',color='gray')
    ax.set(xlabel='False positive rate',ylabel='True positive rate',title='Clean validation ROC')
    cm=confusion_matrix(clean.label,clean.confidence>=.5,labels=[0,1]);bx.imshow(cm,cmap='Blues')
    for i in range(2):
        for j in range(2):bx.text(j,i,str(cm[i,j]),ha='center',va='center')
    bx.set(xticks=[0,1],yticks=[0,1],xticklabels=['Real','Fake'],yticklabels=['Real','Fake'],xlabel='Predicted',ylabel='True',title='Threshold = 0.5')
    chart('final_roc','Ranking quality and concrete errors',
          'ROC uses continuous fake probabilities. The confusion matrix uses threshold 0.5 on the same 200 validation images.',
          'ROC describes the trade-off across thresholds, while the confusion matrix counts errors at our fixed threshold of 0.5. These are validation results and do not reveal private test labels.')
status=verification.get('weights_to_csv_status','not_run')
if verification.get('runs'):
    fig,ax=plt.subplots(figsize=(9,3.3))
    ax.bar([str(r['run']) for r in verification['runs']],[r['process_wall_seconds'] for r in verification['runs']],color='#167d8d')
    ax.set(xlabel='Final five-model run',ylabel='Process wall seconds')
    chart('final_timing','Final five-model measured runtime',html.escape(verification.get('device','Unknown device'))+'; consult verification.json for per-run reproduction results.',
          'These timings come from the final five-model submission script. The device is identified explicitly; CPU measurements are not T4 benchmark results.')

gradcams=[]
for p in list((ROOT/'outputs/gradcam').glob('*.png'))[:4]:
    gradcams.append('<img alt="Grad-CAM attention overlay" src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'">')
page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SMPS | Deepfake detection evidence</title><style>
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f2f3ee;color:#152c35;font:17px/1.6 system-ui,sans-serif}header{background:#153b46;color:#fff;padding:65px max(6vw,24px)}header h1{font-size:clamp(35px,5vw,66px);line-height:1.08;max-width:900px;margin:18px 0}header p{max-width:780px;color:#d6e8e6}.eyebrow{font-size:12px;letter-spacing:.18em;font-weight:700;color:#167d8d}header .eyebrow{color:#a9e4d8}nav{display:flex;gap:24px;padding:18px 6vw;background:#fff;position:sticky;top:0;z-index:2;box-shadow:0 1px 0 #ddd}a{color:#167d8d}main{max-width:1120px;margin:auto;padding:30px 24px}section{padding:32px;margin:24px 0;background:white;border-radius:8px}h2{font-size:29px;line-height:1.2}section>img{width:100%;height:auto}details{background:#edf5f2;padding:16px;margin-top:14px}summary{cursor:pointer;font-weight:650}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}.metric{border-left:3px solid #167d8d;padding:12px}.metric strong{display:block;font-size:30px}.warning{border-left:4px solid #bc633e;padding:16px;background:#fff4e9}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#102d36;color:#ddf2e8;padding:18px;font-size:14px}.gallery{display:grid;grid-template-columns:1fr 1fr;gap:15px}.gallery img{width:100%}button{background:#167d8d;color:white;border:0;padding:12px 18px;cursor:pointer;font:inherit}@media(max-width:650px){section{padding:20px}.metrics{grid-template-columns:1fr}.gallery{grid-template-columns:1fr}nav{position:static;font-size:14px;gap:15px}}@media print{nav,button{display:none}section{break-inside:avoid}details{display:block}}
</style><header><span class="eyebrow">TEAM SMPS / IIIT BANGALORE HACKATHON</span><h1>Detecting synthetic faces.<br>Showing the evidence.</h1><p>Five architectures, fixed probability blending, and a clear distinction between completed measurements and verification still to run.</p><button onclick="document.documentElement.requestFullscreen?.()">Presentation mode</button></header><nav><a href="#status">Run status</a><a href="#evidence">Results & graphs</a><a href="#explain">Explainability</a><a href="#run">Run commands</a></nav><main>'''
page+=f'<section id="status"><span class="eyebrow">VERIFICATION</span><h2>Submission readiness</h2><div class="metrics"><div class="metric"><strong>3,000</strong>Unique prediction IDs</div><div class="metric"><strong>5</strong>Ensemble models</div><div class="metric"><strong>{html.escape(status.upper())}</strong>Weights-to-CSV reproduction</div></div><p class="warning">Saved CSV blending is a separate check from fresh inference. No official score is known. Four-model robustness and T4 evidence must not be described as five-model results.</p><p id="progress">Live progress appears here when served over localhost.</p><pre>{html.escape(json.dumps(verification,indent=2))}</pre></section><div id="evidence"></div>'
page+=''.join(sections)
page+='<section id="explain"><span class="eyebrow">EXPLAINABILITY</span><h2>Where the detector attends</h2><div class="gallery">'+''.join(gradcams)+'</div><p>Grad-CAM highlights regions influencing a prediction. It is a qualitative diagnostic, not proof of causality or correctness.</p><details><summary>What to say in the video</summary><p>These overlays help inspect the regions influencing the detector. We use them to look for reliance on facial regions and possible shortcuts. A heatmap alone cannot prove that a prediction is correct.</p></details></section>'
page+='''<section id="run"><h2>Reproduce, compare, present</h2><pre>python scripts/verify_final.py
# After downloading the LFS assets:
python scripts/verify_final.py --test-dir datasets/extracted/hackathon_dataset/test/images --runs 3
# Serve this page from a second terminal:
python -m http.server 8765 --bind 127.0.0.1
# Open http://127.0.0.1:8765/presentation/results.html</pre><p>Audit-only mode exits with code 2 when fresh inference is blocked. It never reports a pass without running real inference. Test predictions are not overwritten.</p></section></main><script>
if(location.protocol!=='file:'){let last='';setInterval(async()=>{try{let r=await fetch('outputs/final_verification/progress.json?t='+Date.now());if(r.ok){let p=await r.json();document.querySelector('#progress').textContent='LIVE: '+p.stage+' | '+p.processed+' / '+p.total+' images'+(p.device?' | '+p.device:'')}}catch{}try{let r=await fetch('outputs/final_verification/verification.json?t='+Date.now());if(r.ok){let s=await r.text();if(last&&last!==s)location.reload();last=s}}catch{}},2500)}
</script></html>'''
(ROOT/'presentation/results.html').write_text(page,encoding='utf-8')
print('[GRAPHS] Built offline presentation/results.html and PNG figures in outputs/final_verification')

# Keep the public dashboard presentation-only after chart regeneration.
import runpy
runpy.run_path(str(ROOT/'scripts/render_dashboard.py'))
