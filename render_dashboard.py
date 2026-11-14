"""Render the presentation dashboard from existing measured charts; no inference."""
import base64
import html
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHARTS = ROOT/'outputs/final_verification'


def data_image(path):
    return 'data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode()


charts = [
    ('final_roc', 'performance', '01 / PREDICTIVE PERFORMANCE', 'A closer look at every decision', 'ROC curve and confusion matrix · 200 balanced validation images · Threshold 0.5', True),
    ('model_auc', 'architectures', '02 / ARCHITECTURE COMPARISON', 'Five perspectives. One ensemble.', 'Recorded individual peak validation AUC', False),
    ('learning', 'training', '03 / LEARNING DYNAMICS', 'Learning versus generalization', 'Saved training and validation histories', False),
    ('final_robustness', 'robustness', '04 / ROBUSTNESS', 'Performance under pressure', 'Final five-model ensemble · 200 validation images · 95% bootstrap intervals', True),
    ('confidence', 'confidence', '05 / PREDICTION DISTRIBUTION', 'The shape of confidence', '3,000 unlabelled test images · Predicted probability of Fake', False),
    ('final_timing', 'efficiency', '06 / EFFICIENCY', 'Measured on Tesla T4', 'Final five-model FP32 pipeline · End-to-end process wall time', False),
]
cards = []
for name, anchor, label, title, caption, wide in charts:
    path = CHARTS/(name+'.png')
    if not path.exists():
        continue
    cards.append(f'''<article id="{anchor}" class="chart {'wide' if wide else ''}">
      <div class="card-heading"><span class="eyebrow">{label}</span><span class="corner" aria-hidden="true">↗</span></div>
      <h2>{title}</h2><p class="caption">{caption}</p>
      <button class="visual" aria-label="Enlarge {html.escape(title)}"><img src="{data_image(path)}" alt="{html.escape(title)}"></button>
    </article>''')

gallery = ''.join(f'<button class="visual gradcam" aria-label="Enlarge Grad-CAM example {i+1}"><img src="{data_image(p)}" alt="Grad-CAM example {i+1}: original image, attention heatmap and overlay"></button>'
                  for i,p in enumerate(sorted((ROOT/'outputs/gradcam').glob('*.png'))[:4]))
page = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SMPS — Deepfake Detection</title>
<style>
:root{--ink:#153b35;--muted:#6a7972;--paper:#f5f5ef;--line:#dee5dc;--accent:#b8d998}*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:90px}body{margin:0;background:var(--paper);color:var(--ink);font-family:'Segoe UI',Arial,sans-serif;-webkit-font-smoothing:antialiased}a{color:inherit;text-decoration:none}header{background:#143b35;color:#fff;position:relative;overflow:hidden}header:after{content:'';position:absolute;width:560px;height:560px;border:1px solid #ffffff16;border-radius:50%;right:-70px;top:-180px;box-shadow:0 0 0 75px #ffffff04,0 0 0 150px #ffffff03;pointer-events:none}.topbar{max-width:1320px;margin:auto;padding:26px 44px;display:flex;justify-content:space-between;align-items:center;position:relative;z-index:1}.brand{font-size:22px;letter-spacing:.15em;font-weight:750;display:flex;align-items:center;gap:12px}.brand:before{content:'';width:13px;height:13px;background:var(--accent);border-radius:50%;box-shadow:0 0 0 5px #ffffff0b}.event{font-size:11px;letter-spacing:.17em;color:#bdcfc6}.hero{max-width:1320px;margin:auto;padding:46px 44px 66px;position:relative;z-index:1}.hero .eyebrow{color:#c5dfa9}.hero h1{font:normal clamp(42px,5.5vw,74px)/1.06 Georgia,serif;letter-spacing:-.035em;margin:20px 0 22px;max-width:800px}.hero h1 em{font-weight:normal;color:#c4dfa8}.hero p{margin:0;font-size:16px;color:#c8d6cc;letter-spacing:.015em}.hero-bottom{display:flex;justify-content:space-between;align-items:center;gap:20px}.signature{border:1px solid #ffffff30;border-radius:40px;padding:10px 18px;font-size:11px;letter-spacing:.13em;color:#cce0cf;white-space:nowrap}.nav-wrap{background:#f5f5eff2;backdrop-filter:blur(18px);position:sticky;top:0;z-index:5;border-bottom:1px solid var(--line)}nav{max-width:1320px;margin:auto;padding:0 44px;display:flex;gap:34px;overflow:auto}nav a{display:block;padding:21px 0;font-size:12px;letter-spacing:.06em;font-weight:600;white-space:nowrap;border-bottom:2px solid transparent}nav a:hover{border-color:var(--ink)}main{max-width:1320px;margin:auto;padding:44px}.section-heading{display:flex;justify-content:space-between;align-items:baseline;margin:0 0 25px}.section-heading h2{font:normal 28px Georgia,serif;margin:0}.section-heading span{font-size:10px;letter-spacing:.12em;color:var(--muted)}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}.chart{min-width:0;border:1px solid var(--line);background:#fff;border-radius:14px;padding:28px 28px 16px;box-shadow:0 5px 16px #153b3503;transition:transform .2s,box-shadow .2s}.chart:hover{transform:translateY(-3px);box-shadow:0 13px 28px #153b3509}.wide{grid-column:1/-1}.card-heading{display:flex;justify-content:space-between;align-items:center}.eyebrow{font-size:10px;font-weight:650;letter-spacing:.15em;color:#607c6c}.corner{color:#718676;font-size:20px}.chart h2{font:normal clamp(22px,2.2vw,30px)/1.18 Georgia,serif;letter-spacing:-.025em;margin:17px 0 9px}.caption{font-size:12px;line-height:1.6;color:var(--muted);margin:0 0 22px}.visual{display:block;background:#fff;border:0;padding:0;width:100%;cursor:zoom-in;text-align:inherit;border-radius:8px;overflow:hidden}.visual img{display:block;width:100%;height:auto}.wide .visual{max-width:1030px;margin:auto}.attention{margin-top:56px;background:#e7ede2;border:1px solid #d8e1d2;border-radius:16px;padding:32px}.attention h2{font:normal 32px Georgia,serif;margin:14px 0 12px;letter-spacing:-.025em}.attention p{font-size:12px;color:#607367;margin:0 0 25px}.gallery{display:grid;grid-template-columns:1fr 1fr;gap:18px}.gradcam{border:8px solid white;box-shadow:0 4px 14px #153b3508}.footnote{font-size:11px;line-height:1.8;color:#748178;max-width:880px;margin:27px 0 0}footer{border-top:1px solid var(--line);margin-top:44px;padding:25px 0 0;display:flex;justify-content:space-between;font-size:10px;letter-spacing:.11em;color:#6f7d72}.lightbox{position:fixed;inset:0;z-index:20;background:#0b211bef;display:none;align-items:center;justify-content:center;padding:65px 28px 28px;backdrop-filter:blur(12px)}.lightbox.open{display:flex}.lightbox img{max-width:100%;max-height:100%;object-fit:contain;border-radius:12px;background:white}.close{position:absolute;right:28px;top:18px;color:white;background:#ffffff15;border:1px solid #ffffff30;border-radius:50%;width:38px;height:38px;font-size:25px;cursor:pointer}.visual:focus-visible,a:focus-visible,.close:focus-visible{outline:3px solid #79aa68;outline-offset:5px}@media(max-width:760px){.topbar{padding:24px}.event{font-size:9px;max-width:150px;text-align:right;line-height:1.8}.hero{padding:30px 24px 42px}.hero-bottom{display:block}.signature{display:inline-block;margin-top:24px}nav{padding:0 24px;gap:24px}main{padding:30px 18px}.grid,.gallery{grid-template-columns:1fr}.chart{padding:23px 18px 14px}.wide{grid-column:auto}.section-heading span{display:none}.attention{padding:24px 18px}.attention h2{font-size:28px}footer{gap:15px;line-height:1.8}}@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}.chart{transition:none}}@media print{.nav-wrap,.lightbox{display:none}header{print-color-adjust:exact}main{padding:20px}.chart,.attention{break-inside:avoid;box-shadow:none}.grid{display:block}.chart{margin-bottom:20px}}
</style></head><body>
<header><div class="topbar"><a class="brand" href="#">SMPS</a><span class="event">IIIT BANGALORE · DEEPFAKE ML HACKATHON</span></div>
<div class="hero"><span class="eyebrow">DEEPFAKE DETECTION / EXPERIMENTAL RESULTS</span><h1>Beyond the pixels.<br><em>Inside the evidence.</em></h1><div class="hero-bottom"><p>A visual study of performance, resilience, and attention.</p><span class="signature">FIVE-MODEL ENSEMBLE</span></div></div></header>
<div class="nav-wrap"><nav aria-label="Results sections"><a href="#performance">Performance</a><a href="#architectures">Architectures</a><a href="#training">Training</a><a href="#robustness">Robustness</a><a href="#efficiency">Efficiency</a><a href="#explainability">Explainability</a></nav></div>
<main><div class="section-heading"><h2>The results, in focus.</h2><span>TEAM SMPS / NOVEMBER 2026</span></div><div class="grid">'''
page += ''.join(cards)
page += '<div class="attention wide" id="explainability"><span class="eyebrow">07 / VISUAL EXPLAINABILITY</span><h2>Where attention lands.</h2><p>Original image · Grad-CAM heatmap · Attention overlay</p><div class="gallery">'+gallery+'</div></div></div>'
page += '''<p class="footnote">Validation and robustness charts use a fixed, balanced sample of 200 held-out images. Error bars represent 95% bootstrap intervals. Test confidence distributions do not measure test accuracy. Grad-CAM shows attribution, not causal proof.</p><footer><span>SMPS / DEEPFAKE DETECTION</span><span>MEASURED RESULTS · PRIVATE TEST SCORE UNKNOWN</span></footer></main>
<div class="lightbox" role="dialog" aria-modal="true" aria-label="Expanded chart"><button class="close" aria-label="Close expanded chart">×</button><img alt=""></div>
<script>const box=document.querySelector('.lightbox'),large=box.querySelector('img'),close=box.querySelector('.close');let previous;document.querySelectorAll('.visual').forEach(button=>button.addEventListener('click',()=>{previous=button;large.src=button.querySelector('img').src;large.alt=button.querySelector('img').alt;box.classList.add('open');document.body.style.overflow='hidden';close.focus()}));function dismiss(){box.classList.remove('open');document.body.style.overflow='';previous?.focus()}close.addEventListener('click',dismiss);box.addEventListener('click',e=>{if(e.target===box)dismiss()});document.addEventListener('keydown',e=>{if(e.key==='Escape')dismiss();if(e.key==='Tab'&&box.classList.contains('open')){e.preventDefault();close.focus()}});</script></body></html>'''
intro_head = '''<script>document.documentElement.classList.add('intro-pending');if('scrollRestoration' in history)history.scrollRestoration='manual';</script>
<style>
html.intro-pending{overflow:hidden;background:#f5f5ef}html.intro-pending body{visibility:hidden}
html.intro-pending .topbar,html.intro-pending .hero>.eyebrow,html.intro-pending .hero h1,html.intro-pending .hero-bottom,html.intro-pending .nav-wrap,html.intro-pending .section-heading,html.intro-pending .chart,html.intro-pending .attention,html.intro-pending .footnote,html.intro-pending footer{opacity:0}
.assembling{opacity:0;transform:translate3d(var(--from-x,0),var(--from-y,0),0) scale(.97);transition:opacity .8s ease var(--entrance-delay,0s),transform 1.05s cubic-bezier(.16,1,.3,1) var(--entrance-delay,0s)}
.assembling.assembled{opacity:1;transform:translate3d(0,0,0) scale(1)}
@media(prefers-reduced-motion:reduce){.assembling{transform:none;transition:opacity .25s ease}}
@media print{html.intro-pending body{visibility:visible}.assembling{opacity:1!important;transform:none!important}}
</style>'''
intro_script = '''<script>
(()=>{
const selectors=['.topbar','.hero>.eyebrow','.hero h1','.hero-bottom','.nav-wrap','.section-heading','.chart','.attention','.footnote','footer'];
const elements=[...document.querySelectorAll(selectors.join(','))];
const directions=[[0,-55],[-100,0],[-150,15],[140,0],[0,-40],[-90,0],[-130,35],[130,35],[-130,35],[130,35],[-130,35],[130,35],[0,90],[0,40],[0,35]];
elements.forEach((element,index)=>{const [x,y]=directions[index%directions.length];element.classList.add('assembling');element.style.setProperty('--from-x',x+'px');element.style.setProperty('--from-y',y+'px');element.style.setProperty('--entrance-delay',Math.min(index*.085,1.05)+'s')});
setTimeout(()=>{
window.scrollTo(0,0);
document.documentElement.classList.remove('intro-pending');
requestAnimationFrame(()=>requestAnimationFrame(()=>elements.forEach(element=>element.classList.add('assembled'))));
setTimeout(()=>elements.forEach(element=>{element.classList.remove('assembling','assembled');element.style.removeProperty('--entrance-delay')}),2400);
},Math.max(0,2000-performance.now()));
})();
</script>'''
page = page.replace('<style>', intro_head+'<style>', 1)
page = page.replace('</body>', intro_script+'</body>')
(ROOT/'results.html').write_text(page, encoding='utf-8')
print('Built presentation dashboard with measured charts only.')
