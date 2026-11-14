"""Generate the concise report and self-contained final-run Colab notebook."""
import json
import shutil
import sys
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.pagesizes import A4

ROOT=Path(__file__).resolve().parent
verification_path=ROOT/'outputs/final_verification/verification.json'
verification=json.loads(verification_path.read_text()) if verification_path.exists() else {}
fresh_status=verification.get('weights_to_csv_status','not_run')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyCompact',fontName='Helvetica',fontSize=9,leading=12,spaceAfter=7))
story=[]
def para(text,style='BodyCompact'):story.append(Paragraph(text,styles[style]))
def image(path,w=475,h=170):story.append(Image(str(ROOT/path),width=w,height=h,kind='proportional'))
backup=ROOT/'report_detailed.pdf'
if not backup.exists():shutil.copy2(ROOT/'report.pdf',backup)
para('Deepfake face classification','Title')
para('Team SMPS | IIIT Bangalore Hackathon | Submission technical report')
para('1. Objective, data and validation','Heading2')
para('Predict whether an aligned 224 x 224 face image is real (0) or fake (1). The supplied dataset has 7,000 labelled training images and 3,000 unlabelled test images. The saved stratified split uses 5,600 training and 1,400 validation images. Only the supplied images are used for fine-tuning; ImageNet initialization is permitted. Validation AUC guides checkpoint selection. Private test AUC is unknown. Ethical use requires acknowledging false positives and avoiding treating a detector output as definitive evidence.')
para('2. Architecture and model selection','Heading2')
para('The submission averages original and horizontal-flip probabilities per model, then blends five architectures. Fixed weights are heuristics informed by individual validation performance; they are not claimed to be optimized on hidden labels. A diverse model set does not guarantee an ensemble gain.')
rows=[['Architecture','Saved peak val AUC','Blend weight'],['EfficientNetV2-S','0.9025','0.40'],['ConvNeXt-Tiny','0.8973','0.25'],['Swin-Tiny','0.8953','0.15'],['MaxViT-Tiny','0.8902','0.12'],['DenseNet-121','0.8905','0.08']]
t=Table(rows,colWidths=[255,125,95]);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#153b46')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTSIZE',(0,0),(-1,-1),9),('BOTTOMPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,1),(-1,-1),.3,colors.lightgrey)]));story.append(t)
para('3. Training pipeline and challenges','Heading2')
para('Recorded training uses ImageNet normalization, geometric and photometric augmentation, compression/noise augmentation, AdamW and scheduled learning rates. Most candidate notebooks use binary focal loss (gamma 1.5). Exact settings differ by architecture and are retained in the training notebooks. Training AUC approaching 1 while validation plateaus motivates early checkpoint selection. EVA-02 Tiny recorded AUC 0.7126; data size and architecture are possible explanations, not experimentally isolated causes.')
image('outputs/final_verification/learning.png',475,165)
story.append(PageBreak())
para('4. Evaluation, uncertainty and explainability','Heading2')
if (ROOT/'outputs/final_validation/statistics.csv').exists():
    import csv
    records=list(csv.DictReader((ROOT/'outputs/final_validation/statistics.csv').open()))
    clean=next(r for r in records if r['condition']=='clean')
    worst=max(records,key=lambda r:float(r['auc_drop']))
    para(f"Final five-model FP32 evaluation uses 200 fixed balanced validation images. Clean AUC is {float(clean['auc']):.4f} (95% bootstrap interval {float(clean['auc_ci_low']):.4f}-{float(clean['auc_ci_high']):.4f}). Largest observed AUC drop is {float(worst['auc_drop']):.4f} under {worst['condition']}, with paired interval {float(worst['drop_ci_low']):.4f}-{float(worst['drop_ci_high']):.4f}. Intervals use 2,000 stratified resamples; these are validation-subset estimates, not hidden-test scores or population guarantees.")
else:
    para('Earlier four-model robustness evidence uses 200 fixed validation images, balanced by class. Clean AUC is 0.8813; JPEG Q50 0.8874; JPEG Q70 0.8854; blur 0.8740; noise 0.8684. The largest observed absolute drop is 0.0129. Bootstrap intervals for some drops extend beyond 0.02, so this sample does not establish a population guarantee. These measurements are not five-model robustness results.')
image('outputs/final_verification/final_robustness.png' if (ROOT/'outputs/final_verification/final_robustness.png').exists() else 'outputs/final_verification/robustness.png',475,150)
para('Grad-CAM overlays are included as qualitative diagnostics. They highlight regions influencing predictions, without proving correct reasoning, causal reliance on a region, or generalization to new manipulation methods.')
para('5. Efficiency and reproducibility status','Heading2')
para('Supplied four-model Tesla T4 runs took 77.66, 60.36 and 56.74 seconds for 3,000 images (median 60.36 seconds). The final five-model CLI is a different pipeline. Official test performance and scores remain unknown.')
if verification.get('runs'):
    timings=', '.join(f'{r["process_wall_seconds"]:.2f}' for r in verification['runs'])
    para(f'Final five-model measured process-wall times: {timings} seconds on {verification.get("device","unknown device")}. Fresh weights-to-CSV status: <b>{fresh_status.upper()}</b>. Detailed per-run probability differences, label mismatches and the comparison tolerance are in verification.json. CPU results must not be presented as T4 measurements.')
else:
    para(f'Fresh weights-to-CSV status: <b>{fresh_status.upper()}</b>. The local attempt encountered resource constraints; no completed five-model timing is available. The prepared Colab notebook performs this check on a T4, saves evidence, and regenerates this report from completed results.')
para('The legacy five-model blend reproduces its original CSV exactly. The original is preserved separately when a fresh FP32 submission candidate is generated. Final acceptance requires an independent weights-to-CSV replay against the fresh candidate. verify_final.py performs the latter with a declared tolerance, logs, source/weight hashes and actual device/runtime evidence. A mismatch must be resolved before submission. The HTML page identifies pending checks explicitly.')
para('6. Reproduction commands','Heading2')
para('<font face="Courier" size="8">git lfs pull --include="weights/*,datasets/hackathon_dataset.zip"<br/>python inference_script.py --test_dir PATH/test/images --weights weights/ensemble_config.json --output fresh.csv<br/>python verify_final.py --test-dir PATH/test/images --runs 3<br/>python validate_final.py --dataset-root PATH<br/>python build_results.py</font>')
para('Install requirements.txt in Python 3.10 for the original training environment. The final-run Colab notebook records its actual runtime dependencies; preserve those records with generated evidence. Extract the official archive and pass paths explicitly. The official participant template uses image_id,label,confidence and inference_script.py. Confidence is probability of Fake. Required code, configuration and all five binary weight files must accompany the submission.')
para('Limitations: hidden test labels unavailable; robustness uses a validation subset; no demonstrated final five-model AUC improvement; hardware and precision can alter probabilities. Official points and bonus awards are determined by judges.')
def footer(canvas,doc):
    canvas.setFont('Helvetica',8);canvas.setFillColor(colors.grey)
    canvas.drawString(42,24,'TEAM SMPS | Evidence and limitations stated separately')
    canvas.drawRightString(A4[0]-42,24,str(doc.page))
SimpleDocTemplate(str(ROOT/'report.pdf'),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=32,bottomMargin=36).build(story,onFirstPage=footer,onLaterPages=footer)
if '--report-only' in sys.argv or '--rebuild-notebook' not in sys.argv:
    print('Updated two-page report from actual verification evidence.')
    sys.exit(0)

cells=[]
def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(True)))
def code(s):cells.append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=s.splitlines(True)))
md('# SMPS final five-model verification\nChoose **Runtime > Change runtime type > T4 GPU**, then Run all. This notebook embeds the updated local scripts, downloads only official assets, preserves submitted predictions, records failures honestly, and downloads evidence. Existing four-model evidence is never relabelled as five-model evidence.')
code("import torch\nassert torch.cuda.is_available(), 'Select a T4 GPU runtime'\nassert 'T4' in torch.cuda.get_device_name(), 'This benchmark requires a T4'\nprint(torch.cuda.get_device_name())\n")
code('%pip -q install timm==1.0.31 pandas matplotlib scikit-learn opencv-python-headless reportlab pypdf pymupdf\n'+(ROOT/'colab_setup.py').read_text()+'\nimport subprocess\n')
embedded={name:(ROOT/name).read_text(encoding='utf-8') for name in ['inference_script.py','ensemble_worker_control.py','verify_final.py','build_results.py','validate_final.py','prepare_final_delivery.py','colab_setup.py']}
code('import json\nscripts=json.loads('+repr(json.dumps(embedded))+')\nfor name, source in scripts.items():\n    Path(name).write_text(source, encoding="utf-8")\nprint("Installed exact final local scripts")\n')
code("import zipfile, hashlib\nextract=project/'datasets/extracted'\nextract.mkdir(exist_ok=True)\nwith zipfile.ZipFile('datasets/hackathon_dataset.zip') as z:\n    for item in z.infolist():\n        target=(extract/item.filename).resolve()\n        assert target.is_relative_to(extract.resolve())\n        if not item.is_dir():\n            target.parent.mkdir(parents=True,exist_ok=True)\n            target.write_bytes(z.read(item))\nmetadata=list(extract.rglob('train_metadata.csv'))\nassert len(metadata)==1, 'Expected one official dataset'\ndataset=metadata[0].parent\nprint('Dataset:',dataset)\n")
code("# Preserve both the inference exit status and all diagnostic evidence.\nrun=subprocess.run(['python','-u','verify_final.py','--test-dir',str(dataset/'test/images'),'--runs','3'])\nprint('Verification exit code:',run.returncode)\n# Code 2 means blocked or mismatched reproduction, never a pass.\n")
code("# Measure final five-model validation / robustness with uncertainty.\nsubprocess.run(['python','-u','validate_final.py','--dataset-root',str(dataset)],check=True)\nwith open('outputs/final_verification/environment.txt','w') as env_log:\n    subprocess.run(['python','-m','pip','freeze'],stdout=env_log,check=True)\nsubprocess.run(['python','build_results.py'],check=True)\nsubprocess.run(['python','prepare_final_delivery.py','--report-only'],check=True)\nfrom pypdf import PdfReader\nassert len(PdfReader('report.pdf').pages)==2, 'Check report layout before submission'\nfrom IPython.display import display, HTML\ndisplay(HTML(Path('results.html').read_text()))\n")
code("import zipfile\nfrom google.colab import files\nwith zipfile.ZipFile('/content/SMPS_final_evidence.zip','w',zipfile.ZIP_DEFLATED) as z:\n    z.write('results.html')\n    z.write('report.pdf')\n    for name in scripts:z.write(name)\n    for folder in ['outputs/final_verification','outputs/final_validation']:\n        for path in Path(folder).rglob('*'):\n            if path.is_file():z.write(path)\nfiles.download('/content/SMPS_final_evidence.zip')\nprint(Path('outputs/final_verification/verification.json').read_text())\nprint('Do not submit until weights_to_csv_status is passed. A failed comparison needs investigation.')\n")
(ROOT/'notebooks/SMPS_Final_Five_Model_Verification.ipynb').write_text(json.dumps(dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),accelerator='GPU'),nbformat=4,nbformat_minor=5),indent=1))
print('Generated report.pdf and final-run Colab notebook; detailed report retained.')
