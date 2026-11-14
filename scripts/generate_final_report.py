"""Build the four-page final report from recorded evidence; no model execution."""
from pathlib import Path
import csv
import json
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'docs/reports/final_technical_report.pdf'

def main():
    evidence=json.loads((ROOT/'outputs/final_verification/verification.json').read_text(encoding='utf-8'))
    config=json.loads((ROOT/'weights/ensemble_config.json').read_text(encoding='utf-8'))
    with (ROOT/'outputs/final_validation/statistics.csv').open() as stream:
        stats=list(csv.DictReader(stream))
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyFinal',fontName='Helvetica',fontSize=10,leading=14,spaceAfter=9))
    story=[]
    def p(text,style='BodyFinal'):story.append(Paragraph(text,styles[style]))
    def table(rows,widths):
        rows=[[Paragraph(str(x),styles['BodyFinal']) for x in row] for row in rows]
        t=Table(rows,colWidths=widths,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e3edf1')),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,-1),0.3,colors.HexColor('#c7d4db'))]))
        story.append(t);story.append(Spacer(1,10))
    p('Deepfake face classification','Title')
    p('Team SMPS | IIIT Bangalore Deepfake ML Hackathon | Final technical report')
    p('1. Problem, dataset and evaluation design','Heading1')
    p('The task is binary classification of aligned 224 x 224 face images. Label 0 denotes Real and label 1 denotes Fake. The confidence column is the estimated probability of Fake, rather than confidence in whichever label was selected. The primary ranking metric in the participant package is AUC-ROC. Labels are produced at threshold 0.5; this threshold is not claimed to be optimal for every use case.')
    p('The supplied archive contains 7,000 labelled training images: 3,478 Real and 3,522 Fake. It also contains 3,000 unlabelled official test images. Saved CSVs define a stratified 80/20 split of 5,600 training and 1,400 validation images, with seed 42. Only the supplied dataset may be used for fine-tuning. ImageNet initialization is permitted. Official test images are used for inference, not training or tuning.')
    p('Final robustness statistics use a fixed, balanced subset of 200 held-out validation images, 100 per class. This subset is smaller than the full validation set and is not a second independent test set. Models and checkpoints were selected using validation information. Consequently, results must be interpreted as validation evidence, not an unbiased estimate of private-test performance.')
    p('The final test prediction file contains exactly one row for each official test image ID. Its schema is image_id,label,confidence; image IDs omit the filename extension. Private test labels are unavailable. We do not report private-test AUC, official points, projected points, or a guaranteed ranking.')
    p('Ethical use','Heading2')
    p('False positives can incorrectly flag genuine people and false negatives can miss manipulated media. A model output is supporting evidence for review, not a definitive finding of fabrication. Performance on these aligned face images does not establish performance on arbitrary videos, unseen demographic groups, full images, or new manipulation methods. No demographic fairness or calibration study is claimed.')
    p('Evidence sources: datasets/participant_instructions/, data/train_split.csv, data/val_split.csv, outputs/final_validation/, and outputs/final_verification/. Earlier reports are preserved as historical artifacts and are superseded by this report for final claims.')
    story.append(PageBreak())
    p('2. Architecture and inference','Heading1')
    table([['Model / timm architecture','Blend','Recorded peak AUC']]+[[m['arch'],f"{m['weight']:.2f}",f"{m['val_auc']:.4f}"] for m in config['models']],[285,65,145])
    p('The ensemble combines convolutional and attention-based architectures with different feature-processing mechanisms: EfficientNetV2-S, ConvNeXt-Tiny, Swin-Tiny, MaxViT-Tiny and DenseNet-121. Each produces a single logit; sigmoid converts it to a Fake probability. Preprocessing resizes to 224 x 224, converts to a tensor, and applies ImageNet normalization.')
    p('For each model, inference averages predictions on the original image and its horizontal flip. The final confidence is the weighted sum of these per-model probabilities. Blend coefficients are nonnegative and sum to one. They are fixed heuristic coefficients, not the result of a demonstrated optimal search. Diversity motivates the combination but does not establish that it improves on the strongest individual model.')
    p('Individual peak AUC values above are recorded model-selection results, whereas final ensemble robustness uses the fixed 200-image subset. These samples and selection procedures differ; direct subtraction would not measure ensemble gain. No controlled ablation proving the benefit of all five models is claimed.')
    p('The official CLI loads weights/ensemble_config.json. That file references five checkpoints in the same directory. ensemble_worker_control.py launches each model in a separate process, releasing model memory between workers. The verified path uses FP32 with original-plus-horizontal-flip TTA and batch size 64 by default. Keep the inference entry point and worker controller together.')
    p('Reproduction command','Heading2')
    p('<font face="Courier" size="8">python inference_script.py --test_dir PATH/test/images<br/>--weights weights/ensemble_config.json --output fresh_predictions.csv</font>')
    p('Run the command on one line from the repository root. Input paths are supplied explicitly. Download binary assets through Git LFS; pointer files are not usable checkpoints. Baseline, TensorFlow and ONNX files remain available, but they are not the final verified five-model PyTorch pipeline. Cross-framework numerical parity has not been established.')
    story.append(PageBreak())
    p('3. Training, hyperparameters and challenges','Heading1')
    p('Architecture-specific notebooks under notebooks/training/ are the source of training settings. The generic src/train.py is an earlier baseline scaffold using BCEWithLogitsLoss, AdamW and cosine annealing; running it alone does not reproduce all five architecture-specific training runs. Install requirements-training.txt for that generic pipeline, which uses the Albumentations 1.x API.')
    table([['Notebook family','Recorded source settings'],['ConvNeXt, DenseNet, EfficientNetV2-S, Swin','Batch 32; learning rate 1e-4; seed 42; configured 25 epochs; focal gamma 1.5; default alpha 0.5; AdamW effective weight decay 1e-3; cosine warm restarts T_0=10, T_mult=2, eta_min=1e-6.'],['MaxViT saved-results notebook','Batch 32; learning rate 1e-4; seed 42; configured 20 epochs; focal alpha 0.5, gamma 1.5; AdamW weight decay 1e-4; cosine warm restarts T_0=5, T_mult=2, eta_min=1e-6.']],[145,350])
    p('Configured epoch limits are not claims about the epoch that produced each saved checkpoint. Saved logs and notebook outputs retain actual histories. Some notebooks define a weight-decay variable but pass a different literal value to AdamW; the table reports the effective optimizer call. Settings are therefore not described as identical across architectures.')
    p('Training includes geometric and photometric augmentation and compression/noise perturbations on supplied training images. Validation preprocessing is deterministic. Exact transform parameters belong to each notebook; a single universal augmentation recipe would misdescribe the retained experiments. Checkpoints are selected using validation AUC rather than assuming the final epoch is best.')
    p('Hyperparameter selection and limitations','Heading2')
    p('The repository records chosen configurations and exploratory models rather than a complete systematic hyperparameter search. We do not claim that the learning rate, focal loss, blend coefficients or threshold are globally optimal. Strong training performance with plateauing validation performance motivates careful checkpoint selection and restraint in performance claims.')
    p('EVA-02 Tiny is retained as an exploratory comparison, with recorded peak AUC 0.7126. Architecture and limited data are possible explanations for the result; the project does not isolate a causal failure mechanism. GPU memory and local CPU resource constraints motivated sequential worker inference. A previous precomputed ensemble CSV failed to match the current pipeline and is preserved separately as predictions_original.csv; the final CSV was generated by the actual FP32 pipeline and independently replayed.')
    story.append(PageBreak())
    p('4. Final measurements, robustness and delivery','Heading1')
    run=evidence['runs'][0]
    p(f"Recorded independent replay processed all 3,000 official test images on {evidence['device']}. Maximum probability difference was {run['reproduction']['max_probability_difference']:.1f}, with {run['reproduction']['label_mismatches']} label mismatches. Process-wall runtime was {run['process_wall_seconds']:.2f} seconds. Peak PyTorch allocated GPU memory was {run['metrics']['peak_allocated_mib']:.2f} MiB; this excludes other GPU allocations and is not total board memory.")
    table([['Condition','AUC','AUC drop from clean']]+[[s['condition'],f"{float(s['auc']):.4f}",f"{float(s['auc_drop']):.4f}"] for s in stats],[235,100,160])
    p('The clean subset AUC is 0.8817 with a 95% bootstrap interval of 0.8243-0.9311. Accuracy at threshold 0.5 is 87.5%. The largest observed AUC drop is 0.0110 under Gaussian noise. Its paired 95% interval extends to 0.0244, above 0.02. Therefore, a small observed drop does not establish a population robustness bound. Intervals use 2,000 stratified bootstrap resamples on the fixed subset.')
    p('Earlier four-model runs had median T4 time 60.36 seconds and maximum observed corruption drop 0.0129. Those are different protocols and are not the final five-model measurements. Grad-CAM visualizations are qualitative attribution; they do not prove causal attention to manipulation artifacts, decision correctness, fairness, or generalization. Bonus points and official scores are determined by judges.')
    p('Setup and submission','Heading2')
    p('The documented local setup targets Python 3.10. requirements.txt supplies compatible package pins; the original Colab freeze and original requirements are retained separately. Changing Python, numerical libraries or GPU runtime can change outputs, so run fresh verification in the target environment. New verification/validation runs write to outputs/reproduction/ rather than overwriting recorded evidence.')
    p('Supply one Google Drive folder shared as Anyone with the link can view: source code or a public repository link with README, final predictions.csv, all five trained checkpoints with ensemble_config.json, inference_script.py, ensemble_worker_control.py, requirements.txt and this detailed report. The existing explanatory video is optional. The original participant package specifies a 5 GB model limit and 60-minute T4 inference limit; the scoring specification separately rewards faster inference.')
    p('The repository audit checks file preservation, official IDs, source/weight hashes, setup metadata, CLI entry points and notebook consistency. It is not a new training run or independent T4 benchmark. Exact replay evidence describes the recorded GPU environment; the reorganized workflow still requires a fresh target-runtime replay before claiming new measurements.')
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#5a6872'))
        canvas.drawString(40,24,'TEAM SMPS | Final recorded evidence | Official scores unknown')
        canvas.drawRightString(A4[0]-40,24,f'{doc.page} / 4')
    OUT.parent.mkdir(parents=True,exist_ok=True)
    SimpleDocTemplate(str(OUT),pagesize=A4,leftMargin=40,rightMargin=40,topMargin=35,bottomMargin=40).build(story,onFirstPage=footer,onLaterPages=footer)
    print(OUT)

if __name__=='__main__':main()
