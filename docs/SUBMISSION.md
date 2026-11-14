# Submission guide

Provide one Google Drive folder shared as **Anyone with the link can view**. This guide organizes the supplied requirements; it does not upload files.

| Requirement | Repository contents |
|---|---|
| Source code | Public repository link or zipped source directory with root README |
| Trained model weights | `weights/ensemble_config.json` and its five `.pth` checkpoints |
| Official test predictions | Root `predictions.csv` |
| Reproducible inference | Root `inference_script.py`, `ensemble_worker_control.py`, and `requirements.txt` |
| Detailed documentation | `docs/reports/final_technical_report.pdf`, README, training notebooks and final status/evidence |
| Optional 3–5 minute video | Existing `presentation/Explanatory Video.mp4` |
| Optional supporting evidence | Final validation/verification, training curves and Grad-CAM under `outputs/` |

## Required ensemble files

Keep these together under `weights/`:

```text
ensemble_config.json
efficientnetv2_s_best.pth
convnext_tiny_best.pth
swin_tiny_best.pth
maxvit_tiny_best.pth
densenet121_best.pth
```

Download real binaries with Git LFS; a GitHub source ZIP can contain pointers. Other checkpoints and TensorFlow/ONNX exports are preserved reference assets. The final verified pipeline uses the five PyTorch checkpoints above.

## Reports and final evidence

The participant package requests a report of at least four pages. Submit the four-page `final_technical_report.pdf`, which uses the final five-model evidence and distinguishes validation from hidden-test performance. Original `report_detailed.pdf`, `report.pdf` and `report-sneha.pdf` remain unchanged for history; the old detailed report contains superseded timing, architecture and score claims and should not be submitted as the final report. The existing video is unchanged; its duration was not revalidated.

Root `predictions.csv` is the final FP32 output. `outputs/predictions/predictions_original.csv` is the earlier result retained for auditing. Earlier FP16 comparisons, four-model timing and four-model robustness must be distinguished from final five-model evidence.

Supporting tools now live in `scripts/`. Historical manifests retain original paths and tooling hashes; use `docs/REORGANIZATION_MANIFEST.json` to locate files. Moving utilities does not constitute a new GPU measurement.

Current requirements target Python 3.10 and differ from the recorded Colab freeze. A fresh target-runtime replay remains required to certify the adjusted environment. See [setup audit](SETUP_AUDIT.md). Keep the existing final CSV with its matching original inference and worker sources; do not silently replace it with an unverified fresh candidate.

## Rules and evaluation

- Train and fine-tune exclusively on the official dataset. Do not train or tune on official test images.
- Pretrained architectures are permitted. The participant package permits ImageNet weights and excludes deepfake-specific pretrained detectors.
- Work must be original, complete and reproducible; incomplete or unverified submissions will not be evaluated.
- One final submission per team; updates are allowed until the deadline and only the latest upload counts.
- Evaluation: test performance 50%, efficiency 20%, robustness 20%, documentation 10%.

Original references: [participant README](../datasets/participant_instructions/README.md) and [evaluation specification](../datasets/participant_instructions/evaluation_metric.md). The package also specifies Python 3.10, PyTorch/TensorFlow, maximum model size 5 GB and a 60-minute T4 inference limit. Consult the recorded final runtime environment when reproducing results.
