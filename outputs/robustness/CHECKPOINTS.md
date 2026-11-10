# Completed robustness checkpoints

| Step | Status | Evidence |
|---|---|---|
| Dataset preparation | Complete | Complete extraction under participant package `extracted/hackathon_dataset`; validation files and metadata verified |
| Sample selection | Complete | `sample_manifest.csv`: seed 42, 100 real + 100 fake, no training ID overlap |
| Model loading | Complete | `model_preflight.json`: all four checkpoints strictly matched |
| Corruption preparation | Complete | JPEG Q50/Q70, Gaussian blur sigma 1.5, Gaussian noise sigma 15; `example_*.png` |
| Clean inference | Complete | Four model prediction files, each with 200 images |
| Degraded inference | Complete | Sixteen model prediction files, each with 200 images |
| Ensemble calculation | Complete | Five ensemble prediction files; fixed weights 0.35/0.30/0.20/0.15 |
| Statistical analysis | Complete | `statistics.csv`, `per_model_statistics.csv`; 2,000 paired stratified bootstrap resamples |
| Independent verification | Complete | `verification.json`: all 25 prediction files verified; ensemble recomputed; DenseNet runtime spot check passed |
| Report and chart review | Complete | `REPORT.md`, AUC chart, confusion matrices, corruption examples |

## Findings

Clean ensemble AUC is 0.8813. Worst observed drop is 0.0129 under Gaussian noise. All four observed drops are below 0.02, but blur/noise uncertainty intervals extend above 0.02. This is a 200-image validation study, not proof of hidden-test robustness or a T4 efficiency benchmark.

## Runtime and resuming

`progress.jsonl` records individual batch saves, interruptions, and CPU runtime adjustments. DenseNet used smaller batches, on-demand preprocessing, contiguous inputs, and denormal flushing. Previously saved batches were preserved; the final numerical spot check differed by at most 1.2e-7.

From workspace root, the completed main protocol is `python SMPS/robustness_test.py`. For the lower-memory DenseNet path use `python SMPS/robustness_low_memory.py`. After regenerating the report, run `python SMPS/robustness_verify.py` and `python SMPS/robustness_model_summary.py` to regenerate verification and the supplementary tables/charts. Matching saved predictions are reused.
