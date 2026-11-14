# Setup and submission audit

Audit date: 15 November 2026. This audit covers the reorganized repository. Publication status is recorded in Git history; the checks below describe the working tree audited before publication.

## Findings corrected

| Finding | Correction |
|---|---|
| NumPy 2.4.6 and PyMuPDF 2.1 require Python 3.11, conflicting with the Python 3.10 setup | Current requirements use NumPy 2.2.6 and PyMuPDF 1.28.2. Original Colab pins remain in `docs/environments/requirements_original_colab.txt`. |
| Generic training imports Albumentations, which was absent from requirements | Added `requirements-training.txt`, pinning the 1.x API used by `src/augmentations.py`; exercised both transform pipelines and the training CLI in a separate environment. |
| Earlier detailed report attributes four-model measurements to five models and claims projected scores | Added the four-page `docs/reports/final_technical_report.pdf`. README and submission guide identify it as canonical. All original reports remain unchanged. |
| Fresh verification/validation could overwrite recorded final evidence | Defaults now write under `outputs/reproduction/`. Completed output directories and the recorded final directories are protected. |
| Colab embedded scripts did not explicitly include all sibling imports | Embedded robustness and dashboard helpers, refreshed sources, and checked exact embedded-source consistency. |
| Colab could reuse old validation without measuring the fresh environment | Final notebook now invokes fresh validation explicitly and exports it separately from the historical evidence. |
| Colab setup source could drift and existing cached binary files were not verified | Pinned source/assets to recorded commit `a63df93271c87ca749643a30c4bc188d207f2c31`, verified all six binary assets against known sizes/hashes, and embedded layout migration/current configuration. |
| Colab package freeze could replace the Python 3.10 requirements | Fresh runtime pins are recorded separately in the new evidence directory. |
| Historical T4 importer used a relocated script path at an older Git commit | Added historical-source path fallback and canonical checkpoint lookup; new imports use a separate output directory. |
| CSV file hash differs from Colab due to Git LF normalization | Checked LF/CRLF variants and exact parsed rows against recorded replay. Prediction values match; this is a serialization difference. |

## Checks performed

- Every one of the 268 original tracked files has a mapped current location. Original model/export files, dataset, predictions, video, dashboard, PDFs and recorded evidence are retained.
- All five final checkpoint hashes match `outputs/final_verification/verification.json`; the inference source also matches that recorded run.
- Final predictions contain 3,000 unique official archive test IDs, the required column order, binary labels and finite probabilities. Parsed rows match the saved independent replay exactly.
- Python source parses, supported CLI entry points load, README/submission links resolve, the final notebook code parses, and embedded scripts/inline setup match current sources.
- The final report has four pages and was rendered and visually inspected. Original reports are preserved as history.
- Pip resolved the full current runtime and training requirements for Python 3.10 on Windows using binary distributions and `--ignore-installed`. The resolver report is under `outputs/setup_audit/`. Resolution checks dependency compatibility and package availability; it is not a clean installation or GPU execution.
- The generic training CLI and actual train/validation augmentation calls ran in a separate virtual environment with local scientific packages plus the pinned Albumentations version. This is a smoke check, not retraining.
- The Colab setup executed as a notebook-style cell without `__file__` and verified all five cached model assets plus the dataset. The embedded sources/resources installed into an empty temporary directory and the relocated verifier CLI opened there.
- The historical T4 evidence importer passed against the saved three-run evidence using a new audit output directory. The saved original five-model blend audit also passed; without a fresh test-directory inference run its status correctly remained `blocked`, not `passed`.

Run the read-only delivery audit from the repository root:

```bash
python scripts/check_delivery.py
```

For fresh inference/evaluation, use the README commands. A repeat run must choose a fresh `--output-dir`. `build_results.py` and `generate_final_report.py` summarize the committed historical measurements, whereas new reproduction folders contain fresh measurements.

## Remaining verification boundaries

The recorded Tesla T4 FP32 replay is retained and its source/weights/predictions are checked. This audit did **not** perform a new full T4 replay, retrain all models, install every pinned package in a clean Linux environment, or validate every historical training notebook by execution. The Python-3.10 dependency adjustment changes the environment from the original Colab run. Run the final notebook on a fresh T4 before treating that environment as independently verified.

The notebook's source is pinned to the original recorded commit and overlays the embedded reorganized tooling/configuration; it does not require an unpushed future commit. The original recorded dashboard/report remain distinguishable from newly generated reproduction metrics.

Google Drive sharing, final uploaded folder completeness, repository publication, video duration, originality, and official judging outcomes are not verified by a local code audit. Do not label the submission fully certified on the strength of syntax or file checks alone. Use `docs/SUBMISSION.md` when assembling the actual folder.

Historical tools and reports remain for traceability. Their commands and claims are not the supported final setup unless explicitly listed in the root README. In particular, legacy blending/report generators can replace artifacts and should not be run as the final inference path.
