# Project tools

Run from the repository root. Official inference stays at root; these are supporting tools.

| Purpose | Tools |
|---|---|
| Dataset preparation | `create_split.py`, `eda.py` |
| Final verification/validation | `check_delivery.py`, `verify_final.py`, `validate_final.py` |
| Dashboard and reports | `build_results.py`, `render_dashboard.py`, `prepare_final_delivery.py` |
| Canonical final technical report | `generate_final_report.py` |
| Colab and notebook preparation | `colab_setup.py`, `prepare_t4_notebook.py`, `build_maxvit_notebook.py` |
| Earlier four-model efficiency | `benchmark_t4.py`, `verify_t4_results.py` |
| Earlier robustness | `robustness_test.py`, `robustness_low_memory.py`, `robustness_verify.py`, `robustness_model_summary.py` |
| Preserved exploratory/legacy tools | `calc_correlations.py`, `ensemble_blend.py`, `create_ensemble_bundle.py`, `generate_2page_report.py`, `generate_report_pdf.py` |

Tools can write predictions, evidence, reports and notebooks. Legacy blending, bundle and report generators describe earlier approaches; inspect their outputs before running them against final artifacts. Use the official inference entry point for the verified final configuration.

Fresh final verification and validation default to `outputs/reproduction/verification/` and `outputs/reproduction/validation/`. Use a new `--output-dir` for repeats. Recorded final evidence is protected. `generate_final_report.py` reads the preserved final measurements and writes `docs/reports/final_technical_report.pdf`; it performs no new model evaluation. `prepare_final_delivery.py` is a legacy concise-report tool and cannot rebuild the maintained final notebook.
