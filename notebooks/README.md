# Notebooks

- `SMPS_Final_Five_Model_Verification.ipynb`: final FP32 replay and robustness workflow on Tesla T4.
- `SMPS_T4_Efficiency_Benchmark.ipynb`: earlier four-model benchmark.
- `training/`: all original architecture training and saved-results notebooks.

The two MaxViT template copies are preserved as `training/colab_training_maxvit_tiny.ipynb` and `training/colab_training_maxvit_tiny_template.ipynb`. Saved outputs describe their original runs; source path updates are not new training or benchmark measurements.

The final notebook downloads source/assets at recorded commit `a63df93271c87ca749643a30c4bc188d207f2c31`, checks binary integrity, migrates the source layout and overlays embedded current utilities. Fresh verification/validation outputs go under `outputs/reproduction/`; saved validation is not reused as a new measurement. The exported ZIP includes fresh evidence and the canonical report of the preserved historical run, separately identified. See `docs/SETUP_AUDIT.md` for the remaining target-runtime check.
