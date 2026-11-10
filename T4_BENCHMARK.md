# T4 efficiency benchmark

**Status: prepared; no NVIDIA T4 measurements have been collected yet.** The development computer has CPU-only PyTorch and no accessible NVIDIA GPU.

Open [the Colab notebook](https://colab.research.google.com/github/snehav-Unique/SMPS_MODEL/blob/main/notebooks/SMPS_T4_Efficiency_Benchmark.ipynb), select a T4 GPU runtime, and run all cells. The notebook downloads and verifies the committed dataset and four model checkpoints through Git LFS, runs three independent measurements, and downloads a ZIP of the results. It refuses CPU and non-T4 devices.

The benchmark measures all 3,000 test images using the four-model ensemble, original and horizontal-flip TTA, FP16, batch size 64, and two loader workers. End-to-end time includes fresh Python startup, library imports, model loading, preprocessing, transfers, inference, blending, CSV writing, and profiling overhead. Downloads and extraction are outside the timed process. First-batch initialization counts; repeated runs may benefit from filesystem caches.

Outputs include a report, summary table, manifest, per-run checkpoints, predictions, console logs, model timings, GPU batch latency, and PyTorch peak allocated/reserved VRAM. These memory measures exclude CUDA contexts and other non-PyTorch allocations. GPU batch latencies are per model including two TTA views, not a complete ensemble batch.

The original `inference_script.py` is still a separate single-model submission CLI. This benchmark measures the reference ensemble implementation in `benchmark_t4.py`; the final submission runner must be aligned and timed before claiming the official efficiency score.

For another T4 machine:

```bash
python benchmark_t4.py --test-dir /path/to/test/images --output outputs/efficiency_t4_new --runs 3 --batch-size 64 --workers 2
```

Use a fresh output directory for a new measurement. Failed runs retain diagnostics and do not generate a passing claim. Return the downloaded results ZIP for verification and inclusion in GitHub.

Colab GPU selection and availability: [official Colab runtime guidance](https://colab.research.google.com/notebooks/pro.ipynb).
