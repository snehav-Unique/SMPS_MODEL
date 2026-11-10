# Measured NVIDIA T4 efficiency benchmark

GPU: Tesla T4. Images per run: 3,000. Runs: 3. Median runtime: **60.36 seconds**; range 56.74–77.66 seconds. All measured runs under 300 seconds: **True**.

| Run | End-to-end seconds | Images/sec | GPU batch mean ms | GPU batch p95 ms | Peak allocated MiB | Peak reserved MiB | <300s |
|---|---|---|---|---|---|---|---|
| 1 | 77.66 | 38.63 | 275.32 | 216.92 | 561.9 | 944.0 | True |
| 2 | 60.36 | 49.70 | 259.23 | 228.02 | 561.9 | 944.0 | True |
| 3 | 56.74 | 52.88 | 248.23 | 228.30 | 561.9 | 944.0 | True |


## Measurement protocol

Each run starts a new Python process and performs all four model loads, FP16 inference, original+horizontal-flip TTA, fixed probability blending, and CSV output. Batch size 64; loader workers 2. No artificial warmup is excluded: first-batch initialization and cuDNN algorithm selection count in wall time. Downloads, ZIP extraction, hash checks, and parent setup are outside the timer. Filesystem caches are not flushed; later runs can benefit from warm caches. Profiling synchronization and log writing are included in runtime.

GPU batch times measure both TTA forward passes and probability averaging, excluding loading/preprocessing and host-to-device transfer. They are per model batch, not the latency of a complete four-model ensemble batch. Full-run throughput counts each of the 3,000 unique images once, not once per model or TTA view. PyTorch peak allocated/reserved VRAM excludes non-PyTorch allocations such as CUDA contexts; this is not a total-board memory measurement.

This instrumented ensemble runner is a reference pipeline. The repository's original inference_script.py remains a separate single-model CLI; final submission inference must be aligned with the selected pipeline and benchmarked before claiming the official efficiency score. Timing on Colab does not guarantee timing on the judges' system. No labels or accuracy claims are inferred from the unlabeled test set. FP16 predictions have not been assumed bitwise equivalent to the CPU FP32 robustness predictions.

## Evidence

summary.csv; manifest.json; checkpoint.json; per-run console.log, result.json, model_profile.csv, batch_latency.csv, and predictions.csv. Failed runs stop the benchmark and do not receive a passing claim. Re-running a measurement requires a fresh output directory; completed measurements are not silently reused.

## Evidence verification

All three runs contain 3,000 matching test IDs, finite probabilities, four completed models, and 188 profiled batches. Script and all four checkpoint hashes match the committed project. Timings and throughput were recomputed from the supplied logs. Maximum prediction difference between runs: 0.00075074. See verification.json. This verifies consistency of supplied Colab evidence; it is not an independent GPU rerun.
