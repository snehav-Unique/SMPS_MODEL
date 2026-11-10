"""Build the checked-in Colab notebook without storing execution results."""
import json
from pathlib import Path

def md(text): return {'cell_type':'markdown','metadata':{},'source':text.splitlines(keepends=True)}
def code(text): return {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[], 'source':text.splitlines(keepends=True)}

cells=[md('''# SMPS: NVIDIA T4 efficiency benchmark

**Prepared, not yet measured.** This notebook downloads the real dataset and weights, runs three fresh-process benchmarks, and exports measured results. It does not train models.

1. Select **Runtime → Change runtime type → T4 GPU**, then connect.
2. Run all cells in order. T4 availability is subject to Colab quota and availability.
3. Keep the final downloaded ZIP and send it back for review and inclusion in the repository.

Expected setup download: about 532 MiB of unique Git LFS payloads for the full repository; this notebook pulls only the dataset and four required weights. Setup/downloads are excluded from inference timing. The timed process includes Python/library startup, model loading, image preprocessing, all four models, two-view TTA, blending, and CSV writing. No result or score is prefilled.

Repository: https://github.com/snehav-Unique/SMPS_MODEL

Colab runtime guidance: https://colab.research.google.com/notebooks/pro.ipynb
'''),code('''import subprocess
import torch

assert torch.cuda.is_available(), "Select a T4 GPU runtime before proceeding."
gpu_name = torch.cuda.get_device_name(0)
assert "T4" in gpu_name.upper(), f"T4 required; this runtime has {gpu_name}."
print("Confirmed GPU:", gpu_name)
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
subprocess.run(["nvidia-smi"], check=True)
'''),md('''## Download the project and LFS assets

No access token is embedded. If the repository is private, provide authenticated access in your own runtime before this cell; do not paste credentials into a shared notebook.
'''),code('''import os
import shutil
import sys
from pathlib import Path

REPO = Path("/content/SMPS_MODEL")
if shutil.which("git-lfs") is None:
    subprocess.run(["apt-get", "update", "-qq"], check=True)
    subprocess.run(["apt-get", "install", "-y", "git-lfs"], check=True)
if not REPO.exists():
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")
    subprocess.run(["git", "clone", "https://github.com/snehav-Unique/SMPS_MODEL.git", str(REPO)], env=env, check=True)
else:
    print("Using existing checkout; commit is recorded below. Use a fresh runtime for a fresh checkout.")
assert (REPO / "benchmark_t4.py").exists(), "Checkout lacks benchmark_t4.py; update or use a fresh runtime."
subprocess.run(["git", "lfs", "install", "--local"], cwd=REPO, check=True)
assets = [
    "datasets/hackathon_dataset.zip",
    "efficientnetv2_s_results/efficientnetv2_s_results/efficientnetv2_s_best.pth",
    "weights/convnext_tiny_best.pth",
    "swin_results/swin_results/swin_tiny_best.pth",
    "densenet121_results/densenet121_results/densenet121_best.pth",
]
subprocess.run(["git", "lfs", "pull", "--include=" + ",".join(assets), "--exclude="], cwd=REPO, check=True)
# Keep Colab's CUDA-enabled torch/torchvision pair; install only the missing model-library dependencies.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm==1.0.31", "--no-deps"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub", "safetensors", "pyyaml"], check=True)
subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True)
'''),md('''## Verify assets and extract the test images

SHA-256 hashes must match the committed inventory. The benchmark uses all 3,000 unlabeled test images. No accuracy metrics can be computed from this unlabeled set.
'''),code('''import hashlib
import json
import zipfile

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

inventory = json.loads((REPO / "datasets/asset_inventory.json").read_text())
for asset in inventory["assets"]:
    if asset["path"] in assets:
        path = REPO / asset["path"]
        assert path.stat().st_size == asset["bytes"], f"Size mismatch: {path}"
        assert sha256(path) == asset["sha256"], f"Hash mismatch: {path}"
        print("Verified:", asset["path"])

extracted = REPO / "datasets/extracted"
with zipfile.ZipFile(REPO / "datasets/hackathon_dataset.zip") as archive:
    for entry in archive.infolist():
        # Only test images are needed for efficiency measurement.
        if entry.filename.startswith("hackathon_dataset/test/images/") and not entry.is_dir():
            target = (extracted / entry.filename).resolve()
            assert target.is_relative_to(extracted.resolve()), "Unsafe archive path"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(entry))
TEST_DIR = extracted / "hackathon_dataset/test/images"
assert len(list(TEST_DIR.glob("*.jpg"))) == 3000
print("Ready:", TEST_DIR, "— 3,000 test images")
'''),md('''## Run three fresh measurements

Defaults: FP16, batch size 64, two data-loader workers, all four models sequentially, original + horizontal-flip TTA, and fixed ensemble weights 0.35/0.30/0.20/0.15.

Each run saves checkpoints and evidence. No completed predictions are reused inside timed runs. Initialization and first-batch costs count; filesystem caches are not flushed. This is the ensemble reference pipeline, not the repository's original single-model inference_script.py. Final submission inference still needs to match the selected measured pipeline.

Do not switch to a smaller batch after an OOM and describe it as batch 64. If a run fails, retain its logs, change settings explicitly, and use a new output folder.
'''),code('''from datetime import datetime, timezone

RUNS = 3
BATCH_SIZE = 64
WORKERS = 2
OUTPUT = REPO / "outputs" / ("efficiency_t4_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
OUTPUT.mkdir(parents=True)
with (OUTPUT / "nvidia_smi.txt").open("w") as stream:
    subprocess.run(["nvidia-smi"], stdout=stream, stderr=subprocess.STDOUT, check=True)

command = [sys.executable, "-u", str(REPO / "benchmark_t4.py"),
           "--test-dir", str(TEST_DIR), "--output", str(OUTPUT),
           "--runs", str(RUNS), "--batch-size", str(BATCH_SIZE), "--workers", str(WORKERS)]
print("Running:", " ".join(command), flush=True)
print("Per-batch progress is saved in each run_XX/console.log.", flush=True)
process = subprocess.Popen(command, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
with (OUTPUT / "benchmark_console.log").open("w") as log:
    for line in process.stdout:
        print(line, end="", flush=True)
        log.write(line)
status = process.wait()
print("Exit code:", status, "Outputs:", OUTPUT)
if status:
    print("Benchmark failed. Run the download cell below to preserve diagnostics. No passing timing is claimed.")
'''),md('''## Review the measured report

The report gives end-to-end seconds, images/sec, GPU batch latency, and PyTorch peak allocated/reserved VRAM. GPU batch latency is per model and includes both TTA views; it is not complete ensemble latency. Allocator memory excludes non-PyTorch GPU allocations. Colab timings do not guarantee the judges' timings.
'''),code('''from IPython.display import Markdown, display
import pandas as pd

if (OUTPUT / "summary.csv").exists():
    display(pd.read_csv(OUTPUT / "summary.csv"))
if (OUTPUT / "REPORT.md").exists():
    display(Markdown((OUTPUT / "REPORT.md").read_text()))
else:
    print("No completed report. Check benchmark_console.log and run_XX/console.log.")
'''),md('''## Download all results and diagnostics

The ZIP contains the report, metrics, checkpoint history, per-model/batch timings, environment details, and predictions for each run. Download this even if a run fails, then send it back for review.
'''),code('''from google.colab import files

bundle = shutil.make_archive(str(Path("/content") / (OUTPUT.name + "_results")), "zip", root_dir=OUTPUT)
print("Download:", bundle)
files.download(bundle)
''')]
notebook={'cells':cells,'metadata':{'colab':{'name':'SMPS_T4_Efficiency_Benchmark.ipynb'},
    'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'accelerator':'GPU'},'nbformat':4,'nbformat_minor':5}
for index,cell in enumerate(cells): cell['id']=f't4-cell-{index:02d}'
target=Path(__file__).resolve().parent/'notebooks'/'SMPS_T4_Efficiency_Benchmark.ipynb'
target.parent.mkdir(parents=True,exist_ok=True)
target.write_text(json.dumps(notebook,indent=2,ensure_ascii=False),encoding='utf-8')
print(target)
