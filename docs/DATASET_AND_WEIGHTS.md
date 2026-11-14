# Current dataset and model setup

Run `git lfs pull` from the repository root. The official archive is `datasets/hackathon_dataset.zip`. Extract under `datasets/extracted/` and set `SMPS_DATASET_ROOT` to the directory containing `train_metadata.csv`; the default is `datasets/extracted/hackathon_dataset`.

The final five checkpoints and `ensemble_config.json` are together under `weights/`. Baseline and export files are preserved there; EVA-02 remains under `outputs/models/eva02_tiny/`. Historical inventory paths are mapped in `REORGANIZATION_MANIFEST.json`.

The original setup notes below are retained as project history. Their sibling participant-package paths and model counts describe the earlier layout.

---

> Repository layout update: run supporting tools from `scripts/`; use the [current README](../README.md) for setup and the [path manifest](REORGANIZATION_MANIFEST.json) for historical locations. Earlier run descriptions below are retained. Final five-model evidence is under `outputs/final_verification/` and `outputs/final_validation/`.

# Dataset and model weights

The repository includes the complete dataset archive and all available trained model checkpoints through Git LFS.

```powershell
git lfs install
git lfs pull
```

## Dataset

`datasets/hackathon_dataset.zip` is the original participant archive. It contains 3,522 fake and 3,478 real training images, 3,000 unlabeled test images, training metadata, and the sample submission. The incomplete duplicate extraction from the original workspace is not needed; the complete original images are preserved in this archive.

Extract the archive into `Deepfake ML Hackathon - Participant Package/extracted` beside the cloned project folder. The resulting dataset root is `Deepfake ML Hackathon - Participant Package/extracted/hackathon_dataset`. The robustness script discovers `train_metadata.csv` under that participant-package directory. Alternatively, place the ZIP itself at `Deepfake ML Hackathon - Participant Package/hackathon_dataset.zip` beside the project and let the script extract it.

The original participant instructions and inference template are included under `datasets/participant_instructions/`. Follow those instructions when using the supplied dataset.

For the original training scripts, update `DATASET_ROOT` in `config.py` to the extracted dataset root.

## Model checkpoints

All seven available `.pth` files are included at their original project paths: baseline EfficientNet-B4, ConvNeXt-Tiny (including its duplicate result copy), EfficientNetV2-S, Swin-Tiny, DenseNet-121, and EVA-02 Tiny. An EfficientNet-B2 checkpoint was not present in the supplied files.

Checkpoint hashes for the four robustness ensemble models are recorded in `outputs/robustness/run_manifest.json`. `datasets/asset_inventory.json` records uploaded dataset and checkpoint sizes and SHA-256 hashes.

GitHub displays small LFS pointer files in Git history; `git lfs pull` downloads the actual binaries. The dataset does not require uploading the extracted duplicate images separately.
