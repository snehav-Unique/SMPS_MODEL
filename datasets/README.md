# Official dataset

- `hackathon_dataset.zip`: original dataset archive, tracked with Git LFS.
- `participant_instructions/`: original rules, evaluation metric, inference template and sample submission.
- `asset_inventory.json`: original inventory, recording historical file paths; current paths are in `docs/REORGANIZATION_MANIFEST.json`.
- `extracted/`: ignored local extraction directory.

Download with `git lfs pull --include="datasets/hackathon_dataset.zip"`. Extract locally and set `SMPS_DATASET_ROOT` to the directory containing `train_metadata.csv`. The default is `datasets/extracted/hackathon_dataset`.

Only the official dataset may be used for fine-tuning. Test images are inference inputs and must not be used for training or tuning. Sample predictions are dummy values; the final CSV is root `predictions.csv`.
