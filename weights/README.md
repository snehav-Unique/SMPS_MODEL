# Model assets

The final verified CLI uses `ensemble_config.json` and five PyTorch checkpoints: EfficientNetV2-S, ConvNeXt-Tiny, Swin-Tiny, MaxViT-Tiny and DenseNet-121. Configuration entries and filenames are unchanged.

```bash
git lfs pull --include="weights/*"
python inference_script.py --test_dir PATH/test/images --weights weights/ensemble_config.json --output fresh_predictions.csv
```

`best_model.pth` is the earlier baseline. TensorFlow/ONNX exports are preserved and are not used by the final CLI. The EVA-02 checkpoint remains with its results under `outputs/models/eva02_tiny/`. No checkpoints or exports were removed.
