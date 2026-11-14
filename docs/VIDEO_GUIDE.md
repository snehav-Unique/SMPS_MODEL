> Repository layout update: run supporting tools from `scripts/`; use the [current README](../README.md) for setup and the [path manifest](REORGANIZATION_MANIFEST.json) for historical locations. Earlier run descriptions below are retained. Final five-model evidence is under `outputs/final_verification/` and `outputs/final_validation/`.

# Team SMPS - 3 to 5 minute dashboard video

Follow the graphs from top to bottom. Record the completed page; no new training or inference run is needed.

## What is distinctive?

We combine five established vision architectures, average original and horizontal-flip predictions, test image corruptions with uncertainty intervals, visualize attribution using Grad-CAM, and verify that the final inference script independently reproduces all 3,000 predictions. This is a combination of engineering and evaluation methods, not a claim that we invented these architectures or proved an ensemble improvement.

## Opening - 0:00 to 0:25

"Hello, we are Team SMPS. Our project classifies face images as real or fake. We fine-tune established vision architectures using the supplied dataset and combine five models into one probability-based ensemble. Our focus is prediction quality, robustness, efficient inference and reproducibility."

## Graph 1: ROC and confusion matrix - 0:25 to 1:05

"This graph shows the final five-model ensemble on 200 balanced, held-out validation images. The ROC curve shows how the true-positive rate changes with the false-positive rate across decision thresholds. Its AUC is 0.8817.

"At a threshold of 0.5, we correctly classify 87 real images and 88 fake images. Thirteen real images are flagged as fake, and twelve fake images are missed. That gives 87.5 percent accuracy on this subset. AUC and accuracy measure different things."

## Graph 2: Architecture comparison - 1:05 to 1:45

"We combine EfficientNetV2-S, ConvNeXt-Tiny, Swin-Tiny, MaxViT-Tiny and DenseNet-121. These include convolutional networks and attention-based architectures, giving us different ways to represent an image.

"The bars show each model's recorded peak validation AUC from training. EfficientNetV2-S has the highest recorded individual value, 0.9025. These peaks use a different evaluation sample from the final 200-image chart, so we do not compare those numbers directly.

"For each model we average original and horizontally flipped predictions. Then we blend probabilities with weights of 40, 25, 15, 12 and 8 percent respectively. These are fixed heuristic weights, not a proven optimum."

## Graph 3: Learning curves - 1:45 to 2:10

"These two training histories show why checkpoint selection matters. Training AUC continues rising while validation AUC levels off. This gap indicates overfitting. We select checkpoints using held-out validation performance instead of automatically using the last epoch."

## Graph 4: Robustness - 2:10 to 2:50

"We evaluate JPEG compression, Gaussian blur and Gaussian noise on the final ensemble. Clean validation AUC is 0.8817. Under noise it is 0.8707, giving the largest observed absolute AUC drop of 0.0110.

"The error bars show 95 percent bootstrap intervals based on 2,000 resamples. They make the uncertainty visible. These results show modest observed degradation on this sample, but they do not establish a strict population guarantee."

## Graph 5: Confidence distribution - 2:50 to 3:10

"This histogram contains predictions for all 3,000 official test images. Confidence is the predicted probability of Fake. A threshold of 0.5 converts that probability into a binary label. Without private test labels, this distribution does not establish test accuracy or probability calibration."

## Graph 6: T4 efficiency - 3:10 to 3:35

"The final FP32 five-model pipeline processes all 3,000 images in 172.34 seconds on a Tesla T4, below the five-minute scoring boundary. This is end-to-end process time, not only the model's forward-pass time.

"We also ran the final inference pipeline independently again. All 3,000 labels and probabilities matched the newly generated submission CSV exactly."

## Grad-CAM examples - 3:35 to 4:00

"Each example shows the original image, a Grad-CAM heatmap and an overlay. These help us inspect regions influencing the detector and look for possible shortcuts. A heatmap is a qualitative diagnostic, not proof that a decision is correct or that the highlighted region caused it."

## Closing - 4:00 to 4:20

"Our deliverables include final predictions, inference code, trained weights, a concise report and reproducibility evidence. The strengths are the architecture combination, corruption testing, measured speed and exact replay of final predictions. Our official score remains unknown because test labels are private."

## Recording and submission

Refresh results.html for the two-second blank screen and entrance animation. Scroll in graph order; click charts to enlarge them and press Escape to close. The webpage contains no narration or code.

Use final predictions.csv with its matching inference_script.py and ensemble_worker_control.py. predictions_original.csv is retained for audit and is not the final submission file. The results ZIP excludes large model weights, which must accompany the submission.
