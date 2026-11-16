# Experiment logs

## 2026-11-16 - baseline_cv.ipynb created

Findings from the data:
- Train is 416 videos (26 subjects x 2 views x 8 actions), 30 fps, 1280x720, about 434-548 frames each.
- Test is 1,515 clips of 30 jpgs at 640x360.
- Test clips are 30 back-to-back frames (1 second): mean frame-to-frame pixel change is 0.188 in test and 0.205 in train at a gap of 1 frame, against 0.289 at a gap of 2.
- Decoded train frames and test jpgs have matching color statistics, so no color correction is needed.

Pipeline:
- Backbone: frozen DINOv3 ViT-H+ from timm (`vit_huge_plus_patch16_dinov3.lvd1689m`), fp16, input 224x400.
- Frame feature: CLS token + mean of patch tokens, 2560 dimensions.
- Sampling: every 3rd frame (10 fps), so one 1-second window is 10 frames. Test uses frames 01, 04, ..., 28.
- Features are cached in `cache/dinov3_hplus_224x400_step3/`.
- CV: StratifiedGroupKFold, 5 folds, grouped by subject, scored per window with F1-Macro.
- Target: 8 action classes, cheat probability is the sum of the 4 cheat actions, threshold 0.5.
- Model 1: Logistic Regression (C=0.01) on the mean of the 10 frame features.
- Model 2: Temporal Transformer (2 layers, dim 256, 4 heads, dropout 0.3, 20 epochs, AdamW lr 3e-4, label smoothing 0.1, random window positions during training).
- Output: `leon/submissions/baseline_cv.csv` from the Temporal Transformer, plus OOF and test probabilities in `leon/preds/`.

Results:
- Not run to completion yet.
  Fill in the OOF F1-Macro of both models after the first full run.

Bug fix (same day):
- The first full run failed at the Logistic Regression cell with "Input X contains NaN".
- Cause: the backbone was run with fp16 weights (`.half()`), and activations inside ViT-H+ reach about 154,000, above the fp16 limit of 65,504, so every feature came out NaN.
- Fix: weights stay fp32 and the forward pass uses `torch.autocast` with bf16 (same range as fp32), plus an assert that features are finite.
  CLS cosine similarity against full fp32 is 0.99992 at 28 img/s (fp32 is 11.5 img/s).
- The cache folder is now `cache/dinov3_hplus_224x400_step3_bf16/`.
  The old folder `cache/dinov3_hplus_224x400_step3/` holds only NaN and is unused.
- Smoke test on 4 subjects, 2 folds, 40 test clips ran end to end: LR F1-Macro 0.477, Temporal Transformer 0.515.
  These numbers come from training on 2 subjects only, so they say nothing about the real score.

Decisions:
- DINOv3 7B was dropped: it needs about 13.5 GB at fp16 and the GPU has 8 GB.
  A 4-bit version can be compared against H+ later on a subset.
