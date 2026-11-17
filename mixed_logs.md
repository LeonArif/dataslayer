# Mixed experiment logs (Leon + Fayadh)

Merged from `leon/logs.md` and `fayadh/logs.md` on 2026-11-17.
The per-person logs stay the source of truth, this file is the combined view in time order.

## Scoreboard

| Date | Who | Notebook / file | Approach | CV F1-Macro | Public LB |
|---|---|---|---|---|---|
| 11-16 | Fayadh | `baseline.ipynb` / `submission_baseline.csv` | Hand-made motion features + LGBM, per 1 s clip | 0.581 | 0.5737 |
| 11-16 | Leon | `baseline_cv.ipynb` / `baseline_cv.csv` | Frozen DINOv3 ViT-H+ + temporal transformer, per 1 s clip | 0.565 | 0.5765 |
| 11-17 | Steven | `submission.csv` | LeVJEPA-1.5-Large video model (details not logged) | not logged | 0.4775 |
| 11-17 | Fayadh | `main.ipynb` / `submission_main.csv` | Pose (RT-DETR + ViTPose) + LGBM, rebuilt test videos, Hungarian per group | 0.885 | 0.6872 |
| 11-17 | Fayadh | `submission_main_video.csv` | Same, per video without Hungarian | 0.805 | not submitted |
| 11-17 | Leon | `main.ipynb` (first version) | DINOv3 mean+std per video + LR, Hungarian per group | 0.904 | not submitted |
| 11-17 | Leon | `main.ipynb` / `main.csv` | Same + time-lag features (0.5 s, 1.5 s, 5 s) | 0.952 | not submitted |

Top of the public LB is 1.0000 (3 teams).

## Shared findings

Data:
- Train is 416 `.MOV` videos: 26 subjects x 2 views x 8 actions, 1280x720, 30 fps, 407-558 frames (13.6-18.6 s).
- Test is 1,515 clips of 30 jpgs at 640x360, not 1280x720 as the competition page says.
- Test ids are shuffled and file times follow the id order, so there is no ordering leak.
- Test subjects are new people: the nearest train frame is as far away as a frame of a different train subject.

Test structure (found independently by both, results agree):
- Each clip continues into another clip: frame 30 of one clip is almost identical to frame 01 of the next.
- Chaining with mutual nearest neighbours gives 96 test videos of 14-19 seconds.
  Fayadh got 97 chains plus 2 half chains that merge into one video, Leon got 96 directly with a pixel-difference limit of 3.0.
- 48 videos are front and 48 are side.
- Per view the videos split into 6 groups of 8, so test is 6 subjects x 2 views x 8 videos, the same layout as train.
  Fayadh grouped by HSV colour histogram of the person crop and checked all 96 videos by eye, Leon grouped by Ward clustering of DINOv3 features.
- Leon wrote the rebuilt test to `data/test_rebuilt/` (`mapping.csv` and one mp4 per video).

Modelling:
- One second is too little: per-clip models stay at 0.56-0.58 in CV and on the LB, for both hand-made motion features and DINOv3.
- CV grouped by subject matches the LB for per-clip models (0.581 against 0.5737, 0.565 against 0.5765).
- Predicting 8 actions and mapping to cheat/normal works better than predicting 2 classes.
- Hungarian assignment of 8 actions to the 8 videos of a subject-view group gives the largest single gain in CV for both (Fayadh 0.805 to 0.885, Leon 0.863 to 0.952).
- Features that describe change over time matter most: Fayadh's video-level z-scored pose features, Leon's std and time-lag features.

## Open problem: CV against LB gap for video-level models

- Fayadh's `submission_main.csv` has CV 0.885 but public LB 0.687.
- Fayadh's checks: the test grouping is correct by eye, frame tempo matches train, model confidence on test matches OOF.
- Leon's checks for the DINOv3 model: feature magnitudes, clip-boundary continuity and model confidence on test all match train.
- Leon's `main.csv` agrees with Fayadh's `submission_main.csv` on 84.8% of clips.
  If the public LB is a random 30% of clips, Leon's file can score at most about 0.84, far below its CV of 0.95.
- Hypotheses still open:
  1. The public LB is not a random sample of clips (for example a few subjects only), so its variance is large.
  2. Test labels are per second, not per video.
  3. Test groups do not hold exactly one video per action, so the Hungarian step is wrong on test.
- Probes that would separate them: submit Leon's `main.csv`, submit Fayadh's `submission_main_video.csv` (no Hungarian), and submit an all-ones file to read the cheat share of the public LB.

Update (Leon, 2026-11-17): hypothesis 2 is now the most likely one.
- Every train and test video starts with about 7-8 seconds of plain writing, and the action starts around second 7-8.
- If only the clips from second 7 onward of a cheat video count as cheat, the predicted LB scores are 0.676 for Fayadh's main (real 0.687) and 0.585 for Leon's baseline_cv (real 0.5765).
  With whole-video labels they would be 0.848 and 0.638.
- The exact boundary is not known, so Leon wrote `leon/submissions/main_from6.csv`, `main_from7.csv` and `main_from8.csv` to test it.
- Details and the full table are in `leon/logs.md`.

## Rules risk

- The rules forbid external data.
  Pretrained weights (DINOv3, RT-DETR, ViTPose, LeVJEPA) are used by team decision and still need to be confirmed with the committee.
- The rules forbid manual prediction and hand labelling of test.
  The test rebuild is fully automatic from pixels.

## Timeline

### 2026-11-16 - Fayadh - `baseline.ipynb` (motion + HOG, LGBM)

- Train frames are resized to 640x360 and passed through JPEG (q=90) to look like test, then grayscale, motion computed at 160x90.
- Train clips: 30 frames, stride 15, 12,717 clips.
- Features (908): frame-difference energy, 6x8 motion grid, Farneback optical flow statistics and 3x4 grid, HOG of the mean frame.
- Model: LGBM binary, 400 trees, lr 0.03, 31 leaves, colsample 0.5, StratifiedGroupKFold(5) by subject.

| Variant | CV F1 (thr 0.5) | CV F1 (tuned thr) | Video-averaged F1 |
|---|---|---|---|
| All features (motion + HOG) | 0.5126 | 0.5222 | 0.537 |
| Motion only, 152 features (final baseline) | 0.5811 | 0.5811 | 0.634 |
| Motion z-scored per subject x view | 0.532 | 0.532 | 0.616 |

- HOG took 97.8% of the importance and made the model recognise the subject or view, not the action, so it was switched off.
- Hardest subjects: 6 (0.43), 15 (0.46), 16 (0.50).
- Public LB 0.5737.

### 2026-11-16 - Leon - `baseline_cv.ipynb` created

- Backbone: frozen DINOv3 ViT-H+ from timm (`vit_huge_plus_patch16_dinov3.lvd1689m`), input 224x400.
- Frame feature: CLS token + mean of patch tokens, 2560 dimensions, every 3rd frame (10 fps).
- Features cached in `cache/dinov3_hplus_224x400_step3_bf16/`.
- Models on 1-second windows: Logistic Regression (C=0.01) on the window mean, and a temporal transformer (2 layers, dim 256).
- Bug fixed: fp16 weights overflowed inside ViT-H+ (activations reach about 154,000, fp16 limit is 65,504) and every feature was NaN.
  Fix: fp32 weights with bf16 autocast.
- DINOv3 7B was dropped because it needs about 13.5 GB and the GPU has 8 GB.

### 2026-11-17 - Leon - `baseline_cv.ipynb` results and test structure check

- Logistic Regression OOF 0.5654, temporal transformer OOF 0.5645, 8-action accuracy about 0.26.
- Public LB 0.5765 for the temporal transformer.
- Test clips chain into longer videos and cluster into about 12 groups of roughly 125 clips.

### 2026-11-17 - Fayadh - `main.ipynb` (pose + test structure + Hungarian)

- First try, DINOv2-small frozen (448x252, CLS + mean patch + 3x4 grid, 5 fps) + LogReg: clip 0.52-0.55, best video + Hungarian 0.659, worse than the motion baseline.
- Pose: RT-DETR `PekingU/rtdetr_r50vd_coco_o365` for the person box, ViTPose `usyd-community/vitpose-base-simple` for 17 keypoints at 10 fps.
- On the GTX 1650, fp16 is 4x slower than fp32, so inference runs in fp32.
- Model: LGBM multiclass on 8 actions, StratifiedGroupKFold(5) by subject, 30-frame windows without overlap.

| Variant | clip | video | 4/4 balance | Hungarian | 8-action acc (video) |
|---|---|---|---|---|---|
| Motion baseline | 0.581 | 0.634 | not run | not run | not run |
| DINOv2 base+grid+std+diff, LogReg | 0.551 | 0.563 | 0.591 | 0.659 | 0.320 |
| Pose per window (131 features) | 0.621 | 0.702 | 0.740 | 0.798 | 0.567 |
| + video context | 0.771 | 0.766 | 0.789 | 0.846 | 0.673 |
| + video mean z-scored per subject-view | 0.798 | 0.795 | 0.808 | 0.880 | 0.743 |
| + window z-scored per subject-view | 0.806 | 0.805 | 0.813 | 0.865 | 0.743 |
| Final (z-scored video mean, 300 trees, 3 seeds) | 0.806 | 0.805 | 0.808 | 0.885 | not logged |

- Importance: z-scored video features 78%, video features 21%, window features near 0.
- Main confusions: Looking Around against Repeated Peeking, Checking against Using Smartwatch.
- Public LB 0.6872 against CV 0.885.

### 2026-11-17 - Leon - `main.ipynb` (rebuilt test + DINOv3 per video + Hungarian)

- Reuses the cached DINOv3 ViT-H+ features, no new extraction.
- First version: mean + std over time per video, LR C=0.1: argmax 0.7853, Hungarian 0.9038.
- Current version: + std of feature differences at lags 5, 15, 50 samples, LR C=0.3: argmax 0.8628, Hungarian 0.9519, 8-action accuracy 0.9327.
- Did not help: min/max, quantiles, segment means, mean absolute deviation, centring per subject-view (0.835).
- Mean only gives 0.755 with Hungarian, so the change-over-time features carry most of the signal.

## Ideas not tried yet

- Fayadh: head-turn periodicity features to split Looking Around from Repeated Peeking.
- Fayadh: sequence model (GRU / 1D-CNN) on per-second pose features.
- Fayadh: hand or wrist crop + DINO for Checking against Using Smartwatch.
- Leon: combine DINOv3 video features with Fayadh's pose features in one model.
- Leon: fine-tune a backbone on whole videos.
