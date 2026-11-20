# Mixed experiment logs (Leon + Fayadh + Steven)

Merged from `leon/logs.md`, `fayadh/logs.md` and Steven's `owen/experiment/` run logs (`run1.log` to `run3.log`, `stageA.log`) on 2026-11-17.
The per-person logs stay the source of truth, this file is the combined view.

## Scoreboard

| Date | Who | Notebook / file | Approach | CV F1-Macro | Public LB |
|---|---|---|---|---|---|
| 11-16 | Fayadh | `baseline.ipynb` / `submission_baseline.csv` | Hand-made motion features + LGBM, per 1 s clip | 0.581 | 0.5737 |
| 11-16 | Leon | `baseline_cv.ipynb` / `baseline_cv.csv` | Frozen DINOv3 ViT-H+ + temporal transformer, per 1 s clip | 0.565 | 0.5765 |
| 11-17 | Steven | `submission.csv` | LeVJEPA-1.5-Large video model (`owen/baseline_levjepa_lgbm.ipynb`) | not logged | 0.4775 |
| 11-17 | Fayadh | `main.ipynb` / `submission_main.csv` | Pose (RT-DETR + ViTPose) + LGBM, rebuilt test videos, Hungarian per group, whole-video labels | 0.885 (video) | 0.6872 |
| 11-17 | Leon | `main.ipynb` / `main.csv` | DINOv3 per video + LR, joint front+side Hungarian, whole-video labels | 0.9615 (video) | not submitted |
| 11-17 | Leon | `leon/submissions/submission.csv` (= `main_from7.csv`) | Same video labels, cheat only from second 7 | 0.9615 (video), 0.923 (train-as-test, per clip) | **0.9600702** |
| 11-17 | Steven | `owen/experiment/submissions/submission_take_hung_from7.csv` | Pose + DINOv3 LR ensemble, take-level Hungarian, cheat from second 7 | 0.9904 (video) | not known |
| 11-17 | Leon | `leon/submissions/main_model_only.csv` | Model only: no fixed second, no 4/4 balance | 0.846 (per clip) | not submitted |
| 11-17 | Leon | `leon/submissions/main_model_only_window.csv` | Same, start search 4-11 s | 0.890 (per clip) | not submitted |
| 11-17 | Leon | `leon/main_legit.ipynb` / `main_legit.csv` | Train only: DINOv3 + pose LR, no Hungarian, start window from by-eye train starts | 0.873 (video), 0.894 (per clip) | not submitted |
| 11-18 | Leon | `leon/main_legit.ipynb` / `main_legit.csv` (updated) | Same + per-person centering (each video minus its person-and-camera mean) | 0.942 (video), 0.942 (per clip) | not submitted; 0.945 per clip vs the T1 swap file |
| 11-18 | Leon | `leon/submissions/main_legit_pair_window.csv` | Same + camera pairing by timing and action agreement + start search 4.25-9.25 s | 0.952 (video), train pairing 100% | not submitted; 0.961 per clip vs the T1 swap file |
| 11-18 | Leon | `leon/main_legit.ipynb` / `main_legit.csv` (updated) | Centering + action-agreement camera pairing (start search still 3.2-11.1 s) | 0.952 (video), 0.949 (per clip) | not submitted; 0.960 per clip vs the T1 swap file |
| 11-19 | Leon | `leon/main_legit.ipynb` / `main_legit.csv` (updated) | Centering + action-agreement camera pairing + start search 4.25-9.25 s, no fallback; same output as `main_legit_pair_window.csv` (renamed by Leon to `leon/submissions/submission.csv`) | 0.952 (video), 0.952 (per clip) | not submitted; 0.961 per clip vs the T1 swap file (now `potential_1.csv`) |
| 11-19 | Leon | `leon/main_legit_nosequence.ipynb` / `main_legit_nosequence.csv` | Every test sequence on its own: train cut into 1-second clips with per-second labels, DINOv3 + pose per clip, one LR | 0.742 (per clip) | not submitted; 0.706 per clip vs `1.csv` |
| 11-19 | Leon | scratch `ft_train.py` K=0 | DINOv3 ViT-H+ frozen + new 8-action head on frames, used in main_legit and nosequence | main_legit 0.935 (add) / 0.909 (replace); nosequence 0.742 | not submitted; main_legit 0.9606 / 0.9504 vs `1.csv`, nosequence 0.7117 |
| 11-19 | Leon | scratch `ft_train.py` K=12 / `leon/submissions/nosequence_ft12.csv` | No-sequence LR on the backbone features of the top 12 ViT-H+ blocks fine-tuned (5 folds by video subject, test = fold average) + pose, each clip alone; 2 of 30 test frames | 0.789 (per clip) | not submitted yet; 0.809 per clip vs `1.csv` |
| 11-20 | Leon | scratch `e30c.py` (E3) | No-sequence BiGRU / small transformer over the clip's 30 frames: DINOv3 frame vectors (PCA 128) minus the clip mean + 30-frame pose signals, each clip alone | 0.772 / 0.773 (per clip) | not submitted; 0.762 / 0.743 per clip vs `1.csv` |
| 11-20 | Leon | `leon/submissions/nosequence_blend_v1.csv` | Rank blend of the 12-block fine-tuned backbone + pose (weight 2), E3 BiGRU (1), E3 transformer (1); weights and threshold from held-out train; B part still uses 2 of 30 test frames | 0.816 (per clip) | not submitted yet; 0.822 per clip vs `1.csv` |
| 11-20 | Leon | scratch `d30.py` / `e_lev.py` (D1, E5) | No-sequence LeVJEPA-1.5-Large (frozen video model, all 30 frames as two interleaved halves): LR on summary + pose (D1), BiGRU / transformer on clip-mean-centred per-frame vectors + pose (E5), each clip alone | 0.737 (D1), 0.751 / 0.769 (E5) (per clip) | not submitted; 0.768 (D1), 0.779 / 0.788 (E5) per clip vs `1.csv` |
| 11-20 | Leon | `leon/submissions/nosequence_blend_v2.csv` | Rank blend: fine-tuned backbone + pose (2), E3 BiGRU (1), E3 transformer (1), LeVJEPA LR (1), E5 transformer (1); weights and threshold 0.70 from held-out train; B part still uses 2 of 30 test frames | 0.827 (per clip) | **0.8518237** (submitted as `leon/submissions/submission.csv`); 0.834 per clip vs `1.csv` |
| 11-17 | Steven | `leon/submissions/main_pairfix_from7.csv` (= `take_hung_from7`) | 0.960 file with the `subject_T3` side_01/side_03 pairing swapped | - | **0.9761476** (submitted as `submission_owen.csv`) |
| 11-18 | Leon | `leon/submissions/main_start_guard.csv` | Steven's video labels, per-video EM front+side start, starts outside 5.5-9.5 s -> 7.75 s | about 6.5 public errors expected (train sim) | 0.9626489 (14 wrong, worse than 9) |
| 11-18 | Leon | `leon/submissions/main_t1swap_from7.csv` (now `leon/submissions/1.csv`) | Steven's file + `subject_T1` swap: front_04 + side_01 cheat from 7, front_08 + side_07 normal | - | **1.0000000** (submitted 11-19 as `submission_t1.csv`) |

Video CV = cheat/normal per video; per-clip CV = F1 over 1-second clips against per-second train starts read by eye.

## Shared findings

Data:
- Train is 416 `.MOV` videos: 26 subjects x 2 views x 8 actions, 1280x720, 30 fps, 407-558 frames (13.6-18.6 s).
- Test is 1,515 clips of 30 jpgs at 640x360, not 1280x720 as the competition page says.
- Test ids are shuffled and file times follow the id order, so there is no ordering leak.
- Test subjects are new people.

Test structure (found independently by all three, results agree):
- Each clip continues into another clip: frame 30 of one clip is almost identical to frame 01 of the next.
- Chaining with mutual nearest neighbours gives 96 test videos of 14-19 seconds (Fayadh: 97 chains plus 2 half chains that merge).
- 48 videos are front and 48 are side; the view is predicted with 100% CV accuracy.
- Per view the videos split into 6 groups of 8, so test is 6 subjects x 2 views x 8 videos, the same layout as train.
  Fayadh grouped by HSV colour histogram of the person crop, Leon by Ward clustering of DINOv3 features, Steven by appearance with groups of 8.
- Front and side videos of one take are filmed at the same time, so their motion rises and falls together.
  Pairing them by motion cross-correlation: Leon 95.7% right on train (DINOv3 motion), Steven 208/208 on train (pose motion, same take 0.830 against other 0.372).
- Leon wrote the rebuilt test to `data/test_rebuilt/` (`mapping.csv` and one mp4 per video).

Modelling:
- One second is too little: per-clip models stay at 0.56-0.58 in CV and on the LB.
- Predicting 8 actions and mapping to cheat/normal works better than predicting 2 classes.
- Assigning the 8 actions to the 8 videos of a subject with Hungarian gives the largest single gain (Fayadh 0.805 to 0.885, Leon 0.863 to 0.952).
- Solving front and side together (one action per take) helps again (Leon 0.9519 to 0.9615).
- DINOv3 and pose carry different information: Steven's ensemble of the two reaches 0.9904 video CV against 0.9616 (DINOv3) and 0.9422 (pose).
- Features that describe change over time matter most: z-scored video pose features, DINOv3 std and time-lag features.
- Logistic regression beats LightGBM on the dense DINOv3 features (0.9519 against 0.9199).

## Labels are per second, not per video

- Every train and test video starts with about 7-8 seconds of plain writing; the action starts after that.
- Exact LB decimals decode into whole counts (TP, errors, TN): the public set has exactly 455 clips.
- Whole-video labels are ruled out by those counts; clips of a cheat video only count as cheat from about second 6-7.
  This explains the old CV against LB gap (Fayadh 0.885 against 0.687).
- `main_from7` scored 0.9600702: 125 cheat clips right, 315 normal clips right, 15 wrong.
- The real start varies per video (by eye on train: about 4.75-9 s, median about 7.5 s), so a fixed second leaves boundary errors.

## Timeline

### 2026-11-16 - Fayadh - `baseline.ipynb` (motion + HOG, LGBM)

- Train frames resized to 640x360 and passed through JPEG (q=90) to look like test, motion computed at 160x90 grayscale.
- Train clips: 30 frames, stride 15, 12,717 clips; LGBM binary, StratifiedGroupKFold(5) by subject.

| Variant | CV F1 (thr 0.5) | CV F1 (tuned thr) | Video-averaged F1 |
|---|---|---|---|
| All features (motion + HOG) | 0.5126 | 0.5222 | 0.537 |
| Motion only, 152 features (final baseline) | 0.5811 | 0.5811 | 0.634 |
| Motion z-scored per subject x view | 0.532 | 0.532 | 0.616 |

- HOG took 97.8% of the importance and recognised the subject, not the action, so it was switched off.
- Public LB 0.5737.

### 2026-11-16 - Leon - `baseline_cv.ipynb`

- Frozen DINOv3 ViT-H+ (`vit_huge_plus_patch16_dinov3.lvd1689m`), input 224x400, CLS + mean patch (2560 dims), 10 fps, cached in `cache/dinov3_hplus_224x400_step3_bf16/`.
- fp16 weights overflowed inside ViT-H+ (all NaN); fixed with fp32 weights and bf16 autocast.
- LR OOF 0.5654, temporal transformer OOF 0.5645; public LB 0.5765.

### 2026-11-17 - Steven - LeVJEPA baseline

- `owen/baseline_levjepa_lgbm.ipynb`: LeVJEPA-1.5-Large features + LGBM, public LB 0.4775.
- Decoded counts: TP 10, errors 133, TN 312, so at least 68.6% of the public clips are normal.

### 2026-11-17 - Fayadh - `main.ipynb` (pose + test structure + Hungarian)

- DINOv2-small frozen + LogReg first: best video + Hungarian 0.659.
- Pose: RT-DETR `PekingU/rtdetr_r50vd_coco_o365` person box + ViTPose `usyd-community/vitpose-base-simple` 17 keypoints at 10 fps (fp32, fp16 was 4x slower on the GTX 1650).

| Variant | clip | video | 4/4 balance | Hungarian | 8-action acc (video) |
|---|---|---|---|---|---|
| Pose per window (131 features) | 0.621 | 0.702 | 0.740 | 0.798 | 0.567 |
| + video context | 0.771 | 0.766 | 0.789 | 0.846 | 0.673 |
| + video mean z-scored per subject-view | 0.798 | 0.795 | 0.808 | 0.880 | 0.743 |
| Final (z-scored video mean, 300 trees, 3 seeds) | 0.806 | 0.805 | 0.808 | 0.885 | not logged |

- Main confusions: Looking Around against Repeated Peeking, Checking against Using Smartwatch.
- Public LB 0.6872 (whole-video labels).

### 2026-11-17 - Leon - `main.ipynb` (rebuilt test + DINOv3 per video)

- Video vector: mean + std + std of feature differences at lags 5, 15, 50 (12,800 dims), LR C=0.3.
- OOF: argmax 0.8628, Hungarian per group 0.9519, joint front+side 0.9615 (PAIR_WEIGHT 3.0).
- Did not help: min/max, quantiles, segment means, centring per subject-view, LightGBM.

### 2026-11-17 - Leon - per-second labels and `main_from7`

- LB decoding (see above) pointed to "cheat from second 7"; `main_from7.csv` submitted as `leon/submissions/submission.csv`: **0.9600702**.

### 2026-11-17 - Leon - start of the action per video

- By-eye starts on train (train only): `leon/train_action_start_eye.csv` (30 videos) and `leon/train_action_start_eye_all.csv` (all 104 cheat front videos, 95 readable).
- EM start finder: per-frame "writing or action" LR on frames minus the video's first 3 s, one switch per video, front and side solved together.
- Start error on 187 cheat videos: median 0.25 s, 89% within 1 s.

### 2026-11-17 - Leon - train-as-test CV (per clip)

- Held-out train videos cut into 1-second clips from frame 0, shuffled, rebuilt with the test pipeline, scored per clip against the by-eye starts.
- Rebuild on train: 414 of 418 videos exact, view 100%, subject groups exact in 3 of 5 folds.

| Rule inside a cheat video | F1 per clip |
|---|---|
| Whole video | 0.708 |
| From second 6 / 7 / 8 | 0.892 / 0.923 / 0.932 |
| EM front+side start | **0.937** |
| EM front+side start - 0.5 s / - 1 s | 0.926 / 0.910 |

- With true video labels the best rule reaches 0.979, so the remaining loss is mostly wrong video labels.

### 2026-11-17 - Steven - `owen/experiment/puzzle_take.py` ("6 puzzles", take-level)

- Rebuilds 96 test videos, pose features (RT-DETR + ViTPose, train 2,378 s, test 360 s), window features (131) and video features (418).
- Features z-scored within each subject x view group; front and side paired into 48 takes (test pair correlation median 0.826, min 0.401).
- Hungarian over the 8 takes of a subject (one action per take, both views at once).

Video CV F1 (thr = per-video threshold, angle_hung = Hungarian per view, take_hung = Hungarian per take):

| Model | thr | angle_hung | take_hung |
|---|---|---|---|
| Pose LR | 0.8688 | 0.9470 | 0.9422 |
| Pose LGBM | 0.8708 | 0.9275 | 0.9232 |
| Pose window LGBM | 0.8105 | 0.8841 | 0.8848 |
| Leon's DINOv3 LR (saved OOF) | 0.8586 | 0.9524 | 0.9616 |
| **DINOv3 + pose LR** (used) | 0.9170 | 0.9814 | **0.9904** |
| DINOv3 + pose LR + LGBM | 0.9215 | 0.9710 | 0.9904 |

- 4 train videos still wrong, all from subject 6.
- Least certain test takes: Thinking against Reading_Hidden_Notes (margin 0.89), then Repeated_Peeking against Looking_Around (4.58), then Using against Checking_Smartwatch (5.24).
  The first is the same spot Leon's model ranks closest (`subject_T1` front_04/side_01 against front_08/side_07, cost 1.47).
- Files in `owen/experiment/submissions/`: `submission_take_hung.csv` (whole-video labels, cheat share 0.506), `_from5`, `_from6`, `_from7`, `submission_angle_hung.csv`, `submission_thr.csv`, `submission_onset.csv`, `submission_from7_swapT3.csv`.
- Against Leon's 0.9600702 file, `submission_take_hung_from7.csv` differs on only 2 videos (19 clips): `subject_T3/side_video_01` becomes cheat and `subject_T3/side_video_03` becomes normal.
  So the two models agree on every take except which side video is the Using_Smartwatch twin of `subject_T3/front_video_02`.
- `submission_from7_swapT3.csv` also swaps the uncertain Thinking / Reading_Hidden_Notes takes (53 clips differ from the 0.960 file).

### 2026-11-17 - Leon - model only (no fixed second, no 4/4 balance, no LB feedback)

| Setup | F1 per clip |
|---|---|
| Every clip = video argmax | 0.679 |
| Per-clip LR, clip minus its video's first 3 s, times video probability | 0.820 |
| EM start anywhere, init middle of video, front+side, no Hungarian | **0.846** |
| EM start in 4-11 s, init 7.5 s, front+side, no Hungarian | 0.890 |

- Video cheat/normal without Hungarian: 0.858 single view, 0.870 front x side product; this is now the main loss.
- Test files: `leon/submissions/main_model_only.csv` (44 of 96 videos cheat, 136 clips differ from the 0.960 file) and `main_model_only_window.csv` (108 clips differ).

### 2026-11-17 - Leon - `main_legit.ipynb` (train only) and data insights

- Pose keypoints extracted for train and test with `leon/scripts/extract_pose.py` into `cache/pose_vitpose/` (one window per 1-second clip).
- DINOv3 + pose LR, no Hungarian, front x side camera product, start finder with a window taken from the by-eye train starts (3.2-11.1 s).
- Video F1 0.873 (DINOv3 0.858, pose 0.793), per-clip F1 0.894, 0.977 with true video labels.
- Against the 0.960 file as pseudo ground truth: F1 0.906.
  It calls `subject_T1` front_04 + side_01 cheat, the closest call in both Hungarian models too.
- Data insights for the presentation (start time per action, camera lag, signature gesture per action, same pattern in test) are in `leon/logs.md`, section "Data insights for the presentation (EDA)".

### 2026-11-18 - Steven: pairfix scored 0.9761476
- `submission_owen.csv` = `main_pairfix_from7.csv` = `take_hung_from7`: public LB 0.9761476, decoded 129 cheat right, 317 normal right, 9 wrong (from 15).
- The `subject_T3` side_01 (now cheat) / side_03 (now normal) swap is right: all changed public clips were fixed.
- Next candidate: `subject_T1` front_04 + side_01 cheat, front_08 + side_07 normal (about 10 public clips against 9 errors left).

## Ideas not tried yet

- Combine DINOv3 and pose in the model-only setup (Steven's ensemble shows they add up).
- Hand or wrist crop features for Checking against Using Smartwatch.
- Head-turn periodicity features for Looking Around against Repeated Peeking.
- Force exactly 8 videos per subject group in the rebuild (2 of 5 train folds got a 7/9 split).
- Actions that end before the video ends (seen in 2 train videos).
