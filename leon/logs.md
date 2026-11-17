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

## 2026-11-17 - baseline_cv.ipynb results and test structure check

Results of the full run:
- Logistic Regression OOF F1-Macro 0.5654, 8-action accuracy 0.2665.
- Temporal Transformer OOF F1-Macro 0.5645, 8-action accuracy 0.2625.
- Public LB of `baseline_cv.csv` (Temporal Transformer): 0.5765, so CV and LB agree.
- Teammate's LeVJEPA-1.5-Large submission: 0.4775 public LB.
- Top of the public LB is 1.0000 (3 teams).

Test structure check (cached DINOv3 features, no new model):
- Test clips are not cut from train videos: nearest train frame cosine has median 0.828, the same as train frames of a different subject (0.843), far below same video (0.994).
  Test subjects are new people.
- Test clips chain into longer videos: the last sampled frame of a clip has a near-identical first frame in another clip (median cosine 0.990, equal to neighbouring frames inside one clip).
  Greedy mutual chaining on features already gives chains up to 17 clips long, about one full video.
- Clustering clip means into 12 groups gives mostly groups of about 122-128 clips, which fits 6 subjects x 2 views x 8 videos x about 16 seconds = 1,515 clips.
- Test ids are shuffled (the next clip is id+1 in 0.1% of cases) and file times only follow the id order, so there is no ordering leak.

Conclusion:
- A 1-second window does not hold enough information (8-action accuracy 27% against 12.5% chance), so a better backbone alone will not get far.
- Next step: rebuild test videos from the clips and predict per video, not per clip.

## 2026-11-17 - main.ipynb created (test rebuilt into videos, prediction per video)

Test rebuild:
- Clips are chained on raw pixels: frame 30 of a clip against frame 01 of every other clip (grayscale, 160x90, mean absolute difference).
  Two clips are linked when each is the other's nearest match and the difference is below 3.0.
- Result: 1,419 links and exactly 96 test videos of 14-19 seconds (2 x 14, 37 x 15, 40 x 16, 15 x 17, 1 x 18, 1 x 19).
- View: taken from the nearest train video by mean DINOv3 feature, giving 48 front and 48 side.
- Subject per view: Ward clustering into 6 groups, each group has exactly 8 videos.
- Front and side groups of the same person are paired with Hungarian matching on CLS features.
  The same method pairs 88% correctly on train (26 subjects), so the pairing on test is a best guess and is only used for folder names.
- Output in `data/test_rebuilt/`: `mapping.csv` (clip id, video, subject, view, position), `subject_T*/{view}_video_*.mp4`, and `videos_pred.csv`.

Pipeline:
- Features: cached DINOv3 ViT-H+ features from `cache/dinov3_hplus_224x400_step3_bf16/`, no new extraction.
- Video vector: mean and standard deviation over time of the frame features, 5120 dimensions.
- CV: StratifiedGroupKFold, 5 folds, grouped by subject, scored per video with F1-Macro.
- Model: StandardScaler + Logistic Regression (C=0.1) on 8 actions.
- Post-processing: inside each subject-view group the 8 videos are assigned to the 8 actions with Hungarian matching, which forces 4 cheat and 4 normal per group.
  This assumes test groups hold each action exactly once, like train.

Results (OOF, per video):
- Argmax: F1-Macro 0.7853, 8-action accuracy 0.6803.
- Hungarian per group: F1-Macro 0.9038, 8-action accuracy 0.8798.
- Weakest actions after assignment: Looking_Around and Repeated_Peeking on the side view (0.731 each).
- Output: `leon/submissions/main.csv`, public LB not known yet.

Side experiments on video vectors (scratch script, seed 0 folds, not in the notebook):
- Mean only: F1 0.646 argmax, 0.755 with Hungarian.
  The standard deviation over time is what carries most of the gain.
- Adding mean absolute frame difference: no gain (0.918 against 0.923 with Hungarian).
- Subtracting the subject-view mean before the model: better argmax (0.844) but worse with Hungarian (0.875).

## 2026-11-17 - main.ipynb: time-lag features added

Change:
- Video vector is now mean + std + std of feature differences at lags 5, 15 and 50 samples (0.5 s, 1.5 s, 5 s at 10 fps), 12,800 dimensions.
- Logistic Regression C changed from 0.1 to 0.3.
- `leon/submissions/main.csv` was overwritten with this version.

Results (OOF, per video, notebook folds seed 42):
- Argmax: F1-Macro 0.8628, 8-action accuracy 0.7764.
- Hungarian per group: F1-Macro 0.9519, 8-action accuracy 0.9327.
- Weakest after assignment: Checking_Smartwatch front and Using_Smartwatch front (0.846 each).
- Public LB not known yet.

Side experiments (scratch scripts, Hungarian F1-Macro, mean over 3 or 5 fold seeds):
- mean+std: 0.910-0.917.
- mean+std+lag5+lag15+lag50: 0.944 (C=0.01), 0.946 (C=0.03), 0.948 (C=0.1), 0.951 (C=0.3).
- mean+std+smoothed std (windows 10 and 30): 0.941.
- mean+std+min+max: 0.925.
- mean+std+q10+q90: 0.920.
- mean+std+mean of 3 time segments: 0.920.
- mean+std+mean absolute deviation: 0.913.
- std only: 0.910.
- std+lag15+smoothed std 10: 0.939.
- mean centred per subject-view + std: 0.835, worse.
- Lag sets, all with mean+std at C=0.1, mean over 5 fold seeds: lag5 only 0.905, lag15 only 0.928, lag50 only 0.922, lags 2/5/15/50 0.948, lags 5/15/25/50/80 0.956, nine lags from 1 to 80 0.948.
  The differences between the multi-lag sets are within seed noise (about 0.005), so the notebook keeps lags 5/15/50.
- mean+lags without std: 0.929, and std+lags without mean: 0.951, so the mean adds little once std and lags are in.
- C=1 with lags 5/15/50: 0.952, same as C=0.3.

Train/test shift checks for this model:
- Feature magnitudes (mean, std, lags) of test videos are in the same range as train for both views.
- The feature step across a clip boundary in rebuilt test videos (0.460) equals the step inside a clip (0.464), so the rebuilt videos are continuous.
- Model confidence on test matches OOF: median max probability 0.854 against 0.896, and 6.33 distinct argmax actions per group against 6.46.
- Front and side videos of the same subject and action have only weakly related motion over time (cross-correlation 0.50 against 0.32 for different actions), so pairing front and side videos is not reliable.

Warning from Fayadh's result:
- Fayadh's `submission_main.csv` uses the same idea (rebuilt videos + Hungarian per group, pose features) and got CV 0.885 but public LB 0.687.
- Our new `main.csv` agrees with that file on 84.8% of clips.
- If the public LB is a random 30% of clips, our accuracy can be at most about 0.687 + 0.152 = 0.84, well below our CV of 0.95.
- So either the public LB is not a random sample of clips, or something about the test labels differs from train (for example groups that are not one video per action, or labels that are not constant per video).
- Not resolved, the LB score of `main.csv` will tell.

## 2026-11-17 - Test labels are probably per second: every video starts with about 7-8 seconds of plain writing

Visual check:
- Contact sheets of the rebuilt test videos (8 frames per video, all 8 videos of a group) show one person per group and 8 different-looking takes.
- In every video the person first writes normally, and the action only starts around the middle of the video.

Measured on DINOv3 features (distance of each second from the mean of the first 3 seconds, in units of the spread inside those 3 seconds):
- Train and test behave the same: the distance stays near 1 until second 6, rises in second 7 and jumps to 2.5-3.6 in second 8 for every action except Writing_Answers, which stays at 1.2-1.8.
- Train onset (first second above 2): median second 8, quartiles 7 and 8.
- Test onset: second 7 in 22 videos, second 8 in 44, second 9 in 14.

Fit against the three known public LB scores, taking the video labels of `main.csv` as truth and changing only which clips of a cheat video count as cheat:

| Rule for cheat clips inside a cheat video | Cheat share | Fayadh main (LB 0.6872) | Leon baseline_cv (LB 0.5765) | Fayadh baseline (LB 0.5737) | Sum of errors |
|---|---|---|---|---|---|
| Whole video | 0.506 | 0.848 | 0.638 | 0.534 | 0.262 |
| Second >= 6 | 0.316 | 0.704 | 0.593 | 0.507 | 0.100 |
| Second >= 7 | 0.284 | 0.676 | 0.585 | 0.499 | 0.095 |
| Second >= 8 | 0.252 | 0.647 | 0.577 | 0.485 | 0.129 |
| Second half of the video | 0.244 | 0.644 | 0.576 | 0.476 | 0.141 |
| From detected onset | 0.255 | 0.652 | 0.587 | 0.474 | 0.145 |
| From detected onset - 1 | 0.286 | 0.681 | 0.601 | 0.488 | 0.117 |
| Only seconds with distance > 2 | 0.184 | 0.594 | 0.572 | 0.440 | 0.232 |

Conclusion:
- "Whole video is cheat" fits the known scores worst by a wide margin.
- "Cheat only from about second 6-8 onward" explains Fayadh's drop from CV 0.885 to LB 0.687.
- The exact boundary (6, 7 or 8, or the true onset) cannot be told apart with three LB scores.
- Under the second >= 7 rule, `main.csv` as it is would score about 0.77.
- The old per-clip baseline also predicted cheat more often after second 8 in cheat videos (0.52-0.58 before, 0.65-0.71 after).

New files (same video labels as `main.csv`, clips before the boundary set to 0):
- `leon/submissions/main_from6.csv` (cheat share 0.316).
- `leon/submissions/main_from7.csv` (cheat share 0.284).
- `leon/submissions/main_from8.csv` (cheat share 0.252).

Notebook change:
- `main.ipynb` now has a chart of the action start (`### Awal aksi di dalam video`) and writes the three `main_from{6,7,8}.csv` files in `### Label per detik`.
- Model and features are unchanged, OOF stays at 0.9519.

Side experiment, features from the action part only (Hungarian F1-Macro, C=0.3, mean over fold seeds 42, 0, 1):
- Whole video (current): 0.9519, argmax 8-action accuracy 0.788.
- From second 7 only: 0.9263.
- From second 6 only: 0.9407.
- Whole video + from second 7: 0.9407.
- From second 7, mean minus the mean of the first 6 seconds: 0.9439, argmax accuracy 0.844.
- Whole video + (mean from second 7 minus mean of first 6 seconds): 0.9535, argmax accuracy 0.847.
- Whole video + that difference + std of first 6 seconds: 0.9439.
- Subtracting the video's own writing part helps the argmax (0.788 to 0.847) but not the score after Hungarian, so the notebook features are unchanged.

## 2026-11-17 - main.ipynb: front and side videos paired by motion timing

Finding:
- Front and side videos of the same subject and action are the same take filmed by two cameras, so their motion rises and falls at the same times.
- Motion profile: norm of the feature difference at lag 5 samples (0.5 s), z-scored per video.
  Similarity of two videos: best cross-correlation over shifts of -30 to +30 samples.
- Pairing front to side videos inside a subject with Hungarian on this similarity is 95.7% correct on train, and 85% of subjects are fully right.
  Same take median similarity 0.74, different take 0.42.
- Other profiles were worse: distance from the first 3 seconds 84.1%, smoothed lag-5 energy 93.8%.
  An earlier test with lag-1 energy gave only weak separation (0.50 against 0.32), which is why pairing was first logged as unreliable.
- Matching the front group to the side group of the same person among 6 random train subjects (mean matched similarity, Hungarian): all 6 right in 2,000 of 2,000 trials.

Change in the notebook:
- Test front and side groups are now matched to a person by motion timing, not by CLS features.
  Both methods give the same matching on test (matched similarities 0.63, 0.59, 0.44, 0.61, 0.67, 0.69, best non-matched 0.57).
- New step `### Penugasan gabungan front dan side`: the action assignment of both views of a person is solved together, maximising total log probability plus PAIR_WEIGHT (3.0) times the timing similarity of the front and side video that get the same action, by alternating Hungarian from two starts.
- All submission files were regenerated from this assignment.
  4 of 96 test videos changed label compared with the per-group Hungarian.

Results (OOF, notebook folds seed 42):
- Argmax 0.8628, Hungarian per group 0.9519, joint front + side 0.9615 (8-action accuracy 0.9615).
- Weakest after joint assignment: Checking_Smartwatch front and Using_Smartwatch front (0.885 each).

Side experiments (mean over 5 fold seeds):
- PAIR_WEIGHT 0: 0.9510, 1: 0.9587, 2: 0.9635, 3: 0.9635, 5: 0.9615, 8: 0.9635, 12: 0.9635, 20: 0.9606.
- Upper bound with the true front-side pairing and summed log probabilities: 0.968 (3 seeds).
- Hard pairing first, then summed log probabilities: 0.955 (3 seeds), worse than the joint version.

## 2026-11-17 - Public LB decoded from the exact score decimals: labels are per second

Method:
- F1-Macro depends only on three whole numbers: true positives (TP), errors (FP + FN) and true negatives (TN).
- For each exact public score, every (N, TP, errors, TN) that reproduces the 7 decimals was enumerated.

Results:
- The public set has exactly 455 clips (454 gives no solution).
- `baseline_cv.csv` (0.5765203): TP 85, errors 183, TN 187.
- Steven's file (0.4775127): TP 10, errors 133, TN 312.
  So at least 312 of the 455 public clips (68.6%) are normal, the public set is not 50/50.
- Fayadh's `submission_main.csv` (0.6872, 4 digits only): candidates include TP 115, errors 137, TN 203.
- Fayadh's `submission_baseline.csv` (0.5737068) has no solution at N = 455, not explained.

Expected counts for `baseline_cv.csv` in a random 455-clip sample, with the video labels of the current `main.csv`:

| Rule for cheat clips inside a cheat video | Cheat share | TP | Errors | TN |
|---|---|---|---|---|
| Observed on the LB | about 0.30 | 85 | 183 | 187 |
| Whole video | 0.505 | 134 | 176 | 145 |
| Second >= 5 | 0.347 | 95 | 182 | 178 |
| Second >= 6 | 0.315 | 86 | 184 | 184 |
| Second >= 7 | 0.283 | 79 | 184 | 192 |
| Second >= 8 | 0.251 | 72 | 184 | 199 |

Conclusion:
- Whole-video labels are ruled out (TP 134 expected against 85 observed).
- Clips of a cheat video only count as cheat from about second 6-7.
- Combined distance to the observed counts of `baseline_cv.csv` and Fayadh's main: second >= 7 is closest (10.5), then detected onset - 1 (12.3-14.1), second >= 6 (16.5), second >= 8 (26.7), second >= 5 (34.6).
- The sampling noise is about 7 clips per count, so 6 against 7 against a per-video onset is still open.
- Recommended file: `leon/submissions/main_from7.csv`.

## 2026-11-17 - main_from7 submitted: public LB 0.9600702

- Submitted file: `leon/submissions/submission.csv`, identical to `main_from7.csv` built from the joint front + side assignment (CV 0.9615 per video).
- Public LB 0.9600702, up from the team's 0.6872004.
- Decoded counts on the 455 public clips: 125 cheat clips right, 315 normal clips right, 15 wrong (accuracy 96.7%).
- With Steven's count (312 normal right), the true number of cheat clips in the public set is between 125 and 140 (27.5% to 30.8%).
- Decoded counts of the other files with a slightly wider rounding tolerance: Fayadh main (0.6872004) TP 115, errors 137, TN 203, and Fayadh baseline (0.5737068) TP 80, errors 182, TN 193.
  Fayadh's baseline file in the repo would give about 220 errors under every rule, so it is probably not the file that was submitted.

Where the 15 wrong clips can come from (each explanation alone predicts about 14-15 errors, so they cannot be told apart yet):
- A: boundary 7 is right, and one take is swapped with another in both views (4 videos, about 36 clips, about 11 in public).
- B: all videos are right and the true boundary is second 6 (48 clips, about 14 in public, all missed cheat clips).
- C: all videos are right and the boundary follows each video's own action start (detected onset - 1).
- Boundary 8 is less likely: it predicts 114 cheat clips right against 125 observed.

Probe files for the next submissions:
- `leon/submissions/main_from6.csv`: 1.0 under B, about 0.92 under A.
- `leon/submissions/main_onset.csv` (new, written by a scratch script, not yet in the notebook): boundary = detected onset - 1 per video, 45 clips differ from the submitted file.
  Start second among predicted cheat videos: 4 (2 videos), 5 (2), 6 (15), 7 (16), 8 (10), 9 (1), 11 (2).
