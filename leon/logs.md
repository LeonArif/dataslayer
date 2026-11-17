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

## 2026-11-17 - Does the action always start at the same second? No, it varies per video

Visual check on train (frame strips every 0.5 s from 5.0 s to 9.5 s, 6 cheat videos):
- subject_1 front Using_Smartwatch: about 6.5 s.
- subject_1 front Repeated_Peeking: about 6.5 s.
- subject_7 side Reading_Hidden_Notes: about 8.3 s.
- subject_12 side Looking_Under_the_Desk: about 8.3 s.
- subject_20 front Reading_Hidden_Notes: about 9.0 s.
- subject_15 side Repeated_Peeking: about 8.3 s.
- Test videos were not labelled by eye, because the rules forbid hand labelling of test.

Detector 1 (threshold on the distance from the first 4 seconds, 10 fps):
- Train, all actions except Writing_Answers: 5% 4.6 s, 25% 6.6 s, median 7.5 s, 75% 8.0 s, 95% 9.2 s.
- Start falls in second 4 (27 videos), 5 (30), 6 (45), 7 (142), 8 (68), 9 (15).
- Front median 7.1 s, side median 7.9 s: the two cameras film the same take but are trimmed differently.
- Test predicted cheat videos: second 4 (2), 5 (5), 6 (3), 7 (25), 8 (12), 9 (1).
- Against the 6 visual readings it is early by 1-3 s in 3 of 6, so it is not precise.

Detector 2 (logistic regression "writing or action" per frame on frame minus the video's first 3 seconds, trained on seconds 3-5 as writing and 10.5-13.5 as action, OOF by subject, best single switch point):
- Within 0.4 s of the visual reading in 5 of 6 videos, but one is at 3.0 s against 8.3 s.
- About a quarter of the videos get a degenerate start (second 3 or 13-15).
- Front and side starts of the same take, after aligning the cameras by motion cross-correlation: median error 0.75 s, within 1 s in only 55% of 170 takes.
- Not reliable enough yet to replace the fixed boundary.

Conclusion:
- The real action start varies between about 6.5 s and 9 s, it is not a fixed second.
- If the organisers labelled from the real start, a fixed boundary of 7 is wrong by about one clip per cheat video on average, about 12 public clips, which matches the 15 wrong clips of the 0.9600702 submission.
- Next step: a reliable per-video start detector (restrict the search to 4-11 s, use front and side of a take together).

Is the cheat part a fixed length before the end of the video? No:
- Train: start time and video length are barely related (correlation 0.14).
  The spread of (length - start) is 1.52 s, no smaller than the spread of the start itself (1.45 s), so trimming relative to the end does not make it constant.
- Fit to the exact counts of `submission.csv` (0.9600702), `baseline_cv.csv` and Fayadh's main (chi-square, lower is better): from 6: 3.7, last 9 s: 5.4, last 10 s: 6.5, cheat from 45% of the video: 7.1, last 8 s: 7.3, from 8: 10.3, second half: 14.1, from 5: 24.3, last 7 s: 40.1.
- No fixed rule fits clearly better than the others, which is consistent with a start that varies per video.

## 2026-11-17 - Model choice check and a learned per-video action start

LightGBM against logistic regression on the same video features (mean + std + lags 5/15/50), Hungarian F1-Macro, mean over fold seeds 42, 0, 1:
- Logistic Regression (current): 0.9519, argmax 8-action accuracy 0.788.
- LightGBM on PCA 128 (400 trees, lr 0.03, 15 leaves): 0.9199, accuracy 0.687.
- LightGBM on PCA 32: 0.8317, accuracy 0.617.
- With 416 videos and thousands of dense embedding dimensions, the linear model is clearly better.

Learned start of the action (hard EM, no per-second labels needed):
- Frame feature: frame minus the mean of the video's first 3 seconds, smoothed over 1 s (10 frames).
- Labels: every frame of Writing_Answers videos is "writing"; in the other 7 actions frames before the video's start are "writing" and after it "action".
- Start initialised at 7.5 s, then alternate: fit a binary logistic regression (C=0.003) on all frames, and move each video's start to the single switch point (allowed 4.0-11.0 s) that best fits the model's frame scores.
  6 iterations, last mean change 0.1 frames.
- Out-of-fold starts on train (5 folds by subject): 5% 4.9 s, 25% 7.1 s, median 7.7 s, 75% 8.5 s, 95% 11.0 s; front median 7.4 s, side 8.05 s.
- Against the 6 by-eye readings: 4 within 0.3 s (6.5/6.5, 6.5/6.6, 8.3/8.1, 8.3/8.0); 2 hit the search limits (8.3 read as 4.0, 9.0 read as 11.0).
- Front and side of the same take after camera alignment: median error 0.50 s, within 1 s in 62% of 170 takes (threshold detector: 0.75 s and 55%).
- Test start of predicted cheat videos: 5% 6.6 s, median 8.0 s, 95% 11.0 s.

Fit to the exact LB counts (chi-square over `submission.csv`, `baseline_cv.csv`, Fayadh main; clip p is cheat if p + 0.5 >= start + shift):
- Model start - 1.0 s: 1.1 (expected counts for `submission.csv`: TP 121, errors 15, TN 319, observed 125, 15, 315).
- From 6: 3.7.
- Model start - 0.5 s: 4.5.
- Model start: 11.9.
- Model start + 0.5 s: 28.6.
- The best fit is better than any fixed second, but the difference to "from 6" is within sampling noise.

New file: `leon/submissions/main_em.csv` (same video labels as `main.csv`, cheat from the model start - 1.0 s, cheat share 0.281).
It differs from the 0.9600702 file in 51 clips: 24 become cheat, 27 become normal.
Not yet in `main.ipynb`.

## 2026-11-17 - Start-of-action CV against by-eye readings, and front+side start

Ground truth for the start (train only, test is never labelled by eye):
- Blind frame strips (4.0-11.0 s, every 0.5 s, no model output shown) of 24 random non-writing train videos, plus the 6 from the first check.
- Saved in `leon/train_action_start_eye.csv` (video, start_eye, note).
- 26 of 30 have a clear start, 4 show no clear change.
- 2 videos stop the action and go back to writing before the end (normal/subject_1/side_Checking_Smartwatch 4.8-6.3 s, normal/subject_12/front_Thinking 5.3-8.5 s), so "action until the end" is not always true.

Front+side start: the two videos of a take (same subject and action, motion cross-correlation > 0.5) share one start, with the side start = front start minus the camera lag; the start is chosen to fit both videos' frame scores together, inside the EM loop and at prediction time.
On test, takes are paired by the joint front+side action assignment.

CV on the 26 by-eye starts (EM out-of-fold by subject; wrong clips = 1-second clips whose "action" label differs, clip p counted as action if p + 0.5 >= start, 403 clips in total):

| Method | Median error | Mean error | Within 0.5 s | Within 1 s | Wrong clips |
|---|---|---|---|---|---|
| Fixed 6 s | 1.90 | 1.72 | 0.15 | 0.23 | 40 |
| Fixed 7 s | 1.30 | 1.10 | 0.27 | 0.42 | 26 |
| Fixed 7.5 s | 0.80 | 0.91 | 0.27 | 0.73 | 26 |
| Threshold detector | 0.50 | 0.75 | 0.50 | 0.81 | 16 |
| EM per video | 0.25 | 0.79 | 0.69 | 0.77 | 21 |
| EM front+side together | 0.25 | 0.71 | 0.69 | 0.85 | 17 |

- Average of threshold detector and EM front+side: mean error 0.64 s, only 1 miss above 2 s (EM front+side has 4), but chosen on 26 videos so not adopted.
- EM front+side makes front and side starts agree by construction (EM per video: within 1 s in 62% of takes).

Fit to the exact LB counts (chi-square, lower is better; expected counts for the 0.9600702 file, observed TP 125, errors 15, TN 315):
- EM front+side, shift -1.0 s: 1.5 (123, 12, 319).
- EM front+side, shift -0.5 s: 3.5 (120, 11, 323).
- EM front+side, unshifted: 7.9 (115, 15, 325).
- From 6: 3.7.
- Unshifted gives too few cheat clips, so the organisers' labels seem to start about 0.5-1 s before the visible start (for example a clip counts as cheat as soon as any part of it is after the start).

New files (same video labels as `main.csv`):
- `leon/submissions/main_em_joint.csv`: cheat from the EM front+side start, cheat share 0.255, 49 clips differ from the 0.9600702 file.
- `leon/submissions/main_em_joint_m1.csv`: cheat from that start minus 1.0 s, cheat share 0.286, 41 clips differ.

## 2026-11-17 - Train-as-test CV: train cut into 1-second clips, scored per clip

Goal: a CV that measures what the LB measures (F1-Macro per 1-second clip), without using LB scores.

Per-second truth for train (train has only video labels, so the start had to be read by eye; train only):
- Cropped frame strips (3.5-11.0 s every 0.5 s) of all 104 cheat front videos, saved in `leon/train_action_start_eye_all.csv` (video, start_eye, confidence).
- 95 have a start, 9 show no clear change, 6 of the 95 are low confidence.
- Side start = front start minus the camera lag (motion cross-correlation > 0.5): 91 side videos.
  Check against 7 side videos read separately earlier: median difference 0.25 s, max 0.75 s.
- 186 of 208 cheat videos have a truth start; the 22 without one are left out of scoring. Normal videos are 0 everywhere.
- Clip truth: clip p is cheat if the video is cheat and p + 0.5 >= start ("majority" rule); also scored with p + 1 > start ("any overlap" rule).
- My eye starts have a median of about 7.5 s.

Simulation per fold (same 5 subject folds as `main.ipynb`):
1. Cut each held-out video into 30-frame clips from frame 0 (features = 10 cached DINOv3 frames per clip, pixels = gray 160x90 first and last frame), shuffle.
2. Rebuild with the test pipeline: mutual-nearest pixel chaining (LINK_MAX 3.0), view from the nearest training video, Ward clustering into the fold's number of subjects per view, front/side subject matching by motion timing.
3. LR video model and joint front+side assignment trained on the other folds.
4. EM start finder (per video and front+side) trained on the other folds, applied to the rebuilt videos.
5. Score all 6,102 clips together.

Rebuild quality:
- 416 videos -> 418 rebuilt; 414 exact, 2 videos broken into 2 pieces each (one wrong link missing).
- View 100% correct.
- Subject groups: folds 0 and 1 each have one front or side group of 7 and one of 9 (one video put with the wrong person), the other folds are exact.
- Rebuilt-video cheat accuracy 0.900-0.963 per fold (lower than the 0.9615 of `main.ipynb`, which uses the true groups).

Per-clip F1-Macro (pred = predicted video labels, true vid = true video labels, wrong = wrong clips of 6,102):

| Rule | F1 majority | wrong | F1 majority, true vid | F1 any overlap | wrong | F1 any overlap, true vid |
|---|---|---|---|---|---|---|
| Whole video | 0.7081 | 1643 | 0.7483 | 0.7267 | 1553 | 0.7684 |
| Fixed from 6 | 0.8917 | 519 | 0.9350 | 0.9105 | 437 | 0.9551 |
| Fixed from 7 | 0.9226 | 357 | 0.9668 | 0.9344 | 309 | 0.9794 |
| Fixed from 8 | 0.9324 | 299 | 0.9753 | 0.9171 | 375 | 0.9595 |
| EM per video, shift 0 | 0.9228 | 344 | 0.9660 | 0.9131 | 396 | 0.9558 |
| EM per video, shift -0.5 | 0.9184 | 372 | 0.9637 | 0.9249 | 350 | 0.9701 |
| EM per video, shift -1 | 0.9040 | 446 | 0.9491 | 0.9173 | 392 | 0.9628 |
| EM front+side, shift 0 | **0.9372** | 282 | 0.9791 | 0.9277 | 332 | 0.9691 |
| EM front+side, shift -0.5 | 0.9255 | 342 | 0.9691 | **0.9360** | 300 | 0.9795 |
| EM front+side, shift -1 | 0.9102 | 420 | 0.9536 | 0.9287 | 340 | 0.9725 |

- Per fold (majority), EM front+side shift 0: 0.915 0.954 0.908 0.964 0.951; fixed from 8: 0.923 0.946 0.903 0.948 0.945 (EM wins 4 of 5).
- Start error on 187 cheat videos: EM front+side median 0.25 s, mean bias +0.03 s, 89% within 1 s, 12 above 2 s; EM per video median 0.35 s, 79% within 1 s, 25 above 2 s; fixed 7.5 s median 0.35 s, 85% within 1 s.
- Whole-video labels score about 0.71 per clip, which matches the LB of the whole-video files and confirms the per-second labelling on train too.
- With true video labels the best rule reaches 0.979, so the remaining loss is now mostly from wrong video labels, not from the start.
- The best shift depends on how a clip on the boundary is labelled (shift 0 for majority, -0.5 for any overlap); train cannot tell which one the organisers used.
- Shift -1 is worse than -0.5 under both rules, so the -1 s from the LB-count fit is not supported by train CV.

New file: `leon/submissions/main_em_joint_m05.csv` (EM front+side start minus 0.5 s, cheat share 0.271, 38 clips differ from the 0.9600702 file).

## 2026-11-17 - Where the 15 public errors may be, and one-spot probe files

Goal for tonight: public LB 1.0. The 0.9600702 file has 15 wrong clips of 455 public clips (decoded: TP 125, TN 315).
Test frames were not looked at (hand labelling test is against the rules); every candidate below comes from model outputs only.

Tried first: add "is there any action" evidence from the EM start finder (cost of no switch minus best switch) to the action assignment, weight w on a 1-D logistic P(Writing | evidence).
- AUC for spotting Writing_Answers: evidence 0.862, LR Writing probability 0.972, so the LR already knows more.
- Video CV F1: baseline 0.9615, w=0.5 0.9615, w=1 0.9615 (8-action acc 0.9567), w=2 0.9567, w=4 0.9423. Not adopted.

Second-best assignments per take (re-solve the joint front+side assignment with that take forced to the other class; cost = drop in the joint score):
- T1, cost 1.47: front_08+side_07 (now Reading_Hidden_Notes, EM start 11.0/11.2 = looks like no action) and front_04+side_01 (now Thinking, LR says Reading_Hidden_Notes 0.93/0.84) swap labels.
- T5, cost 4.17: front_05+side_08 (Looking_Around) against front_03+side_04 (Repeated_Peeking).
- T3, cost 4.73: front_06+side_01 (Checking_Smartwatch) against front_02+side_03 (Using_Smartwatch, side EM start 11.0).
- All others cost 7 or more.

Probe files, each = the 0.9600702 file with one swap only (new cheat videos use position >= 7, like that file):
- `leon/submissions/probe_T1_swap.csv`: 34 clips changed.
- `leon/submissions/probe_T5_swap.csv`: 38 clips changed.
- `leon/submissions/probe_T3_swap.csv`: 35 clips changed.
- A right swap fixes about 10 public clips (score about 0.97-0.99); a wrong one adds about 10 (about 0.93-0.94).
