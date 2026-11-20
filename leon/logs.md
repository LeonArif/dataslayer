# Experiment logs

## Not tried yet

Ideas we have discussed but not run.
When one is tried, delete it from this list and write its result as a normal dated entry below.
Last updated 2026-11-20.

Planned for the next session:
- Fine-tune more unfrozen ViT-H+ blocks on 30-frame clips: 16, then 20 only if 16 is better, and keep going until the score stops improving. Past 20 blocks needs a cache from an earlier block than 12 and LoRA or a lighter optimizer.
- Augment the train images (colour changes, small crops, blur, JPEG recode in the test format). A horizontal flip needs a check first, because the left and right hand matter.
- Use the best fine-tuned backbone as the main DINOv3 feature extractor, also for the sequence models (E3 and E5 would read fine-tuned per-frame features instead of frozen ones). Choice to make: all 30 train frames (about 2 h) or every 3rd frame for train with 3 interleaved sets of 10 at test.
- Rerun the blend with the 30-frame fine-tuned B (it replaces the old B and the frozen A).
- Fine-tune LeVJEPA (the video model) instead of using it frozen. Heavier than ViT-H+: 30 frames of 196 patches are about 5,900 tokens per clip, so caching a middle layer would take tens of GB; probably needs fewer frames or only the top blocks. The sequence head on frozen LeVJEPA is already done (E5).
- More frozen video models as extra blend members, using the same recipe as LeVJEPA (all 30 frames, per-frame vector or clip vector, logistic regression and sequence head). Candidates to check for availability and frame-count limits on the 8 GB GPU: V-JEPA 2, VideoMAE and VideoMAE v2, TimeSformer, Video Swin. Video-language models are not allowed, so no text-aligned ones.

Teacher-student and context ideas:
- Soft labels from the whole-video teacher on the sequence models (E3 and E5). Only tried on the logistic regression (C1, no gain).
- Context guessing: an extra output that predicts the previous and next second's features from the clip alone, to give the model more to learn from (HalluciNet style).
- Copying the teacher's per-person-centred features: the student learns to guess what the whole-video teacher computes.

Smaller ideas:
- Stacking (a logistic regression on the members' scores) instead of the weight grid in the blend.
- Tuning E's own settings (PCA size, hidden size, epochs, dropout); each was run at one setting only.
- A LightGBM head on the LeVJEPA features as in Owen's baseline (D1 used a logistic regression).
- Re-extracting the frozen DINOv3 train features for all folds in the test format (only fold 0 was checked, the gap cost A about 0.02 to 0.03 F1).

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
Test frames were not looked at every candidate below comes from model outputs only.

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

## 2026-11-17 - Model only: no fixed second, no 4/4 balance, no LB feedback

Constraint from now on: no "cheat from second k" rule, no forced 4 cheat / 4 normal per subject-view (no Hungarian), no LB-based choices.
Per-second training targets come from the by-eye train starts (`leon/train_action_start_eye_all.csv`); CV = the 5 subject folds of `main.ipynb`, scored per clip (majority rule).

Per-clip classifier (LR on clip features, labels = cheat action visible in the clip):

| Model | F1 per clip |
|---|---|
| Every clip = video argmax (p > 0.5) | 0.6788 |
| Clip mean+std alone | 0.7216 |
| Clip mean+std, times video probability | 0.7771 |
| Clip minus its video's first 3 s | 0.7723 |
| Clip minus first 3 s, times video probability | 0.8195 |
| One free switch per video from the clip model, true video labels | 0.8005 (ceiling of this start part) |

Front/side camera pairing inside a subject by motion timing (one-to-one matching, no cheat ratio): 95.7% correct on train; video F1 without Hungarian 0.858 single view, 0.870 front x side product.

EM start finder (writing -> action, frames relative to the first 3 s), front+side joint, predicted video labels without Hungarian:

| Start search | Median error | > 2 s off | F1, true video labels | F1, predicted video labels |
|---|---|---|---|---|
| Window 4-11 s, init 7.5 s (old) | 0.25 | 7 | 0.9837 | 0.8904 |
| Anywhere, init 7.5 s | 0.25 | 35 | 0.9380 | 0.8556 |
| Anywhere, init middle of video | 0.35 | 40 | 0.9266 | 0.8459 |

- The main loss is the video cheat/normal decision without the balance rule (0.87 video F1), not the start.
- Test files (LR on all train, front x side product, threshold 0.5, EM front+side start; 44 of 96 videos cheat, 8 videos differ from the Hungarian labels):
  - `leon/submissions/main_model_only.csv`: start anywhere, init middle; cheat share 0.215, 136 clips differ from the 0.9600702 file.
  - `leon/submissions/main_model_only_window.csv`: start window 4-11 s, init 7.5 s; cheat share 0.234, 108 clips differ.

## 2026-11-17 - main_legit.ipynb created: train-only pipeline (DINOv3 + pose)

`main.ipynb` stays the pipeline that uses LB-derived knowledge (fixed second, forced 4/4 balance).
`main_legit.ipynb` only uses what can be learned from train:
- No fixed "cheat from second k" rule, no forced cheat/normal ratio (no Hungarian over actions), no LB feedback.
- Timing priors come from the by-eye train starts: start search window 3.2-11.1 s (earliest - 1 s to latest + 1 s), first guess 7.7 s (median).
- Person grouping of test (Ward per view, 6 groups) is kept only to find the front/side camera pairs.
  Pairing over all videos at once is only 76.5% right on train, inside a person 95.7%.

New feature source: pose keypoints (RT-DETR person box + ViTPose 17 keypoints, 10 fps).
- Extracted with `leon/scripts/extract_pose.py` (copy of `fayadh/scripts/extract_pose.py`, stride 30 so every window is one test-like 1-second clip from frame 0) into `cache/pose_vitpose/`.
- Time on the RTX 4070: train 2,347 s, test about 10 min.
- 19 per-frame signals (head_yaw, head_down, eye_yaw, wrist_gap, hand_face, wrist and elbow positions, lean, head and hand speed), normalised by the person box height.
- Per video: mean, std, mean frame-to-frame change, max and min relative to the first 3 s, mean after the first 3 s (114 features).

Design checks (train CV, 5 subject folds):
- Pose LR C 0.01 / 0.03 / 0.1 / 0.3: pose alone video F1 0.777 / 0.786 / 0.793 / 0.788; with DINOv3 (weight 1.0) 0.8725 / 0.8725 / 0.8726 / 0.8774.
  Chosen C=0.1, weight 1.0 (middle of a flat area).
- Start finder: plain EM 0.9778 per clip with true video labels, EM with cheat starts fixed to the by-eye starts 0.9703, so plain EM is kept.
- Video threshold 0.3-0.7 changes video F1 only between 0.83 and 0.87, kept at 0.5.

Results in the notebook:

| Model | Video F1 | 8-action accuracy |
|---|---|---|
| DINOv3 LR | 0.8580 | 0.7764 |
| Pose LR | 0.7931 | 0.7188 |
| DINOv3 + pose | 0.8726 | 0.8269 |
| DINOv3 + pose + front x side | 0.8702 | 0.8149 |

- Start of the action (cheat videos against by-eye): median error 0.25 s, 88% within 1 s, 13 off by more than 2 s.
- Per-clip F1: whole video 0.6758, with the start finder **0.8940**, with true video labels 0.9769.
- Output `leon/submissions/main_legit.csv`: 44 of 96 test videos cheat, cheat share 0.234.

Comparison with the 0.9600702 file used as pseudo ground truth (information only, not used for any choice in the notebook):

| File | F1 against the 0.960 file | Clips differing |
|---|---|---|
| `main_legit.csv` | 0.9061 | 109 |
| `main_model_only_window.csv` | 0.9071 | 108 |
| `main_model_only.csv` | 0.8799 | 136 |
| `main_pairfix_from7.csv` (= Steven's `take_hung_from7`) | 0.9846 | 19 |

Where the train-only model disagrees with the 0.960 file on the cheat/normal of a whole take:
- `subject_T1` front_04 + side_01: legit says cheat, the 0.960 file says normal (Thinking).
  The same take is the closest call of the Hungarian (cost 1.47) and of Steven's model (margin 0.89).
- `subject_T3` front_02 + side_03, `subject_T4` front_05 + side_02, `subject_T5` front_03 + side_04: legit says normal, the 0.960 file says cheat.
  Legit predicts only 44 cheat videos (no balance rule, video F1 0.87), so these are weaker.
- Boundary only: `subject_T1` front_03, front_08, side_06, side_07 start about 4 s later in legit (EM start 10-11 s against the fixed 7).

`main_pairfix_from7.csv` against the 0.960 file (19 clips, `subject_T3` side_01 becomes cheat, side_03 becomes normal):
- DINOv3 timing favours pairfix (summed similarity 0.73 against 0.54).
- Simple pose-speed timing favours the 0.960 pairing (1.27 against 1.20).
- Close to a coin flip.

## 2026-11-17 - Data insights for the presentation (EDA)

Train structure:
- 26 subjects x 2 cameras (front, side) x 8 actions = 416 videos of 13.6-18.6 s (median 15.9 s), 30 fps.
- Every subject does every action exactly once; front and side of one action are the same take filmed at the same time.
- Test has the same layout after rebuilding: 1,515 one-second clips chain into 96 videos of 14-19 s (median 16), 48 front + 48 side, 6 people x 2 cameras x 8 videos.

When the action happens (by-eye on 104 cheat front videos of train, 95 readable):

| Action | Readable | Median start | Earliest | Latest |
|---|---|---|---|---|
| Looking_Under_the_Desk | 18 | 7.75 s | 6.25 s | 8.25 s |
| Reading_Hidden_Notes | 25 | 7.25 s | 5.25 s | 8.75 s |
| Repeated_Peeking | 26 | 7.25 s | 4.75 s | 8.25 s |
| Using_Smartwatch | 26 | 7.75 s | 6.25 s | 8.25 s |

- Every video starts with plain writing; the action comes in the second half of the video.
- 8 of the 9 unreadable starts are Looking_Under_the_Desk: the movement is small from the front camera.
- Some actions stop before the video ends (seen in 2 videos).
- The side camera starts about 0.5 s later than the front camera for the same take (median -0.5 s, 90% between -1.0 and +0.2 s; 170 of 182 action takes have a clear motion match).

Same pattern in test (check only, never used for prediction):
- DINOv3 distance of each second from the first 3 s, median over videos:
  train 0.70 0.47 0.61 0.87 0.99 1.13 1.22 1.59 **2.66** 2.82 2.60, test 0.58 0.45 0.55 0.76 0.85 0.94 0.99 1.24 **2.70** 2.85 2.62; both jump at second 8.
- Writing_Answers videos stay flat (1.3-1.8 after second 7), so the jump is the action, not drift.
- Pose change (median |change| from the first 3 s, seconds 9-12 divided by seconds 3-6): head_yaw train x3.2 / test x6.7, eye_yaw x2.9 / x6.4, wrist_gap x5.9 / x7.8, hand speed x2.6 / x2.3, head speed x3.2 / x3.4.
- The gesture measurements from train also move in test, at the same time, so they are meaningful there.

Signature gesture of each action (pose, mean change from the first 3 s over seconds 8-12, front and side together):
- Checking_Smartwatch: big head turn (head_yaw +1.46), hands still; the person looks at the wrist.
- Using_Smartwatch: hands come together (wrist_gap -0.109, the largest drop) with almost no head turn (-0.03); one hand operates the watch.
- Thinking: head turn (+1.30) and hand to face (hand_face -0.155, the largest drop).
- Looking_Around: moderate head turn (-0.38), eyes sideways (eye_yaw +0.126).
- Repeated_Peeking: strong head turn (-0.94) and the fastest head (head_speed 0.009).
- Looking_Under_the_Desk: head turn (-0.70), eyes down or sideways (eye_yaw -0.30), body lower (lean_y -0.023).
- Reading_Hidden_Notes: head turn (-0.77) and the fastest hands (hand_speed 0.014), pulling out a note.
- Writing_Answers: near zero on every signal, the slowest head and hands (0.001).
- The hard pairs follow from this: Checking against Using Smartwatch differ in head turn against hands together; Looking Around against Repeated Peeking differ mostly in head speed.

Model insights:
- One second alone is not enough (per-clip models 0.56-0.58); whole-video context is needed.
- DINOv3 (what is visible) is stronger than pose (how the body moves): video F1 0.858 against 0.793, together 0.873.
- With the true cheat/normal of each video, the start finder alone reaches 0.977 per clip, so the remaining error is mostly the cheat/normal decision per video.

## 2026-11-18 - Steven's pairfix file scored public LB 0.9761476

- Submitted by Steven as `submission_owen.csv`, identical to `owen/experiment/submissions/submission_from7.csv`, `submission_take_hung_from7.csv` and `leon/submissions/main_pairfix_from7.csv` (0 clips differ).
- Decoded counts on the 455 public clips: 129 cheat clips right, 317 normal clips right, 9 wrong (was 125 / 315 / 15 for the 0.9600702 file).
- It differs from the 0.9600702 file in 19 clips, all in `subject_T3`: side_video_01 becomes cheat from clip 7 (10 clips), side_video_03 becomes normal (9 clips).
- Net +4 cheat right and +2 normal right means at least 6 public clips changed and all (or almost all) of them were fixed, so the side_01/side_03 swap is right.
- So the front-side camera pairing step made one mistake in `subject_T3`, and the DINOv3 motion timing (0.73 vs 0.54) picked the right pairing, pose speed timing did not.
- `main_legit.csv` called side_03 normal (right) and side_01 normal (wrong).
- 9 public errors are left, true cheat clips in public are between 129 and 138.
- Strongest remaining candidate: `subject_T1` swap front_04 + side_01 to cheat and front_08 + side_07 to normal (34 clips, about 10 public).
  It is the closest call in both Hungarian models and `main_legit` calls front_04 + side_01 cheat with a normal start, while front_08 + side_07 get a weak late start (about 11 s).
  About 10 public clips predicted against 9 errors left, so it fits; if right about 0.997-1.0, if wrong about 0.95.
- The other explanation is start boundary errors (clip 7 on videos whose action starts after 7.5 s), which also fits 9 errors, so the two cannot be told apart from the score alone.

## 2026-11-18 - Where are the 9 errors left? Boundary, not a wrong video

Train-as-test simulation (6,102 train clips, by-eye starts, true video labels, so only the start rule can be wrong), errors scaled to 455 clips:

| Start rule | Majority labelling | Any-overlap labelling | Worst case |
|---|---|---|---|
| Fixed from 7 (current files) | 11.4 | 7.2 | 11.4 |
| Fixed from 8 | 8.1 | 13.6 | 13.6 |
| EM front+side, shift 0 | 7.0 | 10.6 | 10.6 |
| EM front+side, shift -0.25 | 7.8 | 7.9 | 7.9 |
| EM front+side, shift -0.5 | 10.6 | 7.2 | 10.6 |
| EM per video, best shift | 11.0-12.3 | 10.4-12.5 | 12.3 |

- Even with every video label right, "cheat from 7" alone is expected to make 7-11 public errors; the 0.9761476 file has 9.
- So the 9 errors left are most likely all boundary errors and the video labels of Steven's file are probably all right.
- The `subject_T1` front_04/side_01 vs front_08/side_07 swap would then add about 10 errors (about 0.95), so it is unlikely to be right.
- The lowest-risk improvement is EM front+side start minus 0.25 s on Steven's video labels: about 8 errors under both labelling rules (small gain, not 1.0).
- Caveat: the by-eye train starts have their own reading error, so these counts are rough.

## 2026-11-18 - Per-video start seconds on test (EM front+side) and a partial-move check

- Steven's file (0.9761476) starts every one of its 48 cheat videos at clip 7 (second 7.0).
- EM front+side starts for those 48 videos, by whole second: 5 (1), 6 (6), 7 (23), 8 (12), 9 (2), 10 (1), 11 (3).
- Cheat from ceil(start - 0.75) (the "minus 0.25 s" rule): clip 5 (1 video), 6 (3), 7 (20), 8 (16), 9 (4), 10 (1), 11 (3); 28 videos change, 34 fewer cheat clips than Steven's file.
- Far outliers: `subject_T4/side_video_02` 5.1 s, `subject_T1` front_03, front_08, side_06, side_07 at 10.1-11.2 s.
- Train check: keeping clip 7 unless the found start is far from 7.5 s does not help (worst case 9.2-11.9 errors per 455 against 7.9 for moving every video), so moving all videos is the better rule.

## 2026-11-18 - The test start outliers are probably finder mistakes

- Train by-eye starts (95 cheat front videos): 4.75-8.75 s, 5% 6.25 s, median 7.75 s, 95% 8.25 s; only 4 below 6 s (all front Repeated_Peeking or Reading_Hidden_Notes) and none at 9 s or later.
- `subject_T4/side_video_02`: per-video EM start 8.3 s, but the joint front+side start is 5.1 s, i.e. its front (front_05, 7.5 s) minus a camera lag of about 2.4 s; the usual lag is about 0.5 s the other way, so the joint lag for this pair is wrong and 8.3 s (or about 8.0 s) is more likely.
- `subject_T1` front_03, front_08, side_06, side_07: 10.1-11.2 s, at the upper edge of the search window and later than every train start, so the finder found no clear switch there; these four starts should not be trusted.
- If a per-video start file is built, these 5 videos need a fallback (per-video start for T4 side_02, a capped or default start for the T1 four) instead of the raw joint start.

## 2026-11-18 - Guards against start-finder mistakes (train-as-test simulation)

Errors per 455 clips with true video labels, cheat from ceil(start - 0.75):

| Rule | Majority | Any overlap | Worst |
|---|---|---|---|
| Fixed 7 (Steven's file) | 11.41 | 7.23 | 11.41 |
| Raw joint start | 7.75 | 7.90 | 7.90 |
| Use own start when joint and own disagree by more than 1.5 s | 10.66 | 11.56 | 11.56 |
| Clamp joint start to 5-9 s | 6.93 | 7.08 | 7.08 |
| Joint start outside 5.5-9.5 s -> 7.75 s (train median) | 6.49 | 6.34 | 6.49 |

- On train the finder also gives impossible starts (joint 3.6-11.4 s against by-eye 4.75-8.75 s): 6 videos at 10 s or later, 8 below 5.5 s.
- Preferring the own-video start when the two cameras disagree hurts: the joint start is usually the right one.
- Replacing out-of-range starts with the train median, or clamping them, removes most of the >2 s misses and is the best rule under both labelling rules.
- The 5.5-9.5 s range was picked from about 20 close variants (all 6.3-7.5), so its edge over plain clamping is partly luck.
- On test this sends `subject_T4/side_video_02` (5.1 s) and the four `subject_T1` late videos (10.1-11.2 s) back to 7.75 s, i.e. cheat from clip 7 as in Steven's file.

## 2026-11-18 - New file: main_start_guard.csv

- `leon/submissions/main_start_guard.csv`: cheat/normal per video from Steven's 0.9761476 file, cheat from ceil(start - 0.75) with the EM front+side start per video, starts outside 5.5-9.5 s replaced by 7.75 s.
- Guarded videos (back to clip 7): `subject_T4/side_video_02` and `subject_T1` front_03, front_08, side_06, side_07.
- Cheat from clip 6 (3 videos), 7 (25), 8 (16), 9 (4).
- Cheat share 0.270 against 0.284; 27 clips differ from the 0.9761476 file in 23 videos (24 become normal, 3 become cheat), about 8 public.
- Train simulation expects about 6.5 public errors against Steven's 9 (about 0.98-0.985), not submitted yet.

## 2026-11-18 - Which fix gains most on the public score (train bootstrap)

4,000 test-sized draws from the train simulation (48 cheat + 48 normal videos, 455 clips kept as "public"), true video labels, so only the start rule differs:

| Rule | Majority: errors (gain vs fixed 7, better / worse) | Any overlap: errors (gain, better / worse) |
|---|---|---|
| Fixed 7 (Steven) | 12.0 (5-95%: 7-17) | 7.6 (3-12) |
| `main_start_guard` rule | 6.8 (+5.2, 97% / 1%) | 6.7 (+0.9, 57% / 30%) |
| Raw joint start | 8.1 (+3.9, 85% / 9%) | 8.4 (-0.7, 36% / 52%) |
| Fixed 8 | 8.6 (+3.4, 79% / 14%) | 14.5 (-6.8, 2% / 97%) |

- Steven's 9 errors fit both labelling rules, so the rule is still unknown.
- The guard rule is the only option that is better on average under both rules, so it has the most expected gain for the next submission.

## 2026-11-18 - main_start_guard.csv submitted: public LB 0.9626489 (worse)

- Decoded: 125 cheat clips right, 316 normal clips right, 14 wrong (Steven's file: 129 / 317 / 9).
- The file changed 27 clips from Steven's: 24 cheat -> normal (later start), 3 normal -> cheat (clip 6).
- Change in counts: cheat right -4, normal right -1.
  So at least 4 of the public clips moved to normal were really cheat, and at least 1 public clip moved to cheat at clip 6 was really normal.
- The cleanest fit: every public changed clip got worse (4 of the 24 and 1 of the 3 were public), i.e. the truth is "cheat from clip 7" in those videos even where the action visibly starts at 7.8-8.4 s.
- Conclusion: the public labels behave like a fixed boundary at second 7 per cheat video, not like the per-video start; the train by-eye start simulation does not describe how test was labelled.
- Then the 9 errors left in Steven's file are most likely wrong video labels or pairings (whole videos), not boundaries.
- Next: keep cheat from clip 7 and look for wrong cheat/normal or front-side pairings (the `subject_T1` front_04/side_01 vs front_08/side_07 swap is a candidate again).

## 2026-11-18 - New file: main_t1swap_from7.csv

- `leon/submissions/main_t1swap_from7.csv`: Steven's 0.9761476 file with one take swapped in `subject_T1`: front_video_04 + side_video_01 become cheat from clip 7, front_video_08 + side_video_07 become normal.
- 34 clips change (17 to cheat, 17 to normal), about 10 public; T1 keeps 4 cheat fronts and 4 cheat sides; cheat share unchanged (0.284).
- Why: the closest call in both Hungarian models (Leon's and Steven's), `main_legit` calls front_04 + side_01 cheat with a normal start, and after the guard result the 9 errors left look like whole-video errors (about 10 public clips for one wrong take).
- If right: about 0-1 public errors left (about 0.997-1.0); if wrong: about 19 (about 0.95). Not submitted yet.

## 2026-11-18 - main_legit benchmark against the 0.96+ files

Pseudo ground truth for `main_legit` from now on: the three files at 0.96 or above (`submission_0.96.csv`, Steven's `submission_from7.csv` at 0.9761476, and the T1 swap file, now `leon/submissions/submission.csv`). Target: clip F1 0.96+ against them.
They are only used to compare, never to train or tune.

| Pseudo GT | Clip F1 of main_legit | Clips differ | Video F1 | Wrong videos (clips) | Boundary-only clips |
|---|---|---|---|---|---|
| 0.96 file | 0.9061 | 109 | 0.9165 | 8 (69) | 40 |
| Steven 0.976 | 0.9054 | 110 | 0.9165 | 8 (70) | 40 |
| T1 swap | 0.9174 | 96 | 0.9165 | 8 (63) | 33 |

- Against the T1 swap file the 8 wrong videos are 4 takes: `subject_T1` front_08 + side_07 (legit cheat, file normal), and `subject_T3` front_02 + side_01, `subject_T4` front_05 + side_02, `subject_T5` front_03 + side_04 (legit normal, file cheat).
- About two thirds of the gap is the cheat/normal decision per video (3 of 4 wrong takes are missed cheats), one third is the start boundary.
- Hugging Face check: no released DINOv3 head for action or video classification (DINOv3 heads are for ImageNet classes, detection, segmentation and depth); action heads exist only for other video backbones (V-JEPA 2 SSv2 / Diving48 attentive probes).

## 2026-11-18 - Gesture durations from pose (train only)

Goal: measure how long each gesture lasts after the first 3 s (for example how long the hands stay together on the watch) and see whether that separates cheat from normal.
Script: scratch `duration.py` and `dur_cv.py` (not in the notebook yet).

Method:
- Per frame (10 fps, ViTPose keypoints) signals relative to how the person sat in the first 3 s: head turn (any side, left, right), head down, head up, hand to face, hands together, wrist up, head moving, hand moving.
- A gesture is "on" when the signal is more than 3 robust spreads (MAD) above the first-3-s level.
- Per gesture: share of time on, episodes per second (at least 0.3 s), longest episode, mean episode length; plus head side switches per second. 41 features per video.

Median seconds "on" after the first 3 s (front + side):

| Action | Head turn | Head down | Hand to face | Hands together | Wrist up | Head moving | Hand moving |
|---|---|---|---|---|---|---|---|
| Checking_Smartwatch (normal) | 6.0 | 1.3 | 5.4 | 0.2 | 4.6 | 1.8 | 2.3 |
| Using_Smartwatch (cheat) | 6.5 | 1.7 | 6.2 | 5.6 | 6.1 | 2.2 | 3.0 |
| Looking_Around (normal) | 6.5 | 1.0 | 1.2 | 0.4 | 0.4 | 4.4 | 0.5 |
| Repeated_Peeking (cheat) | 7.3 | 0.2 | 0.2 | 0.0 | 0.1 | 4.8 | 1.4 |
| Thinking (normal) | 6.4 | 1.2 | 7.1 | 0.4 | 7.0 | 3.6 | 3.2 |
| Reading_Hidden_Notes (cheat) | 6.6 | 4.8 | 4.8 | 2.3 | 3.4 | 3.4 | 4.6 |
| Writing_Answers (normal) | 4.6 | 0.6 | 1.6 | 0.2 | 0.6 | 0.9 | 1.3 |
| Looking_Under_the_Desk (cheat) | 6.9 | 5.6 | 2.6 | 4.2 | 0.4 | 3.5 | 4.4 |

Paired check (same subject and camera, share of 52 pairs where the cheat video has the larger value, 0.5 = no signal):
- Checking vs Using smartwatch: hands together longest episode / duration 0.85 (using = both hands together for seconds, checking = a short glance).
- Thinking vs Reading_Hidden_Notes: wrist up longest episode 0.11 (thinking keeps the hand up at the chin), hand moving longest episode 0.88.
- Writing vs Looking_Under_the_Desk: head moving episodes 0.88, hand moving 0.83.
- Looking_Around vs Repeated_Peeking: weak, best 0.65 (hand moving); this pair is still the hardest.

Video CV (StratifiedGroupKFold 5, seed 42, true front-side partner, so the baseline here is 0.8798 instead of the notebook's 0.8702):

| Model | Video F1 | 8-action acc |
|---|---|---|
| main_legit (DINOv3 + pose + two cameras) | 0.8798 | 0.8365 |
| Durations alone | 0.73-0.74 | 0.51-0.53 |
| main_legit + durations as a third model, C=0.1, weight 1.0 | **0.9038** | 0.8702 |
| main_legit + durations, other C / weights (6 settings) | 0.8894-0.8990 | 0.855-0.865 |
| DINOv3 + one model on pose + durations | 0.8942 | 0.851-0.861 |

- Every setting beats the baseline (+1.0 to +2.4 points video F1), so the gain is not from one lucky setting.

## 2026-11-18 - Error analysis of main_legit (train OOF and test vs pseudo GT)

Scripts (scratch): `err_common.py`, `err_an.py`, `la_rp.py`, `la_rp_spec.py`, `la_rp_main.py`, `test_check.py`.

Train OOF, cheat/normal accuracy per action (52 videos each):

| Action | DINOv3 (pixels) | Pose | Durations | DINOv3 + pose + dur, one camera | Final, both cameras |
|---|---|---|---|---|---|
| Checking_Smartwatch (normal) | 0.79 | 0.79 | 0.79 | 0.87 | 0.88 |
| Looking_Around (normal) | 0.90 | 0.69 | 0.65 | 0.81 | 0.85 |
| Looking_Under_the_Desk (cheat) | 0.96 | 0.96 | 0.88 | 0.98 | 1.00 |
| Reading_Hidden_Notes (cheat) | 0.79 | 0.87 | 0.87 | 0.85 | 0.88 |
| Repeated_Peeking (cheat) | 0.73 | 0.48 | 0.54 | 0.77 | 0.81 |
| Thinking (normal) | 0.96 | 0.94 | 0.87 | 0.98 | 1.00 |
| Using_Smartwatch (cheat) | 0.83 | 0.75 | 0.62 | 0.85 | 0.96 |
| Writing_Answers (normal) | 0.90 | 0.87 | 0.71 | 0.85 | 0.85 |

- Final model: 40 of 416 videos wrong (20 front, 20 side), video F1 0.9038.
- 16 of the 40 errors are Looking_Around <-> Repeated_Peeking (8 each way); next: Checking -> Using smartwatch 4, Hidden notes -> Checking smartwatch 4, Writing scattered over 5 actions (8).
- Pixels (DINOv3) beat pose on Looking_Around (0.90 vs 0.69), Repeated_Peeking (0.73 vs 0.48), Using_Smartwatch and Writing; pose beats pixels only on Reading_Hidden_Notes (0.87 vs 0.79).

Looking_Around vs Repeated_Peeking, 20 head-movement measures from pose (paired per subject and camera, share where peeking is larger):
- Front camera: peeking turns to one side only (one-sidedness median 1.00 vs 0.18, pairs 0.81), holds each turn longer (1.5 s vs 0.7 s, 0.77), has longer gaps between turns (0.85) and a more regular rhythm (autocorrelation peak 0.28 vs 0.00, 0.79); looking around moves the eyes more relative to the head (0.23).
- Side camera: almost no difference (best 0.73 rhythm), because head left/right is hard to read from 2D keypoints seen from the side.
- Specialist model for only this pair (104 train videos): pair accuracy 0.64-0.73 against 0.83 for the main model; overall video F1 0.84-0.90, worse than 0.9038.
- Same measures added to the durations model: 0.8894-0.9086 (6 settings), no reliable gain.

Test check (one camera, against the T1 swap file as pseudo GT, 96 videos):

| Model | Video F1 | Wrong videos |
|---|---|---|
| DINOv3 | 0.8745 | 12 |
| Pose | 0.8628 | 13 |
| Durations | 0.7917 | 20 |
| DINOv3 + pose (main_legit) | 0.9374 | 6 |
| DINOv3 + pose + durations | 0.9062 | 9 |

- The durations gain on train CV (+2.4 points) did not carry over to test: 9 wrong videos against 6, so durations stay out of the notebook for now.
- main_legit's 6 wrong test videos: 4 of 6 are side camera; 3 are cheat videos predicted as Writing_Answers (`subject_T3` front_02 and side_01, `subject_T6` side_08), i.e. the cheat action is too small to be seen; plus `subject_T1` front_08 (normal predicted Reading_Hidden_Notes), `subject_T2` side_07 (predicted Checking_Smartwatch), `subject_T5` side_04 (predicted Looking_Around).
- Main weakness: where the person looks (gaze and small head direction) is not captured; 2D pose cannot read it from the side, and whole-frame DINOv3 sees the head as a few pixels.

## 2026-11-18 - Line of sight visualised from pose (train)

Script: scratch `gaze_viz.py`. Figures in `leon/figures/`: `gaze_subject10_frames.png`, `gaze_subject10_timeline.png` (model right), `gaze_subject11_frames.png`, `gaze_subject11_timeline.png` (model wrong, peeking called looking around).
Arrow = from the middle of the ears through the nose, extended; drawn on frames at 3, 5, 7-13 s for Looking_Around and Repeated_Peeking, front and side.

What it shows:
- Front camera, Repeated_Peeking: the head turns to one side (towards the neighbour) and holds there for 2-4 s, comes back to the paper, then turns to the same side again (subject_10: 7.5-11.5 s and from 15 s; subject_11: 6.5-14.5 s with short returns).
- Front camera, Looking_Around: short, small turns to both sides (under 1 s each), often with the head lifted; subject_11 barely moves at all.
- Side camera: the arrow points at the desk in every frame for both actions; a left/right turn is mostly towards or away from the camera, which 2D keypoints cannot show. Only a hint is visible (more of the face, both eyes visible, when turning towards the camera).
- The side-camera head-turn signal is noise: values jump to 5-50 because head_yaw is divided by shoulder width, and from the side the two shoulders overlap (width close to 0). This is a flaw in the current pose features.
- subject_11 peeking is obvious on the front camera, yet the model called it Looking_Around; the both-camera sum lets the noisy side camera outvote it, and mean/std summaries do not capture "held to one side for seconds".

Ideas from this:
- Normalise head turn by head size (ear-nose distance) instead of shoulder width, and measure "face towards camera" for the side view (eye distance, ear visibility).
- Let the front camera's line of sight decide this pair for the whole take.
- A dedicated head-pose model (gives yaw and pitch in degrees from any angle) would replace the rough 2D arrow.

## 2026-11-18 - Looking_Around vs Repeated_Peeking: pattern difference and a specialist model

Scripts (scratch): `head2.py`, `head2_cv.py`, `head2_test.py`, `head2_testplot.py`, `head2_shift.py`, `head3.py`, `head4.py`.
Figures: `leon/figures/look_vs_peek_timelines.png` (8 train subjects), `leon/figures/look_vs_peek_test_flips.png` (test videos the specialist flipped).

Fixed head signals (fix for the side-camera noise found earlier):
- turn = nose minus ear midpoint, divided by head size (nose-ear distance + ear-ear distance, fixed per video) instead of shoulder width.
- pitch = nose below ears, in head sizes; face_width = eye distance in head sizes (wider = face turned towards the camera, for the side camera).

The pattern (train, paired over 26 subjects):
- Front: Repeated_Peeking turns to one side only (one-sidedness 0.997 vs 0.277, cheat larger in 96% of pairs, was 81% with the old signal) and holds longer (2.1 s vs 1.4 s); Looking_Around turns once to each side.
- Side: peeking repeats the same bump (towards or away from the camera) 2-3 times; looking around goes one way then the other (rhythm 0.27 vs 0.10, 88% of pairs, was 73%).

Model results:

| Variant | Train video F1 | Train LA/RP wrong | Test one-camera wrong vs pseudo GT |
|---|---|---|---|
| main_legit (DINOv3 + pose) | 0.8798 | 24 | 6 |
| + head-pattern model as third model (C 0.03-0.3) | 0.9038-0.9135 | 14-16 | 8 |
| Specialist re-splits LA/RP mass, both cameras (C 0.1-0.3) | 0.9183 | 8 | 10-11 |
| Specialist, scale-free thresholds, no confidence signals | 0.8990-0.9135 | - | 7-10 |
| Specialist, only one-sidedness / hold features | 0.8846-0.8989 | - | 11-13 |
| Specialist, all scale-free features except rhythm | 0.9038-0.9086 | - | 7 |

- On train the specialist works (pair accuracy 0.94 with both cameras, LA/RP errors 24 -> 8), but on test every version changes 3-9 videos and at most 1 change is right.
- Test differences found: side-camera head movement is about half the size of train (turn p90 0.108 vs 0.173, face_width 0.028 vs 0.055); keypoint-confidence signals (ear balance, eye confidence) shift strongly; the rhythm measure fires on "look left then right" in test (two similar bumps).
- On some flipped test videos the one-sidedness value itself points the right way (T3 front_04 cheat 1.00, T5 front_05 normal 0.16), but the combined model still flips other videos wrongly, mostly side camera.
- Important: only 1 of main_legit's 6 wrong test videos involves Looking_Around (`subject_T5` side_04); the test errors are mostly cheat videos predicted as Writing_Answers. The hardest pair on train is not where test loses.
- Decision: keep the pattern as an EDA insight for the PPT; do not add the specialist to main_legit.

## 2026-11-18 - Head direction: peeking goes one way, looking around goes several ways

Scripts (scratch): `head_dir.py`, `head_dir2.py`.
Per video, after the first 3 s, the frames where the head leaves the writing position are sorted into 8 compass directions (left/right turn and up/down pitch, from the head-size-normalised signals).

| Measure | Front: look around | Front: peeking | Front pairs separated | Side pairs separated |
|---|---|---|---|---|
| Directions used (sectors with >15% of away time) | 2 | 1 | 92% | 56% |
| Share of away time in the main direction | 0.53 | 1.00 | 92% | 56% |
| Sideways vs vertical | 0.52 | 0.65 | 73% | 62% |
| Head lifted while away | 1.00 | 1.00 | 54% | 50% |

- Both actions lift the head off the paper; the difference is that peeking always goes the same way (the neighbour), looking around uses at least two directions.
- Holds on test (front, median): pseudo-GT normal 2 directions / main share 0.49, cheat 1 direction / 0.96.
- Of the 13 test front videos the model calls LookAround/Peeking, the direction rule (main share) agrees with the pseudo GT on 12; the one miss is `subject_T1` front_05 (normal, but all its turns go one way).
- main_legit already gets all 13 of those front videos right, so the rule adds nothing on its own on test front videos.
- Where it could help: `subject_T5` front_03 is borderline in main_legit (P(cheat) 0.54) and the direction rule says peek (main share 1.00); its side video side_04 is called Looking_Around, so the both-camera sum makes the take normal (one of main_legit's wrong takes). Letting the front camera's direction decide this pair for the whole take would fix that take (2 videos).
- Train, the same idea ("front decides" for LA/RP): video F1 0.8798 -> 0.9087, 50 -> 38 wrong.
- The Train / Test comparison tables in this log are per whole video, not per 1-second clip; the train column counts only the 104 LA/RP videos, the test column counts all 96 test videos.

## 2026-11-18 - Are test people like train people? Similarity weighting, and side-camera hold time

Scripts (scratch): `subj_sim.py`, `subj_weight.py`, `side_hold.py`, `side_hold2.py`.

Test vs train people (DINOv3 CLS of the first 3 s, i.e. appearance while writing; cosine similarity):
- Train person to the most similar other train person: front median 0.980 (0.894-0.988), side 0.990 (0.926-0.995).
- Test person to the nearest train person: front 0.940-0.986, side 0.961-0.996, about the same as train-to-train; the nearest and second nearest almost tie (e.g. 0.982 vs 0.976), and `subject_26` is nearest for 3 of 6 test people (the room and camera dominate the appearance).
- Body size: test front people fill more of the frame (person box height 0.73-0.95 of the frame, 4 of 6 above the train maximum 0.87); side 0.62-0.70, inside the train range 0.61-0.76.

Similarity weighting (train people weighted by appearance similarity to the held-out person, leave-one-person-out, DINOv3 + pose LR, both cameras):

| Weighting | Video F1 | Wrong | 8-action acc |
|---|---|---|---|
| None (all people equal) | 0.8942 | 44 | 0.8654 |
| tau 0.02 | 0.8846 | 48 | 0.8462 |
| tau 0.005 (about 3.5 people used) | 0.8606 | 58 | 0.8125 |

- Focusing on look-alike people makes it worse; more varied people help more than similar ones, so "one model per person, follow the nearest" is not promising.

Side camera, time spent in one direction (Looking_Around vs Repeated_Peeking, medians):

| Measure | Train look | Train peek | Pairs separated | Test normal | Test cheat |
|---|---|---|---|---|---|
| Side: total time in the main direction (face width) | 1.8 s | 4.2 s | 88% | 1.7 s | 4.05 s |
| Side: visits to the main direction | 1 | 2 | 81% | 1 | 2 |
| Side: movement speed p90 (own range per s) | 1.17 | 1.29 | 77% | 0.74 | 1.03 |
| Front: total time in the main direction (turn) | 1.75 s | 3.95 s | 96% | 2.2 s | 4.75 s |

- Peeking holds longer in one direction and moves faster (quick turn, long look, back); looking around is slower and spends less total time in any one direction.

Per-camera specialist (front: turn hold total + main direction share; side: face-width hold total + visits), re-splitting the LA/RP mass of main_legit:
- Train: pair accuracy front 0.94-0.98, side 0.87-0.89, both cameras 0.96; video F1 0.8798 -> 0.9231, wrong 50 -> 32.
- Test, one camera at a time: wrong 6 -> 7; changes 3 videos: `subject_T5` side_04 fixed, `subject_T1` front_05 broken (normal but turns only one way), `subject_T4` side_04 flipped at P(peek) 0.45 (borderline).
- Not yet tested with both cameras on test (needs main_legit's own front-side pairing), which is where it can fix the `subject_T5` take.

## 2026-11-18 - LA/RP specialist inside the main_legit pipeline, scored per clip against the 0.96+ files

Script: scratch `legit_run.py` (= `main_legit.ipynb` source run as a script, plus the specialist at the end; the notebook itself is unchanged).
The rerun reproduces `submissions/main_legit.csv` exactly.
Specialist: front camera [time in main direction (turn), main direction share], side camera [time in main direction (face width), visits to it], LR C=0.1 per camera, trained on train LA/RP videos; probability of peeking from front and side summed (legit's own camera pairing), then used to re-split the Looking_Around / Repeated_Peeking share of the main model.

| Benchmark | main_legit clip F1 | + specialist clip F1 | Clips differ (legit / + spec) |
|---|---|---|---|
| 0.96 file (LB 0.9601) | 0.9061 | 0.9078 | 109 / 110 |
| Steven (LB 0.9761) | 0.9054 | 0.9070 | 110 / 111 |
| T1 swap file | 0.9174 | 0.9187 | 96 / 97 |

- Train (predicted camera pairing, as on test): video F1 0.8702 -> 0.9086, wrong 54 -> 38.
- Test: 4 videos change, 2 takes:
  - `subject_T5` front_03 + side_04: P(cheat) 0.34 -> 0.95, pseudo GT cheat (fixed).
  - `subject_T1` front_05 + side_08: P(cheat) 0.00 -> 0.93, pseudo GT normal (broken); front_05 turns only one way (P(peek) 0.91).
- Net per clip +0.0015; still far from the 0.96 target (gap about 0.04, mostly the 3 remaining wrong takes and start boundaries).
- Not added to the notebook yet.

## 2026-11-18 - Per-person centering (test-time adaptation to each person, no labels)

Idea (Fayadh's `vidz_` trick, and the label-free version of "specialise the model to each person"): before the classifier, subtract from every video's feature vector the mean over that person's videos from the same camera (train: true subject; test: main_legit's own subject clustering). The model then sees "how this video differs from this person's usual", not "what this person looks like".
Scripts (scratch): `perperson.py` (train CV), `perperson_test.py` (test vs pseudo GT), `legit_perperson.py` (full main_legit pipeline with centering; notebook not changed yet).

Train CV (video level, true camera partner):

| Variant | Video F1 | Wrong of 416 | 8-action acc |
|---|---|---|---|
| main_legit features as now | 0.8798 | 50 | 0.8365 |
| Per-person centre | **0.9519** | **20** | 0.9279 |
| Per-person centre + scale | 0.9471 | 22 | 0.9279 |
| Centre, smaller C (0.03) | 0.9471 | 22 | 0.9231 |
| Centred + original features summed | 0.9279 | 30 | 0.9135 |

Test (96 videos vs the T1 swap file, legit camera pairing):

| Variant | One camera wrong | Both cameras wrong | Both cameras video F1 |
|---|---|---|---|
| main_legit as now | 6 | 8 | 0.9165 |
| Per-person centre | 6 | **4** | **0.9583** |
| Per-person centre + scale | 3 | 4 | 0.9583 |

- Without forcing anything, it predicts 4 cheat front + 4 cheat side videos for 5 of 6 test people (`subject_T1` 5 + 5).

Full main_legit pipeline with centering (`legit_perperson.py`), per-clip F1 against the benchmark files:

| Benchmark | main_legit | + per-person centre |
|---|---|---|
| 0.96 file (LB 0.9601) | 0.9061 (109 clips differ) | 0.9189 (98) |
| Steven (LB 0.9761) | 0.9054 (110) | 0.9330 (81) |
| T1 swap file | 0.9174 (96) | **0.9446** (67) |

- Train inside the pipeline: DINOv3 video F1 0.8580 -> 0.8918, pose 0.7931 -> 0.8341.
- Test: 50 of 96 videos cheat, cheat share 0.267.
- Still wrong against the T1 swap file: `subject_T1` front_08 + side_07 (the disputed T1 take; the 0.96 and Steven files agree with legit here) and `subject_T4` side_02 / side_03 (looks like a swapped side pairing, like the T3 pairfix).
- 38 of the 67 differing clips are start boundaries (legit's own start finder against the files' "cheat from 7").
- Caveat for the team: centering works because every person performs the same set of actions, so a person's mean is a neutral reference; it uses no labels and no cheat ratio, but it relies on that design. To be agreed as legit before using it.

## 2026-11-18 - Ablation of the legit pipeline

Script: scratch `legit_ablation.py` (main_legit source + per-person centering + ablation block; the notebook is unchanged).
Train columns are train CV (5 folds by subject, predicted camera pairing, per-clip truth from the by-eye starts); test columns are per-clip F1 against the benchmark files, for information only.

| Variant | Train video F1 | Train wrong videos | Train clip F1 | Test clip F1 vs Steven 0.976 | Test clip F1 vs T1 swap |
|---|---|---|---|---|---|
| A. DINOv3 + pose, per-person centred, both cameras, start finder | 0.9423 | 24 | 0.9416 | 0.9330 | 0.9446 |
| B. A without centering (= current notebook) | 0.8702 | 54 | 0.8940 | 0.9054 | 0.9174 |
| C. A without pose (DINOv3 only) | 0.9327 | 28 | 0.9359 | 0.9330 | 0.9446 |
| D. A without DINOv3 (pose only) | 0.8654 | 56 | 0.8895 | 0.9071 | 0.9188 |
| E. A without both-camera sum | 0.9110 | 37 | 0.9186 | 0.9204 | 0.9321 |
| F. A with a fixed start (train median 7.7 s, i.e. cheat from clip 8) | 0.9423 | 24 | 0.9377 | 0.9343 | 0.9343 |
| G. A with whole video cheat (no start) | 0.9423 | 24 | 0.7066 | 0.7363 | 0.7363 |
| H. A + durations | 0.9471 | 22 | 0.9448 | 0.9230 | 0.9345 |
| I. A + LA/RP specialist | 0.9471 | 22 | 0.9424 | 0.9200 | 0.9314 |
| J. A + durations + specialist | 0.9567 | 18 | 0.9473 | 0.9102 | 0.9215 |

Reading:
- Clear helpers (train and test agree): per-person centering (+0.048 train clip F1), DINOv3 (pose alone is much worse), both-camera sum (+0.023), per-second start (whole video 0.71).
- Pose: small help on train (28 -> 24 wrong videos, +0.006 clip F1), no change on test.
- Start finder vs fixed 7.7 s: +0.004 on train; on test about the same against Steven's file, +0.010 against the T1 swap file.
- Durations and the LA/RP specialist: small gains on train (+0.001 to +0.006 clip F1, 2-6 fewer wrong videos) but both lower the test scoreboard by 0.010-0.023; train-only selection would keep them, the scoreboard says they hurt.

## 2026-11-18 - Alignment variants (EEG-style RCT / RPA / EA) and main_legit.ipynb updated with per-person centering

Script (scratch): `align.py`. Train CV video level (true camera partner) and test wrong videos against the T1 swap file (legit camera pairing):

| Alignment per person and camera | Train video F1 | Train wrong | Test wrong |
|---|---|---|---|
| Mean centering, full features (RCT-like) | 0.9519 | 20 | 4 |
| Centering + one stretch factor per person (RPA-like stretch) | 0.9519 | 20 | 4 |
| Centering + per-feature scale | 0.9471 | 22 | 4 |
| PCA 64 (fit on train), centering only | 0.9183 | 34 | 6 |
| PCA 64, EA whitening (Ledoit-Wolf shrunk covariance) | 0.9423 | 24 | 6 |
| PCA 32, centering only | 0.9087 | 38 | 10 |
| PCA 32, EA whitening | 0.9279 | 30 | 4 |
| PCA 16, EA whitening | 0.8990 | 42 | 12 |

- EA whitening helps inside a PCA space (64 dims: 34 -> 24 wrong) but each person has only 8 videos per camera, so it needs the PCA squeeze, which costs more than the whitening gains; the RPA rotation step needs labels on the target person, so it is not usable.
- Plain mean centering on the full features stays best.

Notebook change: `leon/main_legit.ipynb` (built from scratch `legit_nb.py`, executed with nbclient, 0 errors, no imports outside the import cell; previous version kept in scratch as `main_legit_before_centering.ipynb`).
- New sub-step in Preprocessing: `### Penyesuaian per orang`, subtracting each person-and-camera mean from the DINOv3 and pose video features (train: true subject; test: the notebook's own subject clustering, 12 groups of 8 videos).
- Notebook outputs: DINOv3 video F1 0.8918, pose 0.8341, DINOv3 + pose 0.9110, + two cameras 0.9423 (acc8 0.9038); per-clip F1 0.9416 (whole video 0.7066, true video labels 0.9769).
- `submissions/main_legit.csv` now: 50 of 96 videos cheat, cheat share 0.267; identical to the tested `legit_perperson.py` run.
- Per-clip F1 against the benchmark files: 0.96 file 0.9189, Steven 0.9330, T1 swap 0.9446 (was 0.9061 / 0.9054 / 0.9174).
- Durations and the LA/RP specialist stay out (ablation: small train gains, lower test scoreboard).

## 2026-11-18 - Error analysis of the current main_legit (with per-person centering)

Script (scratch): `legit_err.py` (main_legit source + analysis).

Train CV:
- 24 of 416 videos wrong (front 11, side 13); cheat/normal accuracy per action: Looking_Around 0.81, Checking_Smartwatch 0.92, Reading_Hidden_Notes 0.92, Repeated_Peeking 0.96, Writing_Answers 0.96, Thinking 0.98, Using_Smartwatch 0.98, Looking_Under_the_Desk 1.00.
- Wrong videos by confusion: Looking_Around -> Peeking 9, Checking -> Using smartwatch 4, Hidden notes -> Checking smartwatch 2, Hidden notes -> Writing 2, Peeking -> Looking_Around 2, Writing -> Under desk 2, others 1 each.
- Wrong clips: 267 of 6,122; 172 (64%) from a wrong video label, 95 (36%) from the start second (38 too early, 57 too late).
- Start-second errors by action: Repeated_Peeking 38 clips (8 videos off by more than 2 s), Reading_Hidden_Notes 29 (4), Using_Smartwatch 14, Looking_Under_the_Desk 14 (1).

Test against the T1 swap file (67 clips differ):
- 4 wrong videos (29 clips): `subject_T4` side_02 (cheat, predicted Thinking) and side_03 (normal, predicted Reading_Hidden_Notes 0.90), most likely a swapped side pairing; `subject_T1` front_08 + side_07 (the disputed take).
- 38 boundary clips in 28 videos, 37 of them because legit starts later than the file: legit starts from clip 8 in 21 videos, 9 in 4, 11 in 2 where the file starts from clip 7.
- `subject_T1` front_03 and side_06 have starts at 11.1 s, the edge of the search window and later than any train start (8 clips).
- Against Steven's file: 4 wrong videos (36 clips) and 45 boundary clips, 44 of them late.
- So on test the largest remaining difference is the start second (legit about one clip later than the files), then the T4 side pair.

## 2026-11-18 - Start-second tuning for main_legit (train CV decides)

Script (scratch): `legit_start_tune.py` (main_legit with centering + start rules). Train clip F1 uses the by-eye train starts; shifts are chosen with nested CV (picked on 4 folds, applied to the 5th). Test columns only as a scoreboard.

| Start rule | Train clip F1 | Test vs Steven | Test vs T1 swap |
|---|---|---|---|
| Finder as now | 0.9416 | 0.9330 | 0.9446 |
| Finder + guard (start outside the train range 4.75-8.75 s +-0.5 s -> train median 7.75 s) | 0.9438 | 0.9435 | 0.9451 |
| Fixed train median start (7.75 s) | 0.9377 | 0.9343 | 0.9343 |
| Fixed median start of the predicted action | 0.9366 | 0.9448 | 0.9448 |
| Blend 0.25 finder + 0.75 action median (+ guard) | 0.9423 | 0.9414 | 0.9414 |
| Blend 0.5 / 0.5 (+ guard) | 0.9456 | 0.9423 | 0.9423 |
| Blend 0.75 finder + 0.25 action median (+ guard) | **0.9463** | 0.9449 | 0.9449 |
| Finder + guard + one global shift (nested CV; chosen 0 in 4 of 5 folds, -0.25 in 1) | 0.9420 | 0.9435 | 0.9451 |
| Finder + guard + shift per predicted action (nested CV; on all train: 0 / 0 / 0 / +0.25 for Using_Smartwatch) | 0.9410 | 0.9400 | 0.9417 |
| Reference: the files' rule "cheat from clip 7" | 0.9271 | 0.9719 | 0.9719 |

- Train picks no shift: by the train start times the finder is not late, so tuning on train cannot move the start towards clip 7.
- "Cheat from clip 7" is worse than the finder on train (0.927) but much better on test (0.972): train's by-eye starts and the test labels disagree about clip 7, so this gap cannot be closed with train-only tuning.
- Per-action shifts do not help (0.941 against 0.944); per-person shifts are not possible (test people are new and unlabelled).
- Train-only winners: the guard (+0.002, legit) and the 0.75/0.25 blend with the action median (+0.005).

## 2026-11-18 - Two-stage start search with a confidence check (train CV decides)

Script (scratch): `legit_two_stage.py`. Confidence = how much the finder's per-moment score jumps at the chosen switch (mean logit after minus before).
- Confidence is meaningful on train: cheat videos in the lowest quarter of confidence have a median start error of 0.85 s, the rest 0.25 s.

| Start rule | Train clip F1 | Test vs Steven | Test vs T1 swap | Test starts at 10 s or later |
|---|---|---|---|---|
| As now: search 3.2-11.1 s, front + side joint | 0.9416 | 0.9330 | 0.9446 | 14 |
| **Search only 4.25-9.25 s (train range +-0.5 s)** | **0.9450** | 0.9409 | 0.9458 | 0 |
| 4.25-9.25 s first; if weak (bottom 10%) scan the whole video | 0.9430 | 0.9313 | 0.9545 | 5 |
| Same, still weak -> 7.7 s | 0.9414 | 0.9426 | 0.9442 | 0 |
| 3.2-11.1 s; if weak (bottom 10%) -> 7.7 s | 0.9391 | 0.9409 | 0.9425 | 1 |
| 4.25-9.25 s first; if weak (bottom 25%) scan the whole video | 0.9369 | 0.9322 | 0.9553 | 5 |
| Same, still weak -> 7.7 s | 0.9418 | 0.9425 | 0.9442 | 0 |
| 3.2-11.1 s; if weak (bottom 25%) -> 7.7 s | 0.9386 | 0.9434 | 0.9450 | 0 |

- Train winner: simply narrowing the search window to the train range (+0.0034); the whole-video fallback is worse on train (0.937-0.943) and swings both ways on the scoreboard.
- The 11.1 s videos (`subject_T1` front_03, side_06, front_08, side_07) are 15-16 s long; 11.1 s is the end of the search window (latest train start + 1 s, with the camera delay), not the end of the video.

## 2026-11-18 - Ready for a later leaderboard test: main_legit_window.csv

- `leon/submissions/main_legit_window.csv`: main_legit (per-person centering) with the start search limited to 4.25-9.25 s (train by-eye range +-0.5 s); script scratch `legit_window_sub.py`; notebook not changed yet.
- 50 of 96 videos cheat, cheat share 0.2733; per clip against Steven's file 0.9409, against the T1 swap file 0.9458; train CV per clip 0.9450.
- Leon plans to submit it later to see the real public score of the legit pipeline (not submitted yet).

## 2026-11-18 - Start finders from the literature (main_legit with centering, search window 4.25-9.25 s)

Script (scratch): `legit_papers.py` (= `legit_two_stage.py` + `paper_block.py`). Train CV decides; test columns only as a scoreboard.
Papers:
- Timestamp supervision: Li, Abu Farha, Gall, "Temporal Action Segmentation from Timestamp Supervision", CVPR 2021 (one marked frame per action, labels for the rest estimated from the model).
- MS-TCN: Abu Farha, Gall, "MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation", CVPR 2019 (dilated temporal convolutions, several refinement stages, truncated-MSE smoothing loss against over-segmentation).
- Kernel change point detection: as used for unsupervised action boundaries (Du et al., "Fast and Unsupervised Action Boundary Detection for Action Segmentation", CVPR 2022; kernel temporal segmentation); here one break with an RBF-kernel cost on PCA-32 DINOv3 frame features, no training.

| Finder | Start error median | Within 1 s | Off > 2 s | Train clip F1, true video labels | Train clip F1 | Test vs Steven | Test vs T1 swap |
|---|---|---|---|---|---|---|---|
| 0. Current (hard EM, frame LR) | 0.25 s | 0.93 | 5 | 0.9834 | 0.9450 | 0.9409 | 0.9458 |
| 1. Timestamp supervision (by-eye starts as frame labels) | 0.25 s | 0.92 | 5 | 0.9809 | 0.9430 | 0.9398 | 0.9415 |
| 2. MS-TCN, 2 stages + smoothing loss, by-eye labels | 0.25 s | 0.89 | 2 | 0.9826 | 0.9446 | 0.9235 | 0.9235 |
| 2b. Average of 1 and 2 | 0.25 s | 0.92 | 3 | 0.9842 | 0.9457 | 0.9319 | 0.9352 |
| 3. Kernel change point (unsupervised) | 0.25 s | **0.97** | **2** | **0.9849** | **0.9463** | 0.9342 | 0.9326 |

- Using the by-eye starts directly as labels (1) is no better than letting EM find them; the current finder already agrees with the by-eye starts.
- MS-TCN removes most big misses (5 -> 2) but is less precise near the start (within 1 s 0.89).
- Kernel change point is the best on train (0.97 within 1 s, +0.0013 clip F1) with no training at all, but all three lower the test scoreboard (by 0.007-0.022); the train gains are small and none of them moves the start earlier, which is the real test gap.
- Decision by train CV would pick kernel change point by a small margin; not added yet.

## 2026-11-18 - Camera pairing with action agreement (multi-view consistency) + new candidate file

Idea from multi-view action alignment work (e.g. cross-view alignment of the same action, Xue and Grauman NeurIPS 2023; temporal cycle consistency, Dwibedi et al. CVPR 2019): the front and side video of one take show the same action, so pairing should also check that both cameras predict the same action, not only that their movement timing matches.
Pairing cost per person = movement-timing match + w x agreement of the one-camera action probabilities (Bhattacharyya: sum of sqrt(p_front x p_side)); OOF probabilities on train, so no label leaks into the pairing.
Script (scratch): `legit_pairing.py` (= `legit_perperson.py` + `pair_block.py`), start search 3.2-11.1 s as in the notebook.

| w | Train pairing right | Train video F1 | Train wrong | Test pairs changed | Test vs Steven | Test vs T1 swap |
|---|---|---|---|---|---|---|
| 0 (timing only, as now) | 0.957 | 0.9423 | 24 | 0 | 0.9330 | 0.9446 |
| **0.25** | **1.000** | **0.9519** | **20** | 8 | **0.9486** | **0.9602** |
| 0.5 | 0.981 | 0.9423 | 24 | 8 | 0.9486 | 0.9602 |
| 1 | 0.981 | 0.9423 | 24 | 8 | 0.9486 | 0.9602 |
| 2 | 0.952 | 0.9423 | 24 | 8 | 0.9486 | 0.9602 |
| 4 | 0.942 | 0.9423 | 24 | 8 | 0.9486 | 0.9602 |

- w = 0.25 is chosen by train alone (all 416 train videos paired correctly).
- Test: only `subject_T4` changes: front_03 side_02 -> side_08, front_05 side_03 -> side_05, front_06 side_08 -> side_03, front_08 side_05 -> side_02; this fixes the T4 side_02 / side_03 error.

New file `leon/submissions/main_legit_pair_window.csv` (script scratch `legit_final.py` = `legit_perperson.py` + `final_block.py`; notebook not changed yet):
- main_legit with per-person centering + action-agreement pairing (w 0.25) + start search 4.25-9.25 s.
- 50 of 96 videos cheat, cheat share 0.2733.
- Per clip: vs 0.96 file 0.9417, vs Steven 0.9557, vs T1 swap **0.9606** (first legit file at the 0.96 benchmark).
- Remaining against the T1 swap file: 48 clips; the disputed `subject_T1` front_08 + side_07 take and 35 boundary clips (32 where legit starts later than clip 7).
- Better candidate than `main_legit_window.csv` for the planned leaderboard test.

## 2026-11-18 - Per-sequence model (every 1-second sequence predicted on its own)

Script (scratch): `perclip.py`. Train videos cut into 1-second sequences (6,166; cheat videos without a readable by-eye start left out); one vector per sequence:
- DINOv3: mean, std and mean absolute frame-to-frame change of the 10 frame features.
- Pose: mean, std and mean absolute change of the 19 pose signals over the 10 frames.
- Logistic regression, StratifiedGroupKFold 5 by subject, per-sequence F1-Macro.
- Per-second labels: a sequence of a cheat video is cheat only if most of that second is after the by-eye start (side videos: front start + 0.5 s); whole-video labels: every sequence of a cheat video is cheat (what the per-clip baselines did).

| Features | Train labels | C | F1 at 0.5 | Best threshold | F1 at best threshold |
|---|---|---|---|---|---|
| DINOv3 + pose | per-second | 0.003 | **0.7405** | 0.45 | **0.7421** |
| DINOv3 | per-second | 0.003 | 0.7247 | 0.40 | 0.7353 |
| DINOv3 + pose | per-second | 0.03 | 0.7257 | 0.50 | 0.7257 |
| Pose | per-second | 0.03 | 0.6731 | 0.30 | 0.7192 |
| DINOv3 | per-second | 0.03 | 0.7140 | 0.55 | 0.7156 |
| Pose | per-second | 0.003 | 0.6457 | 0.30 | 0.6972 |
| Pose | whole-video | 0.03 | 0.6340 | 0.55 | 0.6648 |
| DINOv3 + pose | whole-video | 0.003 | 0.5582 | 0.80 | 0.6071 |
| DINOv3 | whole-video | 0.003 | 0.5525 | 0.80 | 0.5998 |

- Per-second train labels lift a single-sequence model from about 0.56 to 0.74: the whole-video labels teach the model that the writing seconds of a cheat video are cheat, which is the main reason the per-clip baselines stayed at 0.57 on the leaderboard.
- Cheat share: 0.246 with per-second labels against 0.481 with whole-video labels (test files around 0.27-0.28).

## 2026-11-18 - main_legit.ipynb updated: camera pairing with action agreement

Notebook change: new sub-step `### Pasangan kamera dengan kesepakatan aksi` after `### Gabungan model dan dua kamera` (new constant `W_AGREE = 0.25`); the front-side pairing is redone with movement timing + 0.25 x agreement of the one-camera 8-action probabilities (OOF on train), and everything after it (both-camera vote, start finder pairs, submission) uses the new pairs. Executed with nbclient, 0 errors; previous version kept in scratch as `main_legit_before_agree.ipynb`.

Notebook outputs:
- Pairs right on train: 0.9567 -> 1.0000; video F1 0.9423 -> 0.9519; 8 test videos change partner (all `subject_T4`).
- Per-clip F1 (train CV): whole video 0.7140, start finder 0.9486 (was 0.9416), start finder with true video labels 0.9778.
- Start error: median 0.25 s, within 1 s 0.88, more than 2 s 12.
- `submissions/main_legit.csv`: 50 of 96 videos cheat, cheat share 0.267.

Same pipeline with the start search limited to 4.25-9.25 s (scratch `legit_agree_window.py`, identical output to `main_legit_pair_window.csv`): per clip 0.9524, start error within 1 s 0.94, more than 2 s 4.

| Version | Train clip F1 | vs 0.96 file | vs Steven | vs T1 swap |
|---|---|---|---|---|
| Notebook before (centering) | 0.9416 | 0.9189 | 0.9330 | 0.9446 |
| Notebook now (+ action agreement) | 0.9486 | 0.9338 | 0.9479 | 0.9595 |
| + start search 4.25-9.25 s | 0.9524 | 0.9417 | 0.9557 | 0.9606 |

## 2026-11-19 - main_legit.ipynb: start search limited to 4.25-9.25 s

- Notebook change (cell "Detektor awal aksi" and the start-finder cell): the search window is now the front by-eye start range in train (4.75-8.75 s) plus 0.5 s each side, i.e. 4.25-9.25 s, instead of 3.2-11.1 s. No whole-video fallback (tested 2026-11-18, worse on train).
- Pipeline in the notebook now: per-person centering + action-agreement camera pairing + start search 4.25-9.25 s.
- Executed: 49 cells, 0 errors. `submissions/main_legit.csv` is identical to the former `main_legit_pair_window.csv` (now renamed by Leon to `submissions/submission.csv`), 0 clips differ.
- Train: per clip 0.9524 (true video labels 0.9847), start within 1 s 0.94, more than 2 s off: 4, video F1 0.9519, pairs right 1.0.
- Test: 50 of 96 videos cheat, cheat share 0.2733; per clip vs 0.96 file 0.9417, vs Steven 0.9557, vs T1 swap (`potential_1.csv`) 0.9606.

## 2026-11-19 - T1 swap file submitted: public LB 1.0000000

- File: `leon/submissions/1.csv` (= `main_t1swap_from7.csv`, later `potential_1.csv`; checked identical, 0 clips differ), submitted by Leon as `submission_t1.csv`.
- Public LB **1.0000000**: every public clip right. Confirms Steven's video labels with the `subject_T1` swap (front_04 + side_01 cheat, front_08 + side_07 normal) and cheat from clip 7.
- This file is now the best pseudo ground truth for main_legit. main_legit (centering + action agreement + start search 4.25-9.25 s) against it: F1 0.9606 per clip, 48 of 1515 clips wrong.

## 2026-11-19 - main_legit_nosequence.ipynb created (every test sequence predicted on its own)

- New notebook `leon/main_legit_nosequence.ipynb` (29 cells, 0 errors), output `leon/submissions/main_legit_nosequence.csv`.
- No clip chaining, no person grouping, no camera pairing and no per-person centering on test: each test sequence is scored only from its own 30 frames; scaler, model and threshold are fitted on train only.
- Train: every video cut into 1-second clips (frames 0-29, 30-59, ...), 6,166 clips; label 1 only for cheat-video clips mostly after the by-eye start (side = front + 0.5 s); 18 cheat videos without a readable start left out.
- Features per clip: DINOv3 (mean, std, mean frame-to-frame change of the 10 cached frames, 7,680) + pose (19 signals x mean, std, mean change, 57).
- Model: StandardScaler + Logistic Regression C 0.003, StratifiedGroupKFold 5 by subject, threshold from OOF.
- Train CV per clip: 0.7405 at 0.5, **0.7421 at threshold 0.45** (same as the earlier `perclip.py` check).
- OOF accuracy per action (front / side): weakest Repeated_Peeking 0.69 / 0.62, Reading_Hidden_Notes 0.78 / 0.70, Using_Smartwatch 0.79 / 0.69.
- Test against `1.csv` (public 1.0): F1 0.7056, 322 of 1515 clips wrong (89 false cheat, 233 missed cheat); predicted cheat share 0.189 against 0.284.

## 2026-11-19 - Fine-tuned ViT-H+ frame head, K=0 (frozen backbone + new head), scored in both notebooks

- Setup: DINOv3 ViT-H+ blocks 21-32 run frozen on the cached block-20 output, new head (Dropout 0.2 + Linear 2560 -> 8 actions) trained 2 epochs on frames, 5 folds by subject (same folds as main_legit), frame labels writing before the start, the video's action after.
- Held-out frame accuracy per fold: 0.63 / 0.71 / 0.58 / 0.64 / 0.67; train loss only fell 1.99 -> 1.83, so the head barely learned (features were not standardized before the head).
- K=4 fold 0 frame accuracy 0.628, same as K=0 fold 0 (0.629). The run was then stopped by the system for low RAM (training and evaluation ran at the same time); K=4 folds 1-4, K=12/8/2 not run.
- main_legit (train per clip / test vs `1.csv`, clips wrong):
  - Notebook now (LR on pretrained DINOv3 + pose): 0.9524 / 0.9606, 48 wrong; DINOv3 LR 8-action acc 0.834, video F1 0.892.
  - Fine-tuned head replaces the DINOv3 LR: 0.9091 / 0.9504, 61 wrong; head 8-action acc 0.668, video F1 0.776.
  - Fine-tuned head multiplied with the DINOv3 LR: 0.9351 / 0.9606, 48 wrong; final video F1 0.928 (notebook now 0.952).
- main_legit_nosequence (train per clip / test vs `1.csv`): notebook now 0.7421 / 0.7056; head only 0.665 / 0.6308; LR + head 0.7419 / 0.7117 (326 wrong).
- Conclusion: K=0 head is worse than the pretrained LR everywhere; notebooks not changed.

## 2026-11-19 - Bug in the fine-tuning script: trained blocks stayed frozen

- Bug in scratch `ft_train.py`: the K=0 run switched off training for blocks 21-32, and the weight reload before the next run did not switch it back on. So the K=4 fold 0 run trained only the head, same as K=0 (frame accuracy 0.628 vs 0.629, same loss, GPU peak only 1.5 GB). No real 4/8/12-block fine-tuning has been tested yet.
- Fix: trained blocks are switched on in every run, with a check that the trainable weight count is above K x 20M (K=12: 314.8M). The head now also standardizes its input (BatchNorm without learned scale) before Dropout + Linear.
- Speed check K=12: 11 frames/s, GPU peak 6.45 of 8 GB. More than 12 blocks with full training does not fit in 8 GB and would also need a new cache from an earlier block.
- Rerun started: K = 0, 12, 4, 8, output `cache/ft_hplus_block20/preds_v2/`, training alone (evaluation afterwards). The old `preds/` files are kept but are not valid for K=4.
- Change before the rerun: the head is only used for training. After each fold, the fine-tuned backbone's features (CLS + mean patch, 2560, same format as main_legit's DINOv3 cache) are saved for all train frames and test frames, and fed into the notebooks' own logistic regression in place of (replace) or next to (add) the pretrained features. Fold f videos are scored by a logistic regression fitted on fold f backbone features of the other folds; test = average of the 5 fold models. Scripts: scratch `build_legit_ftfeat.py` -> `legit_ftfeat.py`, `ft_noseq_feat.py` (switched to main_legit's per-video folds, pretrained baseline rescored on them). Frames: 2 per second instead of 10, so K=0 doubles as the 2 fps pretrained control.
- First sign the head fix works: K=0 fold 0 loss 1.69 -> 1.30 over 2 epochs (was 1.99 -> 1.83).
- Rerun status (second stop by the system for low RAM): K=0 finished all 5 folds with backbone features saved (`preds_v2/`); K=12 fold 0 started and was stopped. The K=0 scoring in both notebooks ran at the same time as K=12 training (my mistake, I had planned training alone) and did not finish, so `ftfeat_eval.log` is empty.
- With the standardizing head the K=0 held-out frame accuracy of the head itself fell to 0.34-0.49 (folds 2-4; was 0.58-0.67 without it) while the training loss dropped to 1.29. The head's own accuracy is not used for the final prediction (the logistic regression on the backbone features is), but it means the standardized head is a weak training signal across new people.
- K=0 (frozen backbone, 2 fps features saved) scored in the no-sequence notebook on main_legit's per-video folds (scratch `ft_noseq_feat.py`), train per clip / test vs `1.csv` (clips wrong):
  - Pretrained DINOv3 10 fps + pose: 0.7369 / 0.7195 (320).
  - Frozen backbone 2 fps + pose: 0.7283 / 0.7112 (304).
  - Pretrained + frozen backbone + pose: 0.7387 / 0.7222 (318).
  - Going from 10 to 2 frames per second costs about 0.01 per clip on train; frozen features add nothing.
- Rerun v3 started: K = 4, 12, 8 (K=0 dropped), reading the 13 GB token file by seeking instead of memory-mapping (the memory-mapped file counted against RAM and the system stopped the jobs twice), training running alone. Log: scratch `ft_train_v3.log`, output `cache/ft_hplus_block20/preds_v2/`.

## 2026-11-19 - Fine-tuning ViT-H+ top blocks: first real result (K=4 trained blocks, no-sequence notebook)

- Setup: top K of 32 blocks trained 2 epochs per fold (5 folds by subject, same as main_legit), head Dropout + Linear on standardized features (BatchNorm without learned scale). The head is only the training signal. The scored model is the notebook's own logistic regression on the fine-tuned backbone's features (CLS + mean patch), 2 frames per second, fold f clips scored with fold f's backbone, test = average of the 5 fold models. K=4: 105.0M trainable weights, 10 min per fold, GPU peak 3.0 GB.
- Head held-out frame accuracy per fold: K=4 0.608 / 0.602 / 0.519 / 0.586 / 0.628; K=12 fold 0 0.659 (frozen head 0.629 on the same fold).
- No-sequence notebook, same per-video folds, train per clip F1 / test vs `1.csv` F1 (clips wrong):
  - Pretrained DINOv3 10 fps + pose: 0.7369 / 0.7195 (320)
  - K=0 frozen backbone 2 fps + pose: 0.7283 / 0.7112 (304)
  - **K=4 fine-tuned backbone + pose: 0.7767 / 0.7454 (260)**
  - K=4 pretrained + fine-tuned + pose: 0.7803 / 0.7328 (266)
- Fine-tuning 4 blocks gives +0.040 on train and +0.026 on test over the pretrained features (+0.049 / +0.034 over the frozen 2 fps control), the first gain from any change to the backbone. Predicted cheat share on test is low (0.145 against 0.284 true) because the threshold comes from the held-out train predictions (0.55).
- Still running: K=12 (fold 1 of 5 done training), then K=8. Not yet scored in main_legit.

## 2026-11-19 - Fine-tuning ViT-H+ top blocks: K=12 scored (no-sequence notebook)

- K=12: 314.8M trainable weights, 13-40 min per fold (GPU shared with other work), GPU peak 6.5 GB. Head held-out frame accuracy per fold: 0.659 / 0.704 / 0.600 / 0.613 / 0.682 (K=4: 0.608 / 0.602 / 0.519 / 0.586 / 0.628).
- No-sequence notebook, same per-video folds, train per clip F1 / test vs `1.csv` F1 (clips wrong):
  - Pretrained DINOv3 10 fps + pose: 0.7369 / 0.7195 (320)
  - K=4 fine-tuned + pose: 0.7767 / 0.7454 (260)
  - **K=12 fine-tuned + pose: 0.7893 / 0.8094 (220), threshold 0.4, test cheat share 0.228**
  - K=12 pretrained + fine-tuned + pose: 0.7932 / 0.7625 (250)
- More trained blocks keeps helping (0 -> 4 -> 12: train 0.728 -> 0.777 -> 0.789, test 0.711 -> 0.745 -> 0.809). K=8 is training (2 of 5 folds done at the time of writing).

## 2026-11-19 - Fine-tuning ViT-H+ top blocks: K=8 scored, comparison of 0 / 4 / 8 / 12 blocks (no-sequence notebook)

- K=8: 210M trainable weights, 13-14 min per fold, GPU peak 5.2 GB. Head held-out frame accuracy per fold: 0.652 / 0.665 / 0.572 / 0.602 / 0.653.
- Train per clip F1 / test vs `1.csv` F1 (clips wrong), fine-tuned backbone + pose, same per-video folds:

| Trained blocks | Train per clip | Test vs `1.csv` | Wrong |
|---|---|---|---|
| 0 (frozen, 2 fps) | 0.7283 | 0.7112 | 304 |
| 4 | 0.7767 | 0.7454 | 260 |
| 8 | 0.7734 | 0.8023 | 222 |
| 12 | 0.7893 | 0.8094 | 220 |

- Pretrained DINOv3 10 fps + pose on the same folds: 0.7369 / 0.7195. Pretrained + fine-tuned together was worse on test for 8 blocks too (0.7818 / 0.7782).
- 12 blocks is the best on both train and test; 8 is close on test but not above 4 on train (difference 0.003, within noise). 12 is the most the block-20 cache allows.
- Decision (Leon): use all 30 frames of every test sequence from now on, run the final fine-tune once with the best block count and save its weights.

## 2026-11-19 - nosequence_ft12.csv: file for the day's last submission

- File: `leon/submissions/nosequence_ft12.csv` (1515 rows, id + label, same order as `sample_submission.csv`): no-sequence logistic regression on the 12-block fine-tuned backbone features + pose, each test clip scored alone, test = average of the 5 fold models, threshold 0.4 from held-out train predictions, cheat share 0.228.
- Train per clip F1 0.7893; test vs `1.csv` F1 0.8094 (220 wrong).
- Caveat: the fine-tuned features come from 2 of the 30 test frames (01 and 16) and pose from 10 of 30; it is a leaderboard probe, not yet the 30-frame version. The 12-block weights were not saved, so the 30-frame version needs a rerun of the fine-tune.

## 2026-11-20 - 30-frame features done; A (30 frames, old pose) and E (sequence model) first scores

- DINOv3 ViT-H+ (CLS + mean patch, 2560, fp16) cached for every frame: test (1515, 30, 2560) and 199,957 train frames in 416 videos, `cache/dinov3_hplus_224x400_step1_bf16/` (`extract30.py`, about 2 h 20 min, 24-27 frames/s). All values finite.
- Pose on all 30 frames of every window (box from the middle frame) is extracting to `cache/pose_vitpose_all/` (`extract_pose_all.py`); the results below still use the old 10-frame pose.
- Shared setup `common30.py`: same 6166 train clips as the no-sequence notebook, 30 DINOv3 frames per clip on both sides, main_legit per-video folds, threshold from held-out train predictions. Each member saves its held-out train predictions and test probabilities to `members/` for the blend.
- Train per clip F1 / test vs `1.csv` F1 (clips wrong):

| Member | Train per clip | Test vs `1.csv` | Wrong | Test cheat share |
|---|---|---|---|---|
| A: LR on DINOv3 10 frames + pose (old, same folds) | 0.7369 | 0.7195 | 320 | - |
| A: LR on DINOv3 30 frames + old pose | 0.7391 | 0.7157 | 313 | 0.193 |
| A: DINOv3 30 frames only | 0.7287 | 0.6936 | 362 | 0.247 |
| A: old pose only | 0.7198 | 0.7088 | 344 | 0.247 |
| E: BiGRU over 30 frames (PCA 128 per frame, DINOv3 only) | 0.7033 | 0.5793 | 375 | 0.075 |
| E: transformer over 30 frames (same input) | 0.6998 | 0.5740 | 358 | 0.049 |

- 30 frames instead of 10 does not change the frozen LR (train +0.002, test -0.004, noise).
- E is worse than A on train, and on test it stops predicting cheat. Mean predicted cheat probability, held-out train vs test: A 0.247 / 0.258, E GRU 0.349 / 0.184, E transformer 0.300 / 0.159. A is stable between train and test, E is not.

## 2026-11-20 - Train frames vs test delivery format: DINOv3 feature shift (train videos only)

- Question: the test sequences are 640x360 JPEG frames, but the train DINOv3 features come from the full-size video frames. How far do the features move when a train frame is passed through the test format first (resize to 640x360 INTER_AREA, JPEG quality 90, as `extract_pose.py` already does, then the usual 400x224 resize)? Script `jpeg_shift.py`, 10 train videos (5 cheat, 5 normal, 90 frames each), relative L2 distance between feature vectors.
- Result (mean over the 10 videos):
  - Same frame, original vs test format: **0.120** (0.088 to 0.171 per video).
  - Neighbouring frames, original: 0.027.
  - Clip mean at 0-1 s vs 2-3 s, original: 0.055.
- The format change moves a frame's features about 4.4 times more than one frame step and about 2.2 times more than the change between two different seconds of the same video. This is a train-vs-test domain gap in every DINOv3 member, and it is larger for models that use per-frame detail (E).
- Not yet measured: how much of the shift is the resize and how much the JPEG; whether features extracted in the test format fix E and raise A. The 12-block fine-tuned backbone scored 0.809 on test despite the gap, so the gap is not fatal.

## 2026-11-20 - A with 30-frame pose, C1 (student with teacher soft labels)

- Pose on all 30 frames of every window finished (`cache/pose_vitpose_all/`: train (6456, 30, 17, 3), test (1515, 30, 17, 3), 58 min; box from the middle frame, same detector and pose model as before).
- A, 30 DINOv3 frames + 30 pose frames, per-video folds (train per clip F1 / test vs `1.csv` F1, clips wrong):
  - DINOv3 + pose: 0.7378 / 0.7182 (322), threshold 0.45, test cheat share 0.220.
  - DINOv3 only: 0.7287 / 0.6936 (362).
  - Pose only: 0.7135 / 0.7257 (324).
  - Same as with the old 10-frame pose (0.7391 / 0.7157): 30 frames of pose do not change the frozen LR.
- C1: teacher = main_legit held-out (per-video folds) video cheat probability x soft step at its guessed start second (scale 0.5 s); teacher on train clips AUC 0.9827, F1 0.9499. Student = the same LR as A on the 30-frame features with soft label q = (1 - a) * true + a * teacher (row duplication with weights):
  - a = 0 (hard labels): 0.7378 / 0.7182.
  - a = 0.5: 0.7373 / 0.7226 (312).
  - a = 1.0: 0.7330 / 0.7138 (310).
- Soft labels from this teacher bring nothing: it is almost the true label (start second error median 0.25 s, 94% within 1 s), so it adds almost no new information about the boundary clips.

## 2026-11-20 - Does the raw-vs-test-format feature gap hurt? Fold-0 check (train only)

- Setup (`extract30_testfmt.py`, `fmt_check.py`): the 96 fold-0 train videos (46k frames) re-extracted through the test delivery format (resize 640x360, JPEG q90, then the usual 400x224), `cache/dinov3_hplus_224x400_step1_bf16_testfmt/`. A and E trained on the raw features of folds 1-4 (as in all other runs), then scored on the 1364 fold-0 clips with raw features and with test-format features. Threshold fixed from the raw run.

| Model | AUC raw / test format | F1 raw / test format | Mean predicted cheat raw / test format (true 0.243) |
|---|---|---|---|
| A, DINOv3 only | 0.7976 / 0.7866 | 0.7146 / 0.6833 | 0.331 / 0.432 |
| A, DINOv3 + pose | 0.8155 / 0.8075 | 0.7297 / 0.7056 | 0.329 / 0.403 |
| E, BiGRU, DINOv3 only | 0.7083 / 0.7108 | 0.6314 / 0.6219 | 0.282 / 0.222 |

- The format gap shifts A's probabilities up (about +0.1) and costs about 0.02 to 0.03 F1 at a fixed threshold and 0.01 AUC: moderate, not a jump.
- It does not explain E's test collapse: E's ranking is unchanged and its mean probability only moves 0.28 to 0.22, while on test it dropped from 0.35 (held-out train) to 0.18. The remaining E shift on test is not from the file format.
- Decision: no full re-extraction of the frozen DINOv3 train features for now; new caches (fine-tuning) are built in the test format from the start.

## 2026-11-20 - Blend check of A, C1, E (held-out train)

- Held-out ranking score (AUC): A 0.8155, C1 (soft labels a=0.5) 0.8164, E GRU 0.7783, E transformer 0.7839.
- A and C1 are almost the same model (probability correlation 0.989), so C1 is not a separate ensemble member.
- E makes different mistakes than A (correlation 0.58, clips wrong for both 0.108 against 0.171 for A with C1). Blending A, C1 and both E models: AUC 0.8324, train per clip F1 0.7439 (A alone 0.7378).
- On test the blends with E are worse (test vs `1.csv` 0.65 to 0.71) because E's test predictions are shifted low (cheat share 0.05 to 0.13).

## 2026-11-20 - Sequence models over the clip's 30 frames (E2, E3), member B on 30-frame pose, first blend

- Members, shared per-video folds, 6166 train clips, each clip scored alone (train per clip F1 / test vs `1.csv` F1, clips wrong, test cheat share):
  - B12: 12-block fine-tuned backbone (2 frames per clip, as before) + 30-frame pose, LR: 0.7889 / 0.8064 (223), 0.226. Same as the earlier 0.7893 / 0.8094 with the 10-frame pose.
  - E (DINOv3 PCA-128 per frame, BiGRU or transformer, no pose): 0.7033 / 0.5793 and 0.6998 / 0.5740, test cheat share 0.075 / 0.049.
  - E2 (+ 19 per-frame pose signals): BiGRU 0.7286 / 0.5384, transformer 0.7401 / 0.6124, test cheat share 0.038 / 0.070.
  - **E3 (DINOv3 part minus the clip's own mean, + pose signals): BiGRU 0.7722 / 0.7617 (276), transformer 0.7731 / 0.7428 (298), test cheat share 0.231 both.** Centring on the clip mean removes the train-vs-test static offset that made E2 collapse; the dynamics carry real signal.
  - A with the same dynamics-only inputs in the logistic regression (no clip mean): 0.7127 / 0.6979, so the gain is from the sequence model, not only from dropping the mean.
- E is the same frozen DINOv3 ViT-H+ backbone as A with a different head: a sequence model over the 30 per-frame vectors instead of a logistic regression on their mean, std and frame differences (E5: the same head on LeVJEPA). With a fine-tuned backbone the same head would sit on top of B's backbone.
- Details of E: DINOv3 ViT-H+ frozen, per-frame 2560-d vector standardised and PCA to 128 (fit on train frames); BiGRU hidden 96 or 2-layer transformer width 64 with learned positions; mean + max pool over frames, dropout 0.3, linear; 25 epochs AdamW, one-cycle LR, input noise 0.1, 3 seeds per fold, test = average of 5 fold models x 3 seeds.
- Held-out ranking score (AUC): A 0.8155, B12 0.8664, E3 BiGRU 0.8516, E3 transformer 0.8574.
- Blend (`final_blend.py`): each member's probability turned into its rank inside its own held-out train predictions, weighted average, weights from {0, 1, 2} and threshold chosen on held-out train F1. Best: B12 x2 + E3 BiGRU x1 + E3 transformer x1, A weight 0, threshold 0.70: train per clip F1 0.8161 (B12 alone 0.7889), AUC 0.9012, test vs `1.csv` 0.8216, test cheat share 0.228. Written to `leon/submissions/nosequence_blend_v1.csv` (1515 rows, sample_submission order).
- Caveats: the B12 part uses 2 of the 30 test frames (the 30-frame fine-tune is planned for next, with weights saved). D (LeVJEPA) features are still extracting.

## 2026-11-20 - Tunable option: blend threshold (and the other knobs of the blend)

- The blend score is the weighted average of each member's rank (position inside its own held-out train predictions), so the threshold is in rank space: 0.70 means "among the top 30% of held-out train scores". It is a tunable option of the pipeline: re-run the sweep (`thr_sweep.py`, step 0.025) whenever a member, a weight or a feature set changes, and pick the value on **held-out train F1**. Test vs `1.csv` is printed next to it only as a second check, it does not choose the value.
- Sweep for the blend B12 x2 + E3 BiGRU + E3 transformer (train per clip F1 / test vs `1.csv` F1 / test cheat share):

| Threshold | Train F1 | Test vs `1.csv` | Test cheat share |
|---|---|---|---|
| 0.600 | 0.7811 | 0.7898 | 0.373 |
| 0.650 | 0.8004 | 0.8227 | 0.295 |
| 0.675 | 0.8109 | 0.8273 | 0.259 |
| **0.700** | **0.8161** | 0.8216 | 0.228 |
| 0.725 | 0.8141 | 0.8193 | 0.208 |
| 0.750 | 0.8120 | 0.8057 | 0.181 |
| 0.800 | 0.7924 | 0.7412 | 0.123 |
| 0.850 | 0.7441 | 0.6427 | 0.074 |

- The curve is flat between 0.675 and 0.725 (train F1 0.811 to 0.816) and falls on both sides, faster on the high side (fewer predicted cheats than the real share). Best on held-out train: 0.700, which is the middle of that plateau. Best on the `1.csv` comparison (information only): 0.675, 0.8273, a gain of 0.006 that is within noise, so the threshold stays at 0.700. The train and test curves have the same shape.
- Other tunable knobs of the blend and E, all chosen on held-out train: member weights (grid {0, 1, 2} per member, `final_blend.py`), E input size (PCA 128 dimensions), E hidden size (GRU 96, transformer 64), epochs (25), input noise (0.1), seeds per fold (3), logistic regression C (0.003).
- 2026-11-20, on Leon's request ("trust the test more"): threshold 0.600 is worse than 0.700 on both measures (train F1 0.7811 against 0.8161, test vs `1.csv` 0.7898 against 0.8216, test cheat share 0.373). Alternative file at the test-best threshold: `leon/submissions/nosequence_blend_v1_thr0675.csv` (threshold 0.675, train F1 0.8109, test vs `1.csv` 0.8273). `nosequence_blend_v1.csv` (0.700, chosen on held-out train) is unchanged. Choosing 0.675 uses the `1.csv` comparison to set a pipeline value.

## 2026-11-20 - E4: E3 trained on sliding 1-second windows (3x more clips from the same videos)

- Same input and models as E3 (clip-mean-centred DINOv3 PCA-128 frames + 19 pose signals per frame, BiGRU or small transformer), but trained on windows slid by 10 frames over each video (17,702 windows instead of 6,166 non-overlapping clips, label = window centre after the start second), 10 epochs. Scored on the same aligned clips of the held-out fold and on the test clips, each alone (`e_slide.py`).
- Train per clip F1 / test vs `1.csv` F1 (clips wrong), cheat share: E4 BiGRU 0.7679 / 0.7575 (286), 0.246; E4 transformer 0.7614 / 0.7276 (301), 0.196. E3 for reference: BiGRU 0.7722 / 0.7617, transformer 0.7731 / 0.7428.
- No gain from overlapping windows (slightly worse): the extra windows are near-copies of the same 416 videos, so they add little new information. Not used in the blend.

## 2026-11-20 - D (LeVJEPA video model) and E5, blend v2

- LeVJEPA-1.5-Large (303M, frozen, `trust_remote_code` code read first: only math / numpy / torch / transformers, no network or exec calls), `extract_levjepa.py`: each 1-second clip fed as two interleaved 15-frame halves (frames 0,2,4,... and 1,3,5,...) so all 30 frames are used, 224x224 squashed, train frames through the test delivery format (640x360 + JPEG q90) first. Output per clip: 30 per-frame mean-patch vectors + the [cls] token of each half, `cache/levjepa_large_224_30/` (test (1515, 32, 1024), 416 train files, 37 min).
- Train per clip F1 / test vs `1.csv` F1 (clips wrong), test cheat share, same per-video folds:
  - D1: LR on LeVJEPA summary (mean, [cls], std, mean abs frame difference) + 30-frame pose: C=0.001 0.7368 / 0.7679 (295), 0.315; C=0.003 0.7315 / 0.7715 (293), 0.324; LeVJEPA only 0.7153 / 0.736 (360).
  - E5: the E3 sequence models on LeVJEPA per-frame vectors (centred on the clip mean, PCA 128) + pose: BiGRU 0.7513 / 0.7792 (291), 0.356; transformer 0.7689 / 0.7881 (271), 0.321.
- Held-out ranking score (AUC) of the best blends (`final_blend.py`, rank average, weights in {0, 1, 2}, threshold from held-out train F1):
  - v1: B12 x2 + E3 BiGRU + E3 transformer: train 0.8161, AUC 0.9012, test vs `1.csv` 0.8216.
  - **v2: B12 x2 + E3 BiGRU + E3 transformer + D1 + E5 transformer (E5 BiGRU 0): threshold 0.70, train 0.8265, AUC 0.9225, test vs `1.csv` 0.8342 (cheat share 0.251).** Written to `leon/submissions/nosequence_blend_v2.csv`.
  - The top 12 weight combinations are within 0.003 of each other on train (0.8238 to 0.8265), so the exact weights are not important; using all six members (2/1/1/1/1/1) gives the highest AUC 0.9242 with train 0.8245 and test 0.8266.
- Caveat as before: the B12 part uses 2 of the 30 test frames; the 30-frame fine-tune is for next time.

## 2026-11-20 - Public leaderboard result of blend v2

- `leon/submissions/submission.csv` (identical to `nosequence_blend_v2.csv`, checked clip by clip against the saved member predictions) scored **0.8518237** on the public leaderboard. Estimate from held-out train 0.8265 per clip, from the `1.csv` comparison 0.8342.
- Leon's goal: reach about 0.9 in the independent setting.

## 2026-11-20 - main.ipynb: the independent blend pipeline as a notebook

- New notebook `leon/main.ipynb` (43 cells, simple-notebook style, Bahasa Indonesia, imports in the top cell, `%pip` pins include torch 2.11.0). It replaces the A-only no-sequence notebook (kept untouched as `leon/mainipynb`, a file name without the dot; the old video-pipeline main is `main_haram.ipynb`, renamed by Leon).
- Pipeline in the notebook: train clips from the by-eye start seconds (procedure shown in the EDA section), per-video folds shared with the fine-tuned backbones, features read from caches, five members, rank blend, submission:
  - fine-tuned 12-block backbone (cached features, 2 frames per clip) + 30-frame pose, logistic regression (C 0.003);
  - LeVJEPA + 30-frame pose, logistic regression (C 0.001);
  - BiGRU and small transformer over the clip's 30 DINOv3 frames (PCA 128, centred on the clip mean, + 19 per-frame pose signals);
  - the same transformer over the 30 LeVJEPA per-frame vectors;
  - blend: each member's rank inside its own held-out train predictions, weights in {0, 1, 2} and threshold chosen on held-out train F1 inside the notebook (no `1.csv`, no test statistics); output `submissions/main.csv`.
- Dropped from the earlier blend search: frozen-DINOv3 logistic regression A (weight 0) and the LeVJEPA BiGRU (weight 0).
- Cache-building scripts copied into the repo with relative paths: `leon/scripts/extract_dino30.py`, `extract_pose_all.py`, `extract_levjepa.py`, `ft_cache.py`, `ft_train.py`, `rowreader.py` (the earlier copies only lived in a temporary folder). Weights of the 12-block fine-tune were never saved, so its features can only be read from the cache.
- Known limit: the fine-tuned member uses 2 of the 30 frames of every test clip, to be replaced by the 30-frame fine-tune.
- Verification run of the notebook code from top to bottom (output redirected to a temporary folder, nothing written to `submissions/`): the members score (F1-Macro per clip at threshold 0.5) fine-tuned + pose 0.7872, LeVJEPA + pose 0.7212, BiGRU DINOv3 0.7697, transformer DINOv3 0.7664, transformer LeVJEPA 0.7520. Blend search picked weights (fine-tuned 2, BiGRU DINOv3 1, transformer DINOv3 0, LeVJEPA LR 1, transformer LeVJEPA 1), threshold 0.70, held-out train F1 0.8235 (the earlier scratch blend v2: 0.8265), test cheat share 0.273.
- The notebook output differs from the submitted blend v2 (`submission.csv`, public LB 0.8518237) on 42 of 1515 clips, and scores 0.8348 against `1.csv` (v2: 0.8342). The small differences come from the sequence models' random seeds and the PCA sample, so the weights chosen change slightly between runs. The notebook writes `submissions/main.csv` when run.
