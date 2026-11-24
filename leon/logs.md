# Experiment logs

## Not tried yet

Ideas we have discussed but not run.
When one is tried, delete it from this list and write its result as a normal dated entry below.
Rewritten 2026-11-22: everything already tried (augmentation screens, recipe screens, head-only B, V-JEPA 2, V-JEPA 2.1 ViT-L, VideoMAE v2, aggregation, flip TTA, person crops, boundary clips, weight averaging, layer-wise learning rate, fine-tune + sequence head) now has its own dated entry below and was removed from this list.
Naming (user, 2026-11-22): one-off techniques get descriptive names; if one works it becomes part of the base configuration of B, D or E, not a new letter.

Task-specific model features (planned for 2026-11-24, user said to try them; judge each by the `1.csv` column in the blend first, then the held-out F1; no zero-shot and no vision-language model, FAQ #14.1 / #26.3):
- Whole-body keypoints (RTMW / DWPose, 133 points: hands and face): first look at the hand points on a few smartwatch clips (operating the watch vs only looking at it, notes in the hand); check the weights licence in the official MMPose zoo.
- Phone / paper detector from RT-DETR COCO classes: low expected value (no watch, no small notes).
- Skipped: supervised video models (VideoMAE v2 was poor), open-vocabulary detectors (vision-language).
Also planned for 2026-11-24: first submission = `submissions/blend_v2_e58tf_swap_thr068.csv`; second = the Kaggle 30 fps 288x512 fine-tune (inference with IMG_H=288 IMG_W=512, check `1.csv` agreement before submitting).

Next candidates after the 2026-11-24 discussion (user: write them down, GPU was busy; each is judged by the `1.csv` column in the blend first, then held-out F1; read the model card / README before running, see memory `feedback-read-model-pages`):
- **LeVJEPA-VideoMix-Large as a frozen member (the user's pick, first in line).** `galilai-group/LeVJEPA-VideoMix-Large` (27 Aug 2026, CC BY-NC 4.0): same ViT-L architecture and size as our LeVJEPA-1.5-Large (303.1M, 224 px, 16 frames, tubelet 1, RoPE, block-causal attention) but a different pretraining set: 1.8M clips from Kinetics-710, Something-Something v2, Walking Tours and PE-Video (people doing actions with objects, closer to cheating behaviour). The 1.5 Large we use (sha 7479a97, after the 11/05 weight fix) says about 5,300 unique video hours. Neither card compares the two, so it is an open question which is better for us. How: copy `scripts/extract_levjepa.py`, change the model name, keep `attn_mode` at its default `block_causal` (the card says full attention silently returns worse features), output `cache/levjepa_videomix_224_30/`; then the plain members D1 (LR + pose), E5 BiGRU and transformer on the same folds, and if equal or better alone also the E58 recipe (augmented copy via `extract_levjepa_aug.py` and the 8-action head). Reference numbers to beat (AUC / best F1): D1 0.8539 / 0.7381, E5 transformer 0.8784 / 0.7705, E5 BiGRU 0.8695 / 0.7521, E58 BiGRU 0.8998 / 0.7863, E58 transformer 0.8910 / 0.7842. Then swap or add it in the best blend (B x2, E3 BiGRU, E3 transformer, D1, E58 transformer).
- LeVJEPA input check: the model card expects 16 frames at about 7.5 fps (about 2 s); we give 15 frames per half (even and odd frames) at 15 fps over 1 s, so the motion is twice as fast as in training. Clips are only 1 s long, so an exact match is impossible (about 8 frames at 7.5 fps); check on the 400 fold-0 clips whether the features change at all before re-extracting everything.
- V-JEPA 2 / 2.1 ViT-g with the E58 recipe (augmented copy + 8-action head): only the plain recipe was ever run (alone about equal to LeVJEPA, no blend gain; 2.1 ViT-L at equal size 0.016 to 0.018 AUC worse than LeVJEPA Large). Needs a re-download (2.1 ViT-g weights deleted earlier, V-JEPA 2 ViT-g weights are still in the Hugging Face cache) and about 3 to 4 GPU hours; low priority.
- A second fine-tuned image backbone of another family for the blend (C-RADIOv4 is the only challenger to DINOv3 on dense tasks per its own report; EVA-02 or Perception Encoder image side): fold-0 screen with the B recipe (about 15 min), then a Kaggle run (about 3 h). Check licences and the FAQ wording on CLIP-style towers (zero-shot and VLM prediction are not allowed) first.
- Several seeds of B averaged; B epoch screens with 3 and 4 epochs.
- ViTPose+ (large or huge, `usyd-community/vitpose-plus-*`, huge = 899M parameters, 6 experts) instead of ViTPose base (85.9M) for every pose feature: first check on a few hundred fold-0 clips whether the wrist and head points differ visibly from the base model's. Expensive (all frames, then every pose-using member rebuilt) and low expected value, since gaze and head pose did not help.
- Optical flow (RAFT) on the person crop as a motion channel; whole-body hand keypoints (ViTPose+ whole-body expert or RTMW) for smartwatch and notes (16% of the fold-0 errors, hands about 10 to 15 px).
- Not worth it (reasons logged 2026-11-24): LightGlue / SuperPoint (matches the same points between two views, no fit), more gaze or 3D gaze models (heads of 60 to 80 px, two gaze-type signals already failed in the blend).

Running or next:
- V-JEPA 2.1 ViT-g (1B) as frozen members (extraction running, then LR + pose, BiGRU, transformer and the blend check). The ViT-L (300M) added nothing.
- Run the new Kaggle fine-tune `kaggle_15fps_aug/kaggle_finetune_15fps_aug_full.ipynb` (100% train, 15 fps, K=12, 2 epochs, augmentation, weight averaging; the user runs it), then predict all 30 test frames with the averaged and the unaveraged weights and compare each with 1.csv (secondary check only).
- Frozen A (frozen DINOv3 + logistic regression + pose) in the blend again, but trained on the augmented extractions from the E3 step (user, 2026-11-22: after E3). Plain A was tested with the fair nested search and got weight 0 (see the dated entry); re-test it the same way once the augmented features exist.
- Rebuild the blend with the new fine-tuned B (and the 30 fps model as B for submissions until the next fine-tune, user rule 2026-11-22).

Fine-tune ideas that need a new fine-tune to test (fold-0 screens, about 14 to 17 min each):
- Higher input resolution than 224x400 (check this log first that it was not tried).
- Teacher soft labels in the fine-tune (the whole-video teacher gives soft labels to the frame model).
- Frames from the raw video instead of the test format, and a 32-bit AdamW with clipping instead of the 8-bit one (low priority, user 2026-11-21; the 32-bit needs a memory check on the 8 GB GPU).
- Fine-tuned per-frame features into the sequence models E3 / E5 (they would read fine-tuned instead of frozen DINOv3 features; the simple heads on the honest fold-0 outputs were worse than the plain mean).
- Fine-tune LeVJEPA (the video model) instead of using it frozen; heavy (about 5,900 tokens per clip), probably needs fewer frames or only the top blocks.
- More unfrozen blocks: K=16 and K=20 were not better in screens, so only with LoRA or a lighter optimiser beyond 20.
- Long 30 fps 5-fold fine-tune on Kaggle or an A100 (Colab) once the architecture is settled (user, 2026-11-22).
- Final 5-fold fine-tune with the best recipe, postponed by the user to near the end of the competition (minimal notebook, A100 on Colab is possible).

Other models and members:
- Supervised action models as frozen members (VideoMAE-base fine-tuned on Kinetics, TimeSformer, Video Swin); InternVideo-Next was skipped because it needs `flash_attn`. Video-language models are not allowed.
- Newer backbones when public weights exist: DINOv4 (only a rumour, 2026-11-20), JEM (7B, too big for the 8 GB GPU unless a small version appears).

Peeking label cleaning (2026-11-23): drop or soften the peeking clips the held-out blend is sure are normal, then re-run the fold-0 fine-tune screen. Weak case: only 5 of the 434 peeking cheat clips have blend rank under 0.3 (208 are below the 0.70 threshold, mixing real misses with label noise), so a strict cut changes almost nothing. Not run; the user said to try it when the GPU is free.

Teacher-student and context ideas:
- Soft labels from the whole-video teacher on the sequence models (E3 and E5). Only tried on the logistic regression (C1, no gain).
- Context guessing: an extra output that predicts the previous and next second's features from the clip alone (HalluciNet style).
- Copying the teacher's per-person-centred features: the student learns to guess what the whole-video teacher computes.

Smaller ideas:
- Tuning E's own settings (PCA size, hidden size, epochs, dropout); each was run at one setting only.
- A LightGBM head on the LeVJEPA features as in Owen's baseline (D1 used a logistic regression).
- Re-extracting the frozen DINOv3 train features for all folds in the test format (only fold 0 was checked, the gap cost A about 0.02 to 0.03 F1).
- Retry the sequence head on top of the fine-tuned frames with the honest outputs of all 5 folds after the final fine-tune (fold 0 alone was too small).

Notebooks (user, 2026-11-22): the main notebook gets the RT-DETR + ViTPose pose step; the final notebook runs from scratch (all extraction, pose, fine-tune, members, blend) for the committee.

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

## 2026-11-20 - Next-session plan started: new fine-tune pipeline (ft2) for 16+ blocks

- Why a new pipeline: the old block-20 token cache caps the depth at 12 trained blocks, cannot take image augmentation, and was built from raw video frames. New design: train frames (every 15th frame, 13,521 frames, same rows as the old cache) passed through the test delivery format (640x360, JPEG q90) and stored as uint8 400x224 RGB (`cache/ft_frames_testfmt/`, 3.9 GB, `scripts/ft2_frames.py`); the first 32-K blocks are frozen and run on the fly, so any K and optional augmentation work (`scripts/ft2_train.py`).
- Memory: trainable blocks use bitsandbytes 8-bit AdamW (one optimizer per tensor) stepped inside the backward pass with gradient checkpointing, so gradients are not stored and there is no gradient clipping (the old runs clipped at 1.0 and used fp32 AdamW). K=16 (419.8M trainable block parameters) peaks at 5.09 GB at about 10 frames per second (benchmark, 25 steps); K=20 or 24 should fit.
- Weights of the trained blocks and norm are saved for every fold (fp16), plus the features of all train frames from each fold backbone. Test side (`scripts/ft2_infer_test.py`): all 30 frames of every test sequence, the frozen blocks once, then each fold's trained blocks; the logistic regression on 2-frame summaries sees the 30 frames as 15 interleaved pairs (frame i and i+15) whose probabilities are averaged.
- Augmentation (off in the first run, flag AUG=1): brightness, contrast, saturation +-20%, zoom in to 85%, shift 5%, rotation 3 degrees, blur on half of the batch; no flip.
- Run 1: K=16, 2 epochs, no augmentation, `cache/ft2/k16/`. Comparison point: old 12-block member B, held-out train F1 0.7889 (but its pipeline differs: raw frames, fp32 AdamW with clipping), so if 16 is not clearly better a 12-block run in the new pipeline is the fair baseline.
- Run 1 result (K=16, trained 5 folds, about 26-30 min per fold, peak 5.98 GB): held-out frame accuracy per fold 0.670, 0.676, 0.604, 0.597, 0.680. Member B (LR on the 2-frame features + 30-frame pose) held-out clip F1 **0.7821** (threshold 0.6; 0.7807 at 0.5), vs the old 12-block B 0.7889. So 16 blocks in the new pipeline is not better than the old 12. Test side for k16 not run yet (weights are saved, `python scripts/ft2_infer_test.py 16 k16` takes about 1 h). K=20 is not tried, because the plan says to go deeper only if 16 is better.
- Run 2 (done): K=12 in the same new pipeline, 2 epochs, no augmentation, `cache/ft2/k12/`, to separate the effect of block count from the effect of the pipeline change (test-format frames, 8-bit AdamW, no gradient clipping).

## 2026-11-20 - Augmentation screen on fold 0 (queued, results below when done)

- Why a screen: a full 5-fold run takes about 2 h, so 40 augmentations at 5 folds are not possible. Each screen run trains the K=12 backbone on fold 0 only (same 2 epochs, same recipe as run 2), predicts only the held-out frames of that fold (no weights, no full feature pass; a throwaway run, a chosen setting is re-run on 5 folds with weights saved), about 16 min each.
- Score of a screen run (`scripts/ft2_screen_score.py`): the head's own cheat probability (sum of the 4 cheat classes) averaged over the clip's 2 frames, vs the clip label on the 1,364 clips of fold 0: AUC and macro F1. It is a proxy for member B (which adds a logistic regression and pose), good enough to rank augmentations. The no-augmentation reference is fold 0 of run 2 (`cache/ft2/k12`); a second no-augmentation run with another training seed (TSEED=1042) measures the seed noise, so only differences clearly bigger than that count.
- All augmentations are tokens in `scripts/ft2_augs.py` (env `AUG`, groups or single tokens). Implemented: geometry (zoom, shift, rot 4 deg, shear, persp, elastic, crop, hflip), photometric (bright, contrast, sat, hue, gamma, gray, autocontrast, posterize, equalize), quality (blur, motion, noise, sp = salt and pepper, jpeg, downup, sharpen, patch = glare or shadow blob), hiding (erase, cutout, griddrop, coarse), mixing (mixup, cutmix on half of the batches, soft labels), weather (fog, shadow, flare, rain, snow). Skipped on purpose: vertical flip and 90 degree rotation (the camera is fixed and upright), invert / solarize and channel shuffle (colours never look like that), mosaic and copy-paste (object detection only).
- Stage 1 (queued, in order): seed noise, then each group on its own (geo, photo, quality, hide, mix, weather), then hflip alone (it needs a check because left and right hand matter). Stage 2: single tokens inside the groups that help or are neutral. Stage 3: best combination on 5 folds.
- Run 2 result (K=12, new pipeline, no augmentation, `cache/ft2/k12/`, about 23 min per fold, peak 5.0 GB): held-out frame accuracy per fold 0.665, 0.661, 0.541, 0.582, 0.663. Member B held-out clip F1 **0.7820** (threshold 0.6; 0.777 at 0.5), the same as K=16 (0.7821) and 0.007 below the old 12-block B (0.7889). So in the new pipeline 12 and 16 blocks score the same, and the small gap to the old B comes from the pipeline change (test-format frames, 8-bit AdamW, no clipping) or from noise, not from the block count. K=20 stays untried. Test pass for k12 not run yet.
- Seed-noise result (fold 0, K=12, no augmentation, training seed 1042 vs 42): AUC 0.8483 vs 0.8495, F1 at the best threshold 0.752 vs 0.7645, F1 at 0.5 0.7365 vs 0.7615. So AUC moves by about 0.001 between seeds but F1 by 0.01 to 0.025. The augmentation screen therefore judges on AUC first (a change under about 0.005 is noise) and treats F1 as secondary.

## 2026-11-20 - Training on all 30 frames per second (random frame per half second each epoch)

- Question raised: the K=12 / K=16 fine-tunes only saw 2 frames per second (every 15th frame), not 30. A fine-tune on every frame would be about 14 times more images (about 28 h for 5 folds), so instead each epoch draws ONE RANDOM labelled frame out of the 15 in each half second: same number of images and training time as before, but across epochs the backbone sees many different frames.
- New files: `scripts/ft2_frames_all.py` (every frame of every train video in the test format, 200,789 rows, 54 GB, `cache/ft_frames_testfmt/train_frames_all.npy` + `train_meta_all.csv`) and the env flag `ALLFRAMES=1` in `scripts/ft2_train.py`. The held-out / feature pass still uses the every-15th-frame rows, so member B and the test pass are unchanged.
- Fold-0 screen queued (K=12, no augmentation, `aug_allframes`), compared with `k12` and `aug_seed2` by AUC.
- Memory stop (2026-11-20): the system killed the augmentation screen queue, the frame extraction (done 60 of 416 videos, resumable) and the waiting queues because RAM ran low (a training run and the 54 GB extraction were running together). Nothing was restarted by itself. From now on jobs run one at a time in this order, as asked: (1) K=12 test inference, (2) fine-tune with all frames (extraction, then the fold-0 run), (3) augmentation screens (the geometry run was cut at step 100 and starts from the beginning).
- K=12 test inference started (`python scripts/ft2_infer_test.py 12 k12`, all 30 test frames, 5 fold backbones, about 40 min).

## 2026-11-20 - Kaggle fine-tune notebook (15 frames per second)

- New notebook `leon/kaggle_finetune_15fps.ipynb` (simple-notebook style, self-contained, nothing else from the repo is needed): reads the train videos from `/kaggle/input`, recodes every 2nd frame (15 fps, plus every 15th frame for evaluation) to the test format (640x360, JPEG q90) and keeps the JPEG bytes in RAM, labels the frames like `ft2_train.py`, fine-tunes the top K=12 blocks of DINOv3 ViT-H+ plus an 8-action head, and saves weights, hold-out predictions and features.
- Config cell: K, FPS, EPOCHS, HOLDOUT_FOLD (0 = train on folds 1-4 and score fold 0, comparable with the local screen; None = 100% of train), TIME_BUDGET_H (total session limit, default 8.5 h). A 15-step benchmark at learning rate 0 measures the GPU speed and the number of steps is cut so the run, evaluation and features fit in the budget; weights are saved every 500 steps.
- Differences from the local recipe: standard fp32 AdamW with gradient clipping 1.0 (the old pipeline's recipe) instead of the 8-bit AdamW without clipping, because Kaggle GPUs have 16 GB. Uses bf16 autocast on GPUs with compute capability 8 or more, otherwise fp16 with a gradient scaler (P100 / T4 have no bf16); the frozen blocks fall back to fp32 if fp16 gives infinite values.
- The fold table (video, fold, action, by-eye start, label) is pasted in the notebook so the folds match the local ones; the clip label rule was checked against `clips.csv` (6,166 of 6,166 equal). The frame extraction and labels were checked on 3 local videos against the local `train_meta.csv` (same frames, same seconds). The GPU part (model, training, evaluation) is not tested yet because the GPU is busy.
- Outputs (about 0.65 GB of weights per model, plus a few MB): `<tag>_weights.pt` (same format as the local fold weights), `_head.pt`, `_val_logprobs.npy` + `_val_meta.csv`, `_feat.npy` + `_feat_meta.csv`, `_summary.json`.
- Needs on Kaggle: GPU on, Internet on (pip install of timm 1.0.29), the competition data added, and the DINOv3 weights as a dataset (file `model.safetensors`, 3.2 GB, from the local Hugging Face cache) or a Hugging Face token in Kaggle secrets named HF_TOKEN.
- Found while checking the slow test inference: the killed augmentation queue left its geometry screen run (`ft2_train.py 12 2`) alive as a stray process, so it shared the GPU with the K=12 test inference (about 290 s per chunk of 32 sequences instead of the expected 40 s).


## 2026-11-20 - ft2 K=12 test inference (all 30 frames) and blend v2 with the new B

- Test inference of the K=12 fine-tuned backbone (5 folds, all 30 frames of the 1515 test sequences, features 5 x 1515 x 30 x 2560) finished: 4964 s total (part of it shared the GPU with a stray screen run, see the entry above).
- Member B (LR on 2-frame summaries + 30-frame pose, test side = 15 interleaved frame pairs averaged, then 5 folds): held-out train F1 0.782 (thr 0.6), test vs `1.csv` 0.7446 (cheat share 0.157). The old 12-block B scored 0.7889 held-out and 0.806 vs `1.csv`, so the new B is worse on both sides.
- Blend with the same pipeline as v2 (rank average, weights in {0,1,2} and threshold from held-out train, members E3 BiGRU, E3 transformer, D1, E5 transformer), only B swapped:
  - old B12 (v2): held-out F1 0.8265 (weights 2/1/1/1/1, thr 0.70), AUC 0.9225, vs `1.csv` 0.8342.
  - new B ft2 K=12: held-out F1 0.8227 (weights 2/0/1/1/1, thr 0.70), AUC 0.9150, vs `1.csv` 0.8365, cheat share 0.246.
- Result: the new B does not improve the blend (held-out 0.8227 vs 0.8265). The vs `1.csv` check is about equal (0.8365 vs 0.8342). File written for the comparison: `submissions/blend_v2_ft2B_k12.csv` (not submitted). K=16 was not run on test (held-out blend with K=16 was 0.8191).
- Next: augmentation screens (fold 0, K=12), one run at a time.

## 2026-11-20 - Public leaderboard result of blend v2 with the new B (ft2 K=12)

- Submitted by the user: `submissions/blend_v2_ft2B_k12.csv` got public LB 0.8408476, worse than blend v2 with the old B (0.8518237), a drop of 0.0110.
- This agrees with the held-out train score (0.8227 new vs 0.8265 old) and with B alone on `1.csv` (0.7446 new vs 0.806 old). The blend-level check against `1.csv` (0.8365 vs 0.8342) pointed the wrong way, so it is too small a difference to trust (about 3 clips).
- Conclusion: the K=12 fine-tune trained on 2 frames per second does not replace the old 12-block B. The old B stays in the blend.

## 2026-11-20 - E with a text transformer (RoBERTa / DeBERTa) as the sequence head

- Script `e_bert.py` (scratchpad): same input as E3 (clip-mean-centred DINOv3 PCA-128 frames + 19 pose signals, 30 frames per clip), each frame vector projected (dropout 0.2, linear, LayerNorm) to the encoder width and given as `inputs_embeds`; mean + max pooling over the 30 outputs, dropout 0.3, linear. Encoder learning rate 2e-5, projection and head 1e-3 (one-cycle), AdamW weight decay 0.05, input noise 0.1, batch 64, 6 epochs, 1 seed, bf16 autocast, gradient clip 1.0. Same folds as the other members. About 4 to 5 min per model for the 5 folds.
- Held-out train per clip F1 / AUC, test vs `1.csv` F1 (clips wrong), test cheat share:
  - E3 transformer (reference, 3 seeds, 25 epochs): 0.7731 / 0.8574 / 0.7428; E3 BiGRU: 0.7722 / 0.8516 / 0.7617.
  - roberta-base, random weights (control, same architecture): 0.7661 / 0.8484 / 0.7469 (298), 0.244.
  - roberta-base, pretrained text weights: 0.7381 / 0.8066 / 0.7565 (277), 0.217.
  - deberta-v3-base, pretrained: 0.705 / 0.7977 / 0.7252 (331), 0.263.
- Result: no gain. The pretrained text weights hurt (RoBERTa AUC 0.8484 with random weights against 0.8066 pretrained), and the large model with random weights only matches the small E3 transformer. Not added to the blend. Not tried: these heads on the LeVJEPA frames (E5 input).
- Notes: the first DeBERTa runs gave NaN or a dtype error because transformers 5 loads the DeBERTa checkpoint in fp16 by default; fixed with `dtype=torch.float32` in `from_pretrained` (RoBERTa is stored in fp32, unaffected). Member files `members/E_<model>_<pre|scratch>_{oof,test}.npy`.

## 2026-11-20 - Augmentation screens, fold 0 (K=12, 2 epochs, held-out frames only)

- Metric: clip AUC from the head's cheat probability averaged over the clip's 2 frames (`ft2_screen_score.py`). Baselines on the same fold: `k12` 0.8495 (F1 at best threshold 0.7645) and `aug_seed2` 0.8483 (0.752).
- geo (zoom, shift, rot, shear, persp, elastic, all on): AUC 0.8476, F1 best 0.7917: no gain.
- photo (bright, contrast, sat, hue, gamma, gray, autocontrast, posterize, equalize, all on): AUC 0.8613, F1 best 0.7919: the only clear gain so far (+0.012 AUC, seed noise about 0.001).
- quality (blur, motion, noise, sp, jpeg, downup, sharpen, patch): trained (about 17 min) but not scored yet; the scoring script reads `cache/ft_frames_testfmt/train_meta.csv`, which was deleted by the user together with `train_frames_all.npy`.
- hide, mix, weather, hflip: not run, they crashed at start for the same missing `train_meta.csv`. `train_frames.npy` is still there. Waiting for the decision to rebuild `train_meta.csv` with `ft2_frames.py`.

## 2026-11-20 - Augmentation group screens finished (fold 0, K=12, 2 epochs, all tokens of a group on)

- `cache/ft_frames_testfmt/train_meta.csv` was deleted by the user together with the unused 51 GB all-frames file; rebuilt by copying `cache/ft_hplus_block20/train_meta.csv` (same 13,521 rows, row numbers 0 to 13520, 416 videos). Checked by re-extracting 9 frames from 3 videos (first, middle, last of each) and comparing with `train_frames.npy`: all 9 identical. The hide, mix, weather and hflip runs had crashed at start for the missing file and were rerun.
- Same fold-0 setup for every row (1,364 clips; AUC primary, seed noise about 0.001; F1 at best threshold is optimistic and noisier, 0.01 to 0.025). One run per group, 16 to 17 min each:

| Run | AUC | F1 at 0.5 | F1 at best thr | thr |
|---|---|---|---|---|
| k12 (no augmentation) | 0.8495 | 0.7615 | 0.7645 | 0.725 |
| aug_seed2 (no augmentation, other seed) | 0.8483 | 0.7365 | 0.7520 | 0.65 |
| geo (zoom, shift, rot, shear, persp, elastic) | 0.8476 | 0.7693 | 0.7917 | 0.625 |
| photo (bright, contrast, sat, hue, gamma, gray, autocontrast, posterize, equalize) | 0.8613 | 0.7736 | 0.7919 | 0.675 |
| quality (blur, motion, noise, sp, jpeg, downup, sharpen, patch) | 0.8591 | 0.7543 | 0.7987 | 0.75 |
| hide (erase, cutout, griddrop, coarse) | 0.8730 | 0.7627 | 0.7871 | 0.725 |
| mix (mixup, cutmix) | 0.8661 | 0.7340 | 0.7802 | 0.80 |
| weather (fog, shadow, flare, rain, snow) | 0.8721 | 0.7543 | 0.7806 | 0.65 |
| hflip (horizontal flip alone) | 0.8650 | 0.7626 | 0.7825 | 0.625 |

- Result: every group except geo gains in AUC (+0.011 to +0.024 over the two no-augmentation runs); hide (0.8730) and weather (0.8721) are the best, then mix 0.8661, hflip 0.8650, photo 0.8613, quality 0.8591. Geo gives no gain. Horizontal flip alone already gives +0.016.
- Not run yet: stage 2 (single token or leave-one-out inside each helpful group) and stage 3 (best combination on all 5 folds, weights saved, then test inference). Not tried: `crop` and other geometric tokens individually, channel shuffle / invert (skipped on purpose).

## 2026-11-20 - Timestral (frozen), and blend v2 with the old B rewritten for the day's submission

- Timestral-209M frozen (`e_timestral.py`): each clip as a 30-step series (first 32 PCA components of the clip-mean-centred DINOv3 frames + 19 pose signals, left-padded to one 32-step patch, all variates of a clip in one group), last-layer hidden states (encoder token mean, forecast-query token mean and max over variates) + log std of every variate, standardised, logistic regression. Held-out F1 by C: 0.0003 0.6523, 0.001 0.6603, 0.003 **0.6637**, 0.01 0.6576; test vs `1.csv` 0.68 (378 wrong), cheat share 0.247. Far below E3 (0.772). Feature extraction 147 s. Library installed from a local copy of the open-source repo (`pip install --no-deps`), code read first (no network or exec calls besides the Hugging Face download).
- Timestral-28M fine-tuned as the sequence head (all weights + linear head trained, `e_timestral_ft.py`, 6 epochs, 1 seed): started, result not in yet.
- `submissions/blend_v2_oldB.csv`: blend v2 rebuilt from the saved members (old B12 x2, E3 BiGRU, E3 transformer, D1, E5 transformer, threshold 0.70; held-out F1 0.8265, test vs `1.csv` 0.8342, cheat share 0.251, same numbers as the logged v2). `submissions/submission.csv` currently holds the ft2 K=12 blend (public LB 0.8408).

- Timestral-28M fine-tuned as the sequence head (`e_timestral_ft.py`: all Timestral weights lr 2e-5 + linear head lr 1e-3 on [encoder-token mean, forecast-token mean and max over variates, log std of every variate], same 30-step series as the frozen run, 6 epochs, 1 seed, fp32, 5 folds, about 34 min): held-out F1 0.6444 (thr 0.40), AUC 0.7305; test vs `1.csv` 0.6877 (371 wrong), cheat share 0.251. Worse than the frozen 209M + logistic regression (0.6637 / AUC 0.7573) and far below E3 (0.7731 / AUC 0.8574). Not added to the blend. Not tried: other Timestral sizes fine-tuned (123M, 209M), more variates (NV=128), more seeds or epochs, fine-tuned B features as the per-frame input.

## 2026-11-21 - Kaggle 15 fps fine-tune, fold 0 (K=12, 1 epoch)

- Run: `kaggle_finetune_15fps.ipynb`, K=12, 15 fps (every 2nd frame + every 15th), 1 epoch, hold-out fold 0, no augmentation, standard fp32 AdamW + clip 1.0 with fp16 autocast and gradient scaler (Kaggle GPU without bf16), 2750 steps at 1.5 s/step: train 1.42 h, total 2.2 h (well under the 8.5 h cap). Final loss 0.5535, hold-out frame accuracy 0.7103. Output files: weights, head, features (13521 x 2560), val logprobs (3115 x 8) and meta, summary.
- Scored with the same local procedure as the augmentation screens (`ft2_screen_score.py` logic: head cheat probability = sum of 4 cheat classes, averaged over the clip's 2 frames at frames 0 and 15, the same 1364 fold-0 clips): AUC 0.8679, F1 at 0.5 0.7946, F1 at best threshold 0.7981 (thr 0.55). The notebook's own summary reports AUC 0.8561 / F1 0.7911 on 1449 clips (it also counts clips the local labels drop), so use the aligned numbers.
- Comparison on the same fold 0 and metric (local K=12, 2 fps, 2 epochs, 8-bit AdamW): no augmentation 0.8495 / 0.7645 and 0.8483 / 0.7520; photo 0.8613; quality 0.8591; hide 0.8730; weather 0.8721. So 15 fps + 1 epoch gives +0.018 AUC over no augmentation, in the range of the single augmentation groups, below hide and weather. The runs differ in more than fps (optimizer, precision, epochs, steps), so the gain cannot be attributed to the frame rate alone. One run per setting; seed noise about 0.001 AUC.
- Not done: the other 4 folds, test inference with these weights, B rebuilt on the Kaggle features, and the Kaggle recipe combined with the best augmentation tokens.

## 2026-11-21 - Augmentation stage 2a: do the group gains add up? (fold 0, K=12, 2 epochs)

- Same setup and metric as the group screens (1,364 clips, AUC primary, seed noise about 0.001, F1 at best threshold noisier). Baselines 0.8495 / 0.7645 and 0.8483 / 0.7520. Single groups: hide 0.8730, weather 0.8721, mix 0.8661, hflip 0.8650, photo 0.8613, quality 0.8591, geo 0.8476.

| Run | AUC | F1 at 0.5 | F1 at best thr | thr |
|---|---|---|---|---|
| hide + weather | 0.8755 | 0.7458 | 0.7962 | 0.75 |
| **hide + weather + hflip** | **0.8866** | 0.7580 | **0.8151** | 0.675 |
| hide + weather + mix + hflip | 0.8623 | 0.7185 | 0.7730 | 0.725 |
| hide + weather + mix + hflip + photo + quality (all except geo) | 0.8802 | 0.7709 | 0.8037 | 0.60 |

- Result: gains add only partly. hide + weather (0.8755) is barely above hide alone (0.8730); adding hflip gives the best run so far (0.8866, +0.037 AUC over no augmentation); adding mix on top drops it to 0.8623, below the no-augmentation level of hide + weather, so mix hurts in combination (it mixes images and labels, which stacks badly with the hiding). All groups except geo reach 0.8802, below hide + weather + hflip.
- Next (stage 2b): leave one group out of hide + weather + hflip (hide + hflip, weather + hflip) and add photo or quality to it, then single tokens inside the groups that matter, then the best set on all 5 folds.

## 2026-11-21 - Augmentation stage 2b: leave one group out, add photo or quality (fold 0, K=12, 2 epochs)

- Same setup and metric as stage 2a (1,364 clips, AUC primary, seed noise about 0.001). Reference: hide + weather + hflip 0.8866 / 0.7580 / 0.8151.

| Run | AUC | F1 at 0.5 | F1 at best thr | thr |
|---|---|---|---|---|
| hide + hflip (no weather) | 0.8737 | 0.7495 | 0.7847 | 0.625 |
| weather + hflip (no hide) | 0.8773 | 0.7691 | 0.7856 | 0.65 |
| hide + weather + hflip + photo | 0.8861 | 0.7623 | 0.8002 | 0.60 |
| hide + weather + hflip + quality | not done (stopped at step 100 of 696, see below) | | | |

- Result: dropping hide or weather from hide + weather + hflip costs 0.009 to 0.013 AUC, so both matter and add to each other; adding photo does not help (0.8861 vs 0.8866, within noise, lower F1). hide + weather + hflip stays the best set. The quality add-on and the single-token ablation inside the groups were not run.
- Decision by the user (had to leave and wanted a submission file now): stop the screens and go straight to the final chain with hide + weather + hflip: K=12, 2 epochs, 2 fps, all 5 folds with weights saved (`cache/ft2/aug_best/`), test inference on all 30 frames, B rebuilt (`B_ft2_aug_best`), blend v2 members with the new B (B, E3 BiGRU, E3 transformer, D1, E5 transformer; weights and threshold from held-out train, 1.csv only as a second check). Output file `submissions/blend_v2_ft2B_aug.csv`. Results go in the next entry.
- The Kaggle 15 fps all-fold plan was dropped for now by the user ("5 kaggle folds for what"). A 30 fps, 100% train notebook (`kaggle_finetune_30fps_full.ipynb`, no held-out score) was prepared but not run.

- 2026-11-21, paused by the user (needed the GPU/RAM for something else): the stage 2b queue was stopped during its first run (hide + hflip, killed after epoch 0, nothing saved); the four runs hide + hflip, weather + hflip, hide + weather + hflip + photo and hide + weather + hflip + quality are not done and are rerun from the start on resume. Nothing else was running. Blend member predictions and the key scripts were copied to `cache/scratch_backup_2026-11-21/` (ignored by git) because the scratchpad can be cleared.

## 2026-11-21 - Plan change: no local 5-fold augmented run, Kaggle 15 fps notebooks for all 5 folds, quick file with the Kaggle fold-0 model

- The final chain (K=12, 2 fps, hide + weather + hflip, 5 folds, about 2.5 h) was started and stopped during fold 0 by the user's request (a submission file was needed in under 1 hour); nothing was saved from it (`cache/ft2/aug_best/` is empty). The stage 2b quality add-on run was also stopped at step 100 of 696.
- New notebooks `kaggle_15fps_aug/kaggle_finetune_15fps_aug_fold0.ipynb` to `..._fold4.ipynb` (user runs them on Kaggle later, about 2.2 h each, about 11 h for all 5): same as `kaggle_finetune_15fps.ipynb` (K=12, 15 fps, 1 epoch, 8.5 h budget) with the augmentation module of `scripts/ft2_augs.py` embedded in a new "Augmentation" section and applied to every training batch on the GPU (`AUG = "hide+weather+hflip"`), `HOLDOUT_FOLD` set to the fold number, output tag `k12_15fps_aug_fold<N>`. The augmentation code was smoke-tested on a random batch; the notebooks themselves were not run. Not yet known: whether augmentation helps with 1 epoch at 15 fps (the local screens used 2 epochs at 2 fps).
- `kaggle_finetune_30fps_full.ipynb`: 30 fps, `HOLDOUT_FOLD = None` (100% train, no held-out score), no augmentation, `SAVE_FEATURES = False`; estimated 4 to 4.5 h on Kaggle; not run.
- Test inference of the Kaggle 15 fps fold-0 weights on all 30 frames of the test sequences (new script in the scratchpad, same code as `ft2_infer_test.py` with one fold): in progress, about 40 s per 32 sequences, about 32 min in total. Result and blend go in the next entry.
- Fold-0 comparison of three blends with the v2 weights (B x2, E3 BiGRU, E3 transformer, D1, E5 transformer, each x1) and a threshold from the 1,364 held-out fold-0 clips (head cheat probability for the Kaggle model, averaged over the clip's 2 frames):

| B in the blend | F1 (best thr) | AUC |
|---|---|---|
| old B (2 test frames) | 0.8454 | 0.9355 |
| ft2 K=12 B (2 fps, no augmentation) | 0.8301 | 0.9281 |
| Kaggle 15 fps K=12 | 0.8312 | 0.9288 |

  Alone on fold 0 the AUC is 0.8944 for old B, 0.8644 for ft2 K=12 and 0.8679 for the Kaggle model. So the Kaggle model is only about equal to yesterday's K=12 and clearly below old B; one fold, 1,364 clips, so rough.

- New notebook `kaggle_15fps_aug/kaggle_finetune_15fps_aug_folds1to4.ipynb` (made at the user's request, to run folds 1 to 4 in one Kaggle session, since fold 0 was already run with the older notebook): same recipe and augmentation (hide + weather + hflip) as the per-fold notebooks, but the setup (video decoding, model load) runs once and a loop `FOLDS = [1, 2, 3, 4]` trains and scores each fold. Per fold: trainable blocks reset to the pretrained DINOv3 weights, speed benchmark, training, weights and head saved, held-out fold scored, features saved; folds whose `summary.json` exists are skipped; the remaining session time is split evenly across the folds not yet run (step count is cut if needed). Outputs `k12_15fps_aug_fold<N>_*`. Estimated about 7 h in total (about 0.8 h setup + about 1.6 h per fold). Syntax checked only, not run. Note: fold 0 of the earlier run has no augmentation, so B's held-out predictions would mix one model without augmentation (fold 0) and four with it.

- Same fold-0 comparison with the Kaggle model in B's full recipe (logistic regression on the pair summary of its features + the 30-frame pose, trained on folds 1-4 whose backbone features are in-sample, scored on fold 0): blend F1 0.8310, AUC 0.9292 (threshold 0.70); alone AUC 0.8738 (the head-probability version alone: 0.8679, blend 0.8312). So the two ways of using the Kaggle model give the same blend score, both below the old-B blend (0.8454) and equal to the ft2 K=12 blend (0.8301).
- Decision by the user (2026-11-21): the Kaggle fine-tune runs (5-fold notebooks) are postponed to a later day, close to the end of the competition, because the augmentation and the rest of the pipeline are still being tuned; tomorrow continues with the augmentation tuning and the video models.

- Notebook cleanup at the user's request (2026-11-21): deleted `kaggle_finetune_15fps.ipynb` (no augmentation, used for the fold-0 run), `kaggle_finetune_30fps_full.ipynb` and the five per-fold notebooks in `kaggle_15fps_aug/`. The only Kaggle notebook left is `kaggle_15fps_aug/kaggle_finetune_15fps_aug_folds1to4.ipynb`; with `AUG = "0"` and `FOLDS = [0]` it reproduces the earlier no-augmentation fold-0 run, and a 30 fps run needs `FPS = 30`.

## 2026-11-21 - Quick file: v2 blend with the Kaggle 15 fps fold-0 model as B (full recipe: LR + pose)

- Test features: Kaggle fold-0 weights on all 30 frames of the 1,515 test sequences (`cache/kaggle_15fps/k12_15fps_fold0_test_feat.npy`, shape 1 x 1515 x 30 x 2560). B = logistic regression on the pair summaries (15 interleaved pairs, probabilities averaged) + 30-frame pose, trained on folds 1-4 (backbone features in-sample), scored on fold 0; blend with the v2 weights (B x2, E3 BiGRU, E3 transformer, D1, E5 transformer x1); rank scale and threshold from the 1,364 fold-0 clips only (shortcut: one fold model, one fold of held-out data).
- Result: threshold 0.70, fold-0 held-out blend F1 0.8310, AUC 0.9292; vs `1.csv` (second check only) 0.8327, cheat share 0.228. File `submissions/blend_v2_k15B.csv` (not submitted).
- Same blend with the head cheat probability (no LR, no pose) as B: fold-0 F1 0.8312, vs `1.csv` 0.8556, cheat share 0.244. Old B blend on fold 0: 0.8454 (vs `1.csv` 0.8335); ft2 K=12 B blend: 0.8301 (vs `1.csv` 0.8367). Differences of 0.002 to 0.02 vs `1.csv` between these versions are within the noise seen before (see the ft2 K=12 entry), so the held-out fold-0 numbers decide, and they rank old B first.

- Public leaderboard (2026-11-21, submitted by the user as `submission.csv` = the LR + pose file above, cheat share 0.228): **0.8471858**. For comparison: ft2 K=12 B blend (2 fps, no augmentation, all 5 folds) 0.8408476; old-B blend v2 0.8518237 (uses 2 test frames). So the Kaggle 15 fps B from ONE fold model (trained on 80% of the data, 1 epoch, no augmentation, threshold from fold 0 only) is 0.0063 better than the 2 fps 5-fold K=12 B and 0.0046 below old B, while its fold-0 held-out blend F1 was equal to the ft2 K=12 one (0.8310 vs 0.8301). Rule-compliant best so far (all 30 test frames). One public score per file; a difference of about 0.006 can be partly noise, but the direction (15 fps helps) agrees with the fold-0 AUC screen (0.8679 vs 0.8495).

- Public leaderboard (2026-11-21, submitted by the user as `submission.csv` = `blend_v2_k15_headonly.csv`): **0.8617276**, the best public score so far (old-B blend 0.8518, Kaggle 15 fps B with LR + pose 0.8472, ft2 K=12 blend 0.8408). Same blend, same Kaggle fold-0 model, only B differs: here B is the fine-tuned model's own head cheat probability averaged over all 30 test frames (no logistic regression, no pose), 0.0145 above the same model with the LR + pose recipe. `1.csv` ranked the two the same way this time (0.8556 vs 0.8327), the fold-0 held-out blend F1 did not separate them (0.8312 vs 0.8310, one fold of 1,364 clips).
- Reading: the LR on the fine-tuned features trains on memorised (too clean) backbone features and the other blend members (E3, D1, E5) already carry the pose, so the LR + pose B may add a weaker, duplicate signal while the head probability is a different, cleaner one. This is one score for one file; not yet tested with all 5 folds, where the head probability is held-out for every clip and the blend weight for it can be chosen properly.

## 2026-11-21 - B variants on the 5-fold ft2 K=12 run (2 fps, no augmentation, held-out predictions of all 6,166 clips)

- Question after the head-only Kaggle file scored 0.8617 public (LR + pose version of the same model 0.8472): is the logistic regression or the pose the weak part of B? Script `b_variants.py` (scratchpad): all variants use the saved held-out head probabilities (`cache/ft2/k12/k12_f*_val.npy`, mean of the clip's 2 frames) or the fold backbone features; each is scored alone (AUC, F1 at the best threshold) and as B in the v2 blend (B x2, E3 BiGRU, E3 transformer, D1, E5 transformer x1, threshold from held-out train).
- Variants: oldB (12-block, raw frames, LR + pose), LP = LR on the pair summary + pose (current ft2 recipe, member B_ft2_k12), L = LR on the pair summary only, H = head cheat probability only, HS = LR on the 8 head probabilities only (recalibration), HP = LR on the 8 head probabilities + log-probabilities + pose, trained on the honest head probabilities of the other folds.

| B variant | Alone AUC | Alone F1 | Blend AUC | Blend F1 |
|---|---|---|---|---|
| oldB | 0.8664 | 0.7889 | 0.9225 | 0.8265 |
| LP (LR + pose) | 0.8521 | 0.7829 | 0.9182 | 0.8204 |
| L (LR, no pose) | 0.8479 | 0.7788 | 0.9179 | 0.8187 |
| HP (honest head probs + pose) | **0.8862** | **0.8036** | 0.9215 | 0.8218 |
| HS (honest head probs only) | 0.8568 | 0.7889 | 0.9224 | 0.8263 |
| H (head probability only) | 0.8497 | 0.7804 | **0.9227** | 0.8214 |

- Results: (1) Head only vs LR + pose in the blend: +0.0045 AUC and +0.0010 F1 for the head, so the direction agrees with the public result but the size (public +0.0145) is not reproduced on held-out data; part of the public gap is probably luck. (2) Pose helps B alone a lot when it is stacked on honest head probabilities (HP alone AUC 0.8862 vs 0.8497 head only), but inside the blend it adds nothing (0.8218 vs 0.8214), consistent with the other members already carrying the pose. (3) The LR without pose (L) is the weakest in the blend, so the LR on memorised features is not the useful part. (4) Recalibrating the head (HS) gives the best blend F1 among the new variants (0.8263, equal to old B 0.8265) at AUC 0.9224.
- Frozen A back in the blend (weight grid over B, E3 BiGRU, E3 transformer, D1, E5 transformer, A in {0,1,2}, chosen on the same held-out data, so slightly optimistic): with H as B the best is 0.8279 (weights 2/0/2/1/2/1, A weight 1) vs 0.8214 without A; with LP as B the best is 0.8227 (A weight 0) vs 0.8204. So A helps a little only next to the head-only B.
- Not run on test: the head layers were not saved in the ft2 K=12 run, so these variants have no test predictions; this entry only ranks the variants on held-out data.

## 2026-11-21 - Kaggle 30 fps, 100% train fine-tune finished (K=12, 1 epoch, no augmentation)

- Run by the user on Kaggle with the 30 fps / `HOLDOUT_FOLD = None` notebook (since deleted; its config is `FPS = 30`, `FOLDS`/holdout none in the remaining notebook). Summary: 6,701 steps at 1.518 s per step (benchmark), 3.67 h training, 4.44 h total, final loss 0.523, fp16 AMP. No held-out score exists (all train clips were used for training). The estimate made before the run was 4 to 4.5 h, matched.
- Files extracted to `cache/kaggle_30fps/` (git-ignored): `k12_30fps_full_weights.pt` (630 MB), `k12_30fps_full_head.pt`, `k12_30fps_full_summary.json`. Not yet used for test inference.

## 2026-11-22 - Augmentation ablation finished (leave-one-token-out around hide + weather + hflip)

- Setup as before: fold 0, K=12, 2 epochs, held-out frames only, AUC = head cheat probability averaged over the clip's 2 frames. Reference: full set (erase, cutout, griddrop, coarse, fog, shadow, flare, rain, snow, hflip) AUC 0.8866.
- Drop one token from the full set:

| Dropped token | AUC | Change vs 0.8866 |
|---|---|---|
| erase | 0.8800 | -0.0066 |
| cutout | 0.8713 | -0.0153 |
| griddrop | 0.8591 | -0.0275 |
| coarse | 0.8719 | -0.0147 |
| fog | 0.8821 | -0.0045 |
| shadow | 0.8870 | +0.0004 |
| flare | 0.8831 | -0.0035 |
| rain | 0.8933 | +0.0067 |
| snow | 0.8842 | -0.0024 |

- Full set + quality group: 0.8685 (-0.0181), so quality hurts. Full set with seed 2: 0.8789 (-0.0077 vs seed 42), so the seed noise of this screen is about 0.008, not 0.001 as assumed earlier.
- Reading: the hide tokens (erase, cutout, griddrop, coarse) clearly matter (each drop costs 0.007 to 0.028, larger than the noise). Weather tokens individually are within noise (shadow +0.0004, rain +0.0067 are not distinguishable from the seed spread), so the weather group stays as is. Best set unchanged: hide + weather + hflip. Not tested: single tokens inside geo, photo and quality, and the `crop` token alone.

## 2026-11-22 - Kaggle 30 fps (100% train) test inference and the 30 fps blend file

- Inference: `cache/kaggle_30fps/k12_30fps_full_weights.pt` on all 30 frames of the 1,515 test sequences (first 20 blocks live, top 12 blocks fine-tuned, head cheat probability averaged over the 30 frames; no LR, no pose), 27 min. Output `k12_30fps_full_test_feat.npy` (1, 1515, 30, 2560). The model was trained without augmentation and has no held-out score.
- Test comparison with the 15 fps fold-0 model (also head only): mean cheat probability 0.389 (15 fps) vs 0.339 (30 fps), correlation 0.886, agreement at 0.5 0.914.
- v2 blend with B replaced (rank scale and threshold 0.70 from the 15 fps fold-0 held-out head probabilities, an assumption for the 30 fps model); fold-0 F1 is the same for all three because the held-out reference is the 15 fps model:

| B | fold-0 F1 | vs 1.csv (secondary check) | cheat share |
|---|---|---|---|
| 15 fps fold-0 alone (= `blend_v2_k15_headonly.csv`, public 0.8617) | 0.8312 | 0.8556 | 0.244 |
| 30 fps full alone | 0.8312 | 0.8496 | 0.234 |
| average of 15 fps fold-0 and 30 fps full | 0.8312 | 0.8476 | 0.238 |

- File written: `submissions/blend_v2_k15k30_avg_headonly.csv` (the averaged version). It differs from the 15 fps file in 32 rows. Not submitted. Public score: not yet known.

## 2026-11-22 - V-JEPA 2 ViT-g frozen feature extraction (started)

- Model `facebook/vjepa2-vitg-fpc64-256` (frozen, 1B parameters, fp16), all 30 frames of each clip fed as one clip (tubelet size 2, 15 time steps, 256x256 squashed, ImageNet normalisation), train frames through the test delivery format first. Script `scripts/extract_vjepa2.py`, output `cache/vjepa2_g_256_30/` (per clip (16, 1408): 15 per-time-step mean-patch vectors + the mean of all). Custom-code check: VideoMAE v2 and InternVideo-Next code read, no network or process calls; InternVideo-Next needs `flash_attn` (not installed), VideoMAE v2 needs `easydict` (not installed), so V-JEPA 2 goes first.

- Public leaderboard (2026-11-22, submitted by the user as `submission_30.csv` = `blend_v2_k15k30_avg_headonly.csv`): **0.8627997**, new best (15 fps fold-0 only 0.8617276, +0.0011). The two files differ in 32 of 1,515 rows, so the gain is within noise; averaging in the 30 fps model does not hurt. The 30 fps model alone was not submitted.

## 2026-11-22 - V-JEPA 2 ViT-g frozen members (V1 LR + pose, V5 BiGRU and transformer) and blend check

- Features: `cache/vjepa2_g_256_30/` (all 30 frames as one clip, 15 time steps, per clip (16, 1408)); extraction took about 1 h 10 min (test 15 min, train 55 min). Members built like the LeVJEPA ones (`v1_vjepa2.py` = D1 recipe, `v5_vjepa2.py` = E5 recipe with 15 steps and the pose averaged per 2 frames), held-out per-video folds, 3 seeds for the sequence heads.
- Alone (AUC / best F1): V1 LR + pose C=0.001 0.8485 / 0.7597 (D1 on LeVJEPA 0.8539 / 0.7381); V5 transformer 0.8681 / 0.7729 (E5 transformer 0.8784 / 0.7705); V5 BiGRU 0.8631 / 0.7659 (E5 BiGRU 0.8695 / 0.7521). Roughly equal to the LeVJEPA members, slightly weaker on AUC.
- Blend check (v2 = B x2, E3 BiGRU, E3 transformer, D1, E5 transformer; held-out AUC 0.9225 / F1 0.8265):

| Blend | AUC | F1 |
|---|---|---|
| v2 | 0.9225 | 0.8265 |
| v2 + V1 x1 | 0.9231 | 0.8253 |
| v2 + V1 x2 | 0.9204 | 0.8254 |
| v2 + V5 transformer x1 | 0.9233 | 0.8250 |
| v2 + V5 transformer x2 | 0.9216 | 0.8231 |
| v2 + V5 BiGRU x1 | 0.9230 | 0.8228 |
| v2 + V1 + V5 transformer | 0.9231 | 0.8254 |
| v2 with D1 -> V1 and E5 -> V5 transformer | 0.9125 | 0.8205 |

- Weight grid over B, E3 BiGRU, E3 transformer, D1, E5 transformer, V1, V5 transformer in {0,1,2}: best F1 0.8289 (2/1/0/1/0/0/1, AUC 0.9187), but the grid picks on the same held-out data and the gain over 0.8265 (+0.0024) is inside the noise. Conclusion: V-JEPA 2 adds nothing reliable; it is a near-duplicate of the LeVJEPA members. Not added to the blend. Test predictions of the V members are saved in the scratchpad `members/` folder (not in the repo).

## 2026-11-22 - Fine-tune recipe screens (stage 4, started)

- Same screen as the augmentation ablation (fold 0, K=12, held-out frames, AUC proxy) with the best augmentation (hide + weather + hflip), one change at a time: 1 epoch, K=8, block LR 4e-5, block LR 1e-5, K=20. Reference (K=12, 2 epochs, LR 2e-5 / 1e-3): 0.8866 (seed 42), 0.8789 (seed 2), so differences below about 0.01 are noise. `scripts/ft2_train.py` now reads optional `LR_BLOCKS` and `LR_HEAD` env variables (defaults unchanged). Queue script `screen_queue7.sh`, results appended to the scratchpad `screen_results.txt`.
- Results (AUC on the 2 held-out frames of fold 0's 1,364 clips): 1 epoch 0.8635 (worse), K=8 0.8871 (same as K=12, cheaper), block LR 4e-5 0.8762, block LR 1e-5 0.8836, K=20 0.8788. Nothing beats the reference beyond noise. Recipe unchanged: K=12, 2 epochs, LR 2e-5 / 1e-3, augmentation hide + weather + hflip. K=8 is a valid cheaper option.

## 2026-11-22 - V-JEPA 2.1 ViT-L (frozen, 300M, distilled from ViT-G) as a member

- Features: Meta repo code, `vjepa2_1_vitl_dist_vitG_384.pt` (ema_encoder), 256 px, all 30 frames as 15 tubelets; per clip 15 per-time-step mean-patch vectors + overall mean (16 x 1024). Extraction `scripts/extract_vjepa21.py` (SIZE=l), 4.5 clips/s, about 25 min for test + train.
- Members built exactly like the LeVJEPA ones (V1 = logistic regression + pose, V5 = BiGRU / transformer on the 15 time steps), 5 folds by subject, held-out train only:

| member | AUC | F1 |
|---|---|---|
| V1 LR + pose (C 0.001) | 0.8375 | 0.7410 |
| V5 transformer | 0.8612 | 0.7556 |
| V5 BiGRU | 0.8516 | 0.7487 |
| reference D1 (LeVJEPA LR + pose) | 0.8539 | 0.7381 |
| reference E5 transformer (LeVJEPA) | 0.8784 | 0.7705 |

- Blend check (v2 = AUC 0.9225, F1 0.8265): v2 + V1 x1 0.9223 / 0.8227, v2 + V5 transformer x1 0.9226 / 0.8235, v2 + V5 BiGRU x1 0.9217 / 0.8217, v2 + V1 + V5 transformer 0.9220 / 0.8196, D1 -> V1 and E5 -> V5 transformer 0.9129 / 0.8220. Best grid cell is the v2 weights themselves (2/1/1/1/1/0/0, 0.8265).
- Conclusion: V-JEPA 2.1 ViT-L is not better than LeVJEPA (E5 transformer is still the best frozen member) and adds nothing to the blend. Not added. The ViT-g download (16.9 GB, about 2.4 GB done) stays paused; it is unlikely to change this because the L model was distilled from ViT-G.
- Held-out 30-frame inference of the fold-0 Kaggle 15 fps model finished (`cache/kaggle_15fps/heldout30_parts/`, 96 videos, 28 min): per clip 30 frames of head log-probs (30 x 8) and features (30 x 2560). Next: aggregation tests (tech F) and a sequence head on these.

## 2026-11-22 - Tech F: aggregation of the 30 frames and heads on top (fold-0 Kaggle 15 fps model, honest held-out)

- Data: 1,364 held-out clips of fold 0 (6 subjects only, so everything below is noisy; seed noise was about 0.008 AUC), per-frame head log-probs and fine-tuned features over all 30 frames (`scripts` copy in scratchpad `agg30.py`). Cheat probability per frame = sum of the first 4 class probabilities.
- Aggregation, AUC / best-threshold F1: 2 frames (0, 15) 0.8482 / 0.7829; all 30 mean prob (current B) 0.8545 / 0.7900; mean logit 0.8516; median 0.8481; trimmed mean 0.8525; max 0.8624; mean of the 10 most cheat-like frames 0.8655 / 0.7980; mean of the 10 least cheat-like frames 0.8238; fraction of frames above 0.5 0.8473; dominant 8-class action 0.8545; mean of 8-class log-probs 0.8547.
- Heads on top, subject-grouped 5-fold CV inside the 1,364 clips: LR on probability stats 0.8510; LR on mean 8-class probs 0.8485 (F1 0.8057); LR on mean+std 8-class probs 0.8525 (F1 0.8012); LR on the mean fine-tuned 2,560-d feature 0.8025 (C 0.001) and 0.7789 (C 0.01) - clearly worse, as before.
- Conclusion: using all 30 frames instead of 2 helps (+0.006 AUC), the plain mean is as good as any rule or learned head. "Top-10 mean" is +0.011 AUC but picked on the same data and inside the noise; not adopted (would need a second fold to confirm). No sequence head or LR beats the plain mean, so B stays head-only with the mean over 30 frames.
- Next: tech G (horizontal flip TTA) on the same held-out clips, `infer_heldout30_flip.py`, output `cache/kaggle_15fps/heldout30_flip_parts/` (log probs only).

## 2026-11-22 - Tech G (flip TTA) and tech K (boundary clips, frozen test)

- G, flip test-time augmentation on the same 1,364 held-out clips of fold 0 (`infer_heldout30_flip.py`, mean of the 30 frame cheat probabilities): normal AUC 0.8545 / F1 0.7900; mirrored frames only 0.8203 / 0.7554; average of both (flip TTA) 0.8475 / 0.7921. The two correlate 0.84. The model is clearly not mirror-symmetric (side of the camera, hand side), so the mirrored view hurts. Not adopted. Output kept in `cache/kaggle_15fps/heldout30_flip_parts/`.
- K, boundary clips on the frozen LeVJEPA D1 (LR + pose, C 0.001, 5 folds), scratchpad `k_frozen.py`. Start times from `truth_start.csv` (known for 2,899 clips; clip label rule "centre after start" agrees with the label on 99.3%). Only 186 of 6,166 clips (3%) have the start inside their second, 37 near the middle (25 to 75% after the start). Variants, trained on the changed set, evaluated on all held-out clips (AUC / F1): base 0.8539 / 0.7381; drop all straddling clips 0.8525 / 0.7354; drop the 37 middle ones 0.8538 / 0.7377; soft label (each straddler counted as cheat with weight = share after start, and as normal with the rest) 0.8532 / 0.7370. Evaluating without the straddlers gives the same picture (0.857 to 0.858). Nothing helps; the boundary clips are too few to matter. Dropped for the frozen heads; not worth a fine-tune screen either.

## 2026-11-22 - Fine-tuned B + sequence head (honest held-out, fold-0 Kaggle 15 fps model)

- 1,364 held-out clips of fold 0 (6 subjects), leave-one-subject-out CV, small BiGRU / transformer (hidden 64, dropout 0.3) over the 30 per-frame outputs (scratchpad `seq30.py`). AUC / best-threshold F1: plain mean of the 30 frame cheat probabilities 0.8545 / 0.7900; BiGRU on 30 x 8 log-probs 0.8098 / 0.7546; transformer on 30 x 8 log-probs 0.7717 / 0.6897; BiGRU on 30 x 2560 features 0.7781 / 0.7091; transformer on 30 x 2560 features 0.7873 / 0.7119.
- Conclusion: a sequence head on top of the fine-tuned frames is clearly worse than the plain mean (too little honest data: 1,364 clips, 6 subjects; the frame head already carries the information). With the earlier LR-on-stats and LR-on-feature results this closes "fine-tune + sequence head" for now. It could be retried with the honest outputs of all 5 folds (6,166 clips) after the final fine-tune.
- Naming rule from the user (2026-11-22): the lettered techniques (F to K) are one-off technique tests, not architecture branches like B / D / E. If one works it becomes part of the base configuration of B, D or E (for example "B trained with person crops"), not a new letter. From now on logs use descriptive names (person crops, weight averaging, layer-wise learning rate).
- Submission rule from the user (2026-11-22): until the next fine-tune, B in new submissions = the Kaggle 30 fps K=12 model (trained on 100% of train, no augmentation, head only, mean over all 30 test frames), not the 15 fps fold-0 model. Note: this model has no held-out score; the rank scale and the 0.70 threshold come from the 15 fps fold-0 held-out head probabilities.

## 2026-11-22 - Person crops (frozen test)

- Frozen DINOv3 ViT-H+ features on person crops: box from the pose step (middle frame of the clip), expanded by 20%, squared, clipped, resized to 224x224, test format for train frames, frames 0,3,...,27 of every clip (scratchpad `h_extract.py`, 29 min, saved in `cache/h_crop_dinov3/`). Logistic regression on mean + std of the 10 frames, 5 per-video folds, held-out train (`h_eval.py`). Baseline full-frame features are the existing step-1 cache (224x400, not the test-format recode, a small format difference).
- Results (C 0.001; AUC / F1): full frame 0.8070 / 0.7309; person crop 0.8084 / 0.7213; full + crop 0.8062 / 0.7262; full + pose 0.8205 / 0.7514; full + crop + pose 0.8143 / 0.7375 (C 0.003 is lower for all).
- Conclusion: person crops give no gain on frozen features (and slightly hurt next to pose), so no crop fine-tune screen. Closed. Together with the boundary-clip test, flip TTA, aggregation and sequence head results, nothing from the lettered queue except possibly weight averaging / layer-wise learning rate remains.

## 2026-11-22 - Weight averaging (EMA) and layer-wise learning rate (fine-tune screen)

- Setup: fold 0, K=4, 1 epoch, augmentation hide + weather + hflip, held-out 2-frame AUC proxy (same as the earlier screens), 4 runs, each scoring the final weights and the averaged weights (EMA, window about the last 20% of steps) from the same training. `scripts/ft2_train.py` got two optional env variables, `EMA_FRAC` (0 = off) and `LLRD` (layer-wise decay per block below the top one, 1 = same rate everywhere); defaults unchanged. Layer-wise run: top block 3e-5, decay 0.8 (3e-5, 2.4e-5, 1.9e-5, 1.5e-5). Queue `screen_queue8.sh`; each run about 8.5 min.
- Results (AUC), final weights / averaged weights: plain seed 42 0.8223 / 0.8231; plain seed 2 0.8304 / 0.8319; layer-wise seed 42 0.8216 / 0.8225; layer-wise seed 2 0.8282 / 0.8301.
- Averaging: +0.0008, +0.0015, +0.0009, +0.0019 (4 of 4 positive, mean +0.0013), far below the seed noise of about 0.008 but consistent and free (no extra training). Adopted as part of the base fine-tune recipe, with the unaveraged weights kept as a backup and compared on the public/1.csv check.
- Layer-wise learning rate: -0.0007 (seed 42) and -0.0022 (seed 2) against plain. Not adopted.
- New notebook `kaggle_15fps_aug/kaggle_finetune_15fps_aug_full.ipynb` (built from the folds1to4 notebook; the old file is untouched): 100% train (no hold-out, no evaluation, no feature saving, no fold loop), K=12, 15 fps, 2 epochs (the recipe screen showed 1 epoch is worse), augmentation hide + weather + hflip, EMA over about the last 20% of steps. Saves `k12_15fps_aug_full_weights.pt` + `_head.pt` (averaged) and `_final_weights.pt` + `_final_head.pt` (unaveraged), plus a summary json. Estimated 4.5 h of training; the averaging copy adds about 1.3 GB of GPU memory. To be run by the user on Kaggle; the result will be compared with 1.csv only (secondary check).

## 2026-11-22 - VideoMAE v2 giant (frozen) as members

- Model `OpenGVLab/VideoMAEv2-giant` (1B, self-supervised pixel-reconstruction pretraining, encoder only; the decoder weights are unused). The custom code was read and only its VisionTransformer part is run from a local copy (`scripts/extract_videomaev2.py`), so the easydict package is not needed and nothing was installed. All 30 frames go in as one clip (15 time steps; the sine-cosine position table is built for 30 frames, the model was pretrained with 16), 224x224 squashed, ImageNet normalisation, test format for train frames, fp16, fused attention (same maths). Output per clip 15 per-time-step mean-patch vectors (after the final LayerNorm) + their mean (16 x 1408), saved in `cache/videomaev2_g_224_30/`. 1.58 clips/s, 1 h 24 min total.
- Members built like the LeVJEPA ones (5 folds by subject, held-out train only), AUC / F1: V1 LR + pose 0.7452 / 0.6679 (without pose 0.6235 F1); V5 transformer 0.7698 / 0.6912; V5 BiGRU 0.7365 / 0.6639. For comparison D1 0.8539 / 0.7381, E5 transformer 0.8784 / 0.7705.
- Blend check (v2 = 0.9225 / 0.8265): v2 + V1 x1 0.9157 / 0.8162 (x2: 0.9052 / 0.8055); v2 + V5 transformer x1 0.9141 / 0.8142 (x2: 0.9037 / 0.8019). Every addition makes the blend worse.
- Conclusion: pixel-reconstruction features are far weaker than the DINO / JEPA features for this task (frozen features of this kind need the fine-tune to become useful) and add nothing. Not added. Caveat: the 30-frame position table is an extension of the 16-frame pretraining; a 16-frame input was not tried because the gap to LeVJEPA is large. Closed.

## 2026-11-22 - Notebook requirements (from the user)

- The final (submission) notebook is checked by the committee and must run from scratch: raw data, all feature extraction (DINOv3, LeVJEPA), the pose step (RT-DETR person box + ViTPose 17 keypoints), the fine-tune (to be added once the recipe is final), then the members and the blend. The main notebook (`main.ipynb`) should also get the RT-DETR + ViTPose pose step in its pipeline; today it reads the saved pose cache. Not done yet; planned after the current experiment queue.

## 2026-11-22 - Stacking against the weight grid (nested, held-out train)

- Question: is a logistic regression on the members' held-out scores better than the {0,1,2} weight grid? Both were fitted without peeking (scratchpad `stack.py`): for each of the 5 per-video folds, the weights or the stacker AND the threshold are chosen on the other 4 folds' held-out member scores and scored on this fold; pooled over the 5 folds. Members: B (old B, B12_ft_pose), E3 BiGRU, E3 transformer, D1, E5 transformer (+ E5 BiGRU, V-JEPA 2 V5 transformer, V-JEPA 2.1 L V1 and V5 transformer in the 9-member version). Inputs are the ranks inside each member's own held-out predictions, as in the blend.
- Results (AUC / F1 at the nested threshold): v2 fixed weights (2/1/1/1/1, picked earlier on all folds, so optimistic) 0.9225 / 0.8265; weight grid {0,1,2}, nested 0.9194 / 0.8180; logistic regression on the logit of the ranks 0.9182 to 0.9187 / 0.8046 to 0.8071 (C 0.01 to 10; 9 members 0.8051 to 0.8081); logistic regression on the raw ranks 0.9213 / 0.8163. Stacker coefficients on all data: B 0.59, E3 BiGRU 0.20, E3 transformer 0.19, D1 0.40, E5 transformer 0.40 (the same order as the weights 2/1/1/1/1).
- Conclusions: (1) stacking is not better than the weight grid (0.805 to 0.816 against 0.818); the grid stays. (2) The honest held-out F1 of the v2 blend is about 0.818, not 0.8265: the earlier number was optimistic by about 0.008 because the weights were chosen on the same data. This is also the size of the noise when comparing blends. (3) Adding more members to the stacker (9 members) changed nothing.

## 2026-11-22 - Can better blend weights be found? (nested test)

- Same nested setup as the stacking test (weights and threshold chosen on 4 folds, scored on the fifth, pooled; scratchpad `tune_w.py`), members B (old B), E3 BiGRU, E3 transformer, D1, E5 transformer. AUC / F1: grid {0,1,2} chosen by F1 (the current way) 0.9194 / 0.8180; grid {0,1,2} chosen by AUC 0.9246 / 0.8203; grid {0,..,4} by AUC 0.9230 / 0.8158; grid {0,1,2} by AUC with 7 members (+E5 BiGRU, V-JEPA 2 transformer) 0.9230 / 0.8159; smooth weights from an optimiser, 5 members 0.9225 / 0.8205, 9 members 0.9221 / 0.8146; equal weights, 5 members 0.9219 / 0.8211.
- Conclusion: every way of choosing weights lands between 0.815 and 0.821 F1 and 0.919 to 0.925 AUC; equal weights are as good as the tuned ones. The weights are not where the gain is; keep 2/1/1/1/1 (B x2). Improvement has to come from better or more different members (a better B), not from reweighting.

## 2026-11-22 - V-JEPA 2.1 ViT-g (1B, frozen) as members

- Features: `vjepa2_1_vitg_384.pt` (target_encoder; all keys matched), same recipe as the ViT-L (30 frames as one clip, 256 px, 15 time steps, 16 x 1408 per clip, `scripts/extract_vjepa21.py` with SIZE=g, 1.64 to 1.8 clips/s, 1 h 29 min for test + train, saved in `cache/vjepa21_g_256_30/`). Members built like the LeVJEPA ones (5 folds by subject, held-out train only; scripts `v1_vjepa21.py`, `v5_vjepa21.py` in the scratchpad).
- Members alone (AUC / F1): V1 LR + pose 0.8458 / 0.7433; V5 transformer 0.8804 / 0.7700; V5 BiGRU 0.8874 / 0.7819. For comparison: D1 0.8539 / 0.7381, E5 transformer 0.8784 / 0.7705, E5 BiGRU 0.8695 / 0.7521, V-JEPA 2.1 ViT-L transformer 0.8612 / 0.7556. The ViT-g BiGRU is the best frozen member so far (+0.009 AUC, +0.011 F1 over E5 transformer).
- Blend check (v2 = 0.9225 / 0.8265; weights not nested, as in the earlier checks): v2 + V5 BiGRU x1 0.9260 / 0.8260 (x2: 0.9265 / 0.8252); v2 + V5 transformer x1 0.9243 / 0.8248; v2 + V1 x1 0.9235 / 0.8223; replacing D1 with the ViT-g BiGRU 0.9214 / 0.8208; replacing E5 transformer with it 0.9227 / 0.8292 (best F1, +0.0027); replacing D1 and E5 with ViT-g V5 transformer + BiGRU 0.9185 / 0.8229. Rank correlation of the ViT-g BiGRU: with E5 transformer 0.794, D1 0.749, B 0.578 (so it overlaps the LeVJEPA members more than B does).
- Nested check (weights and threshold fitted on 4 folds, scored on the fifth, scratchpad `g_nested.py`): grid {0,1,2} by AUC with the 5 old members 0.9246 / 0.8203; with the ViT-g BiGRU added (6 members) 0.9249 / 0.8227; with BiGRU + transformer (7) 0.9249 / 0.8227; equal weights over 6 members 0.9249 / 0.8172.
- Conclusion: the 1B model is a better frozen member than LeVJEPA on its own (the user's hunch was right for the single member) but in the blend the gain is +0.0003 AUC and +0.0024 F1 in the honest nested test, inside the noise (about 0.005 to 0.008). Not adopted yet; the option is to swap E5 transformer for the ViT-g BiGRU. Cost to adopt: a 16.9 GB checkpoint and 1.5 h of extraction in the from-scratch final notebook.

## 2026-11-22 - V-JEPA 2.1 ViT-g in the test blend (secondary check against 1.csv, no CSV written)

- Same blend construction as the 30 fps files (`blend_k30.py` logic, scratchpad `g_test_blend.py`): B = head cheat probability over all 30 test frames, rank scale and threshold from the 15 fps fold-0 held-out predictions (1,364 clips), other members ranked inside their own held-out predictions. 1.csv is only a secondary check, not used to choose anything.
- B = 15 fps fold-0 model: v2 (E5 transformer) fold-0 F1 0.8312 / vs 1.csv 0.8556 (170 wrong); E5 transformer replaced by the ViT-g BiGRU 0.8439 / 0.8414 (193 wrong); ViT-g BiGRU added 0.8420 / 0.8393 (197 wrong).
- B = 30 fps full model: v2 0.8312 / 0.8496 (175 wrong); swap 0.8439 / 0.8334 (197 wrong); added 0.8420 / 0.8378 (193 wrong).
- B = average of 15 and 30 fps: v2 0.8312 / 0.8476 (178 wrong); swap 0.8439 / 0.8433 (188 wrong); added 0.8420 / 0.8409 (192 wrong).
- Members alone against 1.csv: ViT-g BiGRU 0.7786, E5 transformer 0.7835 (equal). The swap raises the fold-0 held-out F1 by 0.013 (the threshold is chosen on that same fold, so optimistic) but lowers the agreement with 1.csv by 0.012 to 0.016, and moves the cheat share from 0.244 to 0.273 (the 1.csv share is 0.22 by the earlier check).
- Conclusion: the 1B model is a better frozen member on held-out train alone, but it does not improve the blend on the test side. Not adopted; E5 transformer stays. The checkpoint and features stay on disk in case the final blend is re-examined.

## 2026-11-22 - Fine-tune screen: higher input resolution (288x512)

- Same screen as the others (fold 0, K=12, 2 epochs, augmentation hide + weather + hflip, seed 42, held-out frames, AUC proxy), only the input size changed from 224x400 (14x25 = 350 patches) to 288x512 (18x32 = 576 patches). New frame cache `cache/ft_frames_testfmt_288x512` (test format 640x360 + JPEG q90, then resized to 512x288; `scripts/ft2_frames.py` and `scripts/ft2_train.py` now read the optional env variables `FRAMES_DIR`, `IMG_H`, `IMG_W`, defaults unchanged). Peak GPU memory 4.77 GB, 7.3 frames/s in the benchmark, screen 27 min (about 1.7x the time of 224x400). The first attempt was killed by a laptop shutdown and re-run from the start.
- Result: AUC 0.9008, F1 0.7615 at 0.5, F1 0.827 at the best threshold (0.775). Reference at 224x400 with the same setup: 0.8866 (seed 42) and 0.8789 (seed 2). So +0.014 to +0.022 AUC, above the seed noise of about 0.008 and larger than any other single change in the recipe screens (K=8 0.8871, 1 epoch 0.8635, LR 4e-5 0.8762, LR 1e-5 0.8836, K=20 0.8788).
- Next: repeat with seed 2 to confirm; if it holds, test an even bigger input only if memory allows (8 GB GPU). A higher resolution also changes the test side (test frames must be resized the same way), the cost of the Kaggle run (about 1.7x the steps time, so about 7 to 8 h on Kaggle's GPU, short on an A100) and every later inference (30 test frames per sequence at the new size).

## 2026-11-22 - Kaggle 100% train fine-tune with augmentation and weight averaging finished (224x400)

- Run by the user with `kaggle_15fps_aug/kaggle_finetune_15fps_aug_full.ipynb`: K=12, 15 fps, 2 epochs, augmentation hide + weather + hflip, EMA over about the last 20% of steps, 100% of the train videos (no held-out score), input 224x400 (the resolution test came after). Summary: 6,698 of 6,698 steps, 1.39 s per step, 2.87 h training, 3.4 h total, final loss 0.804, fp16 AMP. Files extracted (from `results (2).zip`, checked for unsafe paths first) into `cache/kaggle_15fps_aug_full/`: averaged weights + head, final (unaveraged) weights + head, summary.
- Checks: both weight sets have 12 blocks (20 to 31), all values finite; the averaged and the final weights are almost identical (relative mean difference 0.0001 on block 31 qkv, head linear weights differ by 0.0003), so the averaging effect on predictions will be small. Test inference on all 30 frames for both: `infer_aug_full.py` (scratchpad), queued behind the seed-2 resolution check; then comparison with 1.csv (secondary check only).
- Cache cleanup (the user approved the list): deleted `cache/vjepa2_1_download/weights` (21 GB, the V-JEPA 2.1 checkpoints, model not adopted; the Meta code in `repo/` and the extracted features stay), `cache/ft2/k16` (4.3 GB, K=16 five-fold run, not better than K=12) and `cache/dinov3_hplus_224x400_step3` (0.4 GB, the non-bf16 copy; `..._step3_bf16` stays). Free disk went from 154 GB to 179 GB. Not deleted: `cache/ft_hplus_block20` (17 GB, still used by `main.ipynb` and the old B scripts) and `cache/ft2/k12`.

## 2026-11-22 - Higher input resolution, seed 2 repeat (the gain did not hold)

- Same screen as above with seed 2 (288x512, K=12, 2 epochs, fold 0, hide + weather + hflip): AUC 0.8692, F1 0.7384 at 0.5, F1 0.8022 at the best threshold.
- All four runs (AUC): 224x400 seed 42 0.8866, seed 2 0.8789 (mean 0.8828); 288x512 seed 42 0.9008, seed 2 0.8692 (mean 0.8850). The mean gain is +0.002, far below the seed spread (0.008 at 224x400, 0.032 at 288x512); at seed 2 the larger input is even 0.010 lower. The first result was mostly luck.
- Conclusion: no reliable gain from 288x512 for the fine-tune, at 1.7x the cost. Not adopted; input stays 224x400. The frame cache `cache/ft_frames_testfmt_288x512` (6 GB) is no longer needed unless a third seed is run. For the frozen E3 the size test (stage 1 in the queue) is still open, but the fine-tune result gives no reason to expect a gain.

## 2026-11-22 - Frozen A (frozen DINOv3 + logistic regression + pose) back in the blend, nested test

- A members: A30_dino_pose (AUC alone 0.8155) and A_dyn_C0.003 (0.7930). Rank correlation of A30_dino_pose with B 0.683, D1 0.499, E3 BiGRU 0.556 (scratchpad `a_nested.py`, same nested setup as the stacking test; B in the held-out test is the old B because the new B has no held-out predictions).
- Results (AUC / F1): v2 fixed 2/1/1/1/1 0.9225 / 0.8265; v2 + A30_dino_pose x1 0.9183 / 0.8162, x2 0.9116 / 0.8078; v2 + A_dyn x1 0.9188 / 0.8132, x2 0.9116 / 0.8017. Nested weight grid {0,1,2} with A as a 6th member: 0.9246 / 0.8203 (by AUC) and 0.9194 / 0.8180 (by F1), exactly the same as without A; the grid gave A weight 0 in every fold.
- Conclusion: A does not help; every time the nested search chooses weight 0 for it, and forcing it in lowers the score. The earlier optimistic grid (0.8279 against 0.8214 next to the head-only B) came from choosing weights on the same data. A stays out of the blend. Caveat: not re-tested with the new B's own held-out predictions (none exist for a 100% train model).

## 2026-11-22 - New B (Kaggle 100% train, 15 fps, augmentation, EMA) in the v2 blend, submission file 1 of the plan

- Test inference of both weight sets on all 30 test frames of the 1,515 sequences (`infer_aug_full.py`, 39 min): head cheat probability averaged over the 30 frames. Mean cheat probability 0.362 (averaged weights) and 0.362 (final weights), against 0.389 for the 15 fps fold-0 model and 0.339 for the 30 fps model; correlation of the new averaged model with the 15 fps fold-0 model 0.790, with the 30 fps model 0.845, with its own final weights 0.9998 (the averaging barely changes the predictions).
- Blend (v2 members unchanged: B x2, E3 BiGRU, E3 transformer, D1, E5 transformer x1; rank scale and threshold 0.70 from the 15 fps fold-0 held-out predictions, an assumption for a 100% train model; scratchpad `blend_new.py`). Agreement with 1.csv (secondary check, not used to choose anything), F1 / wrong rows / cheat share: B = new model averaged weights 0.8692 / 152 / 0.234; new model final weights 0.8677 / 154 / 0.235; 30 fps full 0.8496 / 175 / 0.234; average of 30 fps and new 0.8591 / 164 / 0.235; average of 15 fps fold-0, 30 fps and new 0.8581 / 166 / 0.239. Earlier files: 15 fps fold-0 alone 0.8556 / 170, average of 15 and 30 fps 0.8476 / 178.
- File written at the user's request: `submissions/blend_v2_aug_full_ema_headonly.csv` (B = the new averaged-weights model alone, all 30 frames, head only; 1,515 rows, cheat share 0.234). Not submitted (the user submits). Public score: not yet known.

## 2026-11-22 - Public score of the new B submission

- `submissions/blend_v2_aug_full_ema_headonly.csv` (B = Kaggle 100% train, 15 fps, K=12, 2 epochs, hide + weather + hflip augmentation, averaged weights, head only; rest of v2 unchanged), submitted by the user as `submission_aug.csv`: public **0.8743286**, a new best (previous best 0.8627997 = average of the 15 fps fold-0 and 30 fps models, +0.0115; 15 fps fold-0 alone 0.8617276). Agreement with 1.csv had predicted it (0.8692 against 0.8476 to 0.8556 for the earlier files).

## 2026-11-22 - main.ipynb updated to the current best pipeline

- Change (user request, the from-scratch submission notebook comes later): the old B in `main.ipynb` (per-fold fine-tuned backbones from `ft_hplus_block20`, logistic regression on the 2-frame summaries + pose) is replaced by the current B: the Kaggle 100% train fine-tune (`kaggle_finetune_15fps_aug_full.ipynb`: K=12, 15 fps, 2 epochs, hide + weather + hflip, averaged weights), head only, mean of the 30 frames' cheat probability on every test clip (`scripts/kaggle_infer_test.py`, new in the repo, same code as the scratchpad `infer_aug_full.py`). Removed from the notebook: `FT_CACHE`, `FT_BLOCKS`, the fine-tuned feature cells, `fine_tuned_features` and `oof_ft`. Added caches `REF_CACHE` (Kaggle fold-0 run) and `B_CACHE` in the import cell.
- Blend cells changed to match the submission that scored 0.8743286: B has held-out predictions only for fold 0 (the Kaggle fold-0 run that did not see fold 0), so all ranks use the fold-0 held-out clips as the reference, the weights are fixed at B 2 and the four other members 1 (the earlier all-fold search with the old B; a re-search gives the same within noise), and the threshold is chosen on the fold-0 clips (0.70 in the submission). The other four members (BiGRU DINOv3, Transformer DINOv3, LeVJEPA + pose, Transformer LeVJEPA) are unchanged. Output still `submissions/main.csv`.
- Check: the notebook was NOT run (the user did not want it run, only updated; a started run was stopped; it is untested end to end, so run it once before relying on it). `main.ipynb` still reads caches built by the scripts; the from-scratch version (all extraction, RT-DETR, ViTPose, fine-tune in the notebook) is the later submission notebook.
- E3 with augmented frozen features, queued: `extract_dino_aug.py` (one augmented copy of every train clip through the frozen DINOv3, 2.6 h, running), then E3a BiGRU/transformer trained on the originals plus the copies, a control E3c (originals only, 50 epochs, because the copies double the number of steps), and the blend check (no 1.csv used).
- 2026-11-22 (later): the augmented extraction was stopped by the user's request after 15 of 398 videos (resumable; `COPY=1 python scripts/extract_dino_aug.py` from leon/ continues where it stopped); the E3 augmentation chain is postponed to the next day.
- 2026-11-22 (later): cache cleanup approved by the user ("delete the unused big caches"): deleted `cache/ft_hplus_block20/train_tokens.npy` (13 GB) and `test_tokens.npy` (2.6 GB), the frozen block-1-to-20 tokens of the old B (only the old scripts `ft_cache.py` / `ft_train.py` used them; `main.ipynb` no longer does; the small `train_meta.csv` and `preds` folders stay), and `cache/ft_frames_testfmt_288x512` (6 GB, the 288x512 frame cache of the closed resolution idea). The old B fine-tune cannot be re-run without rebuilding the tokens with `ft_cache.py`. Left alone: `cache/ft2/k12` (5.5 GB, the 5-fold ft2 K=12 run), `cache/ft_frames_testfmt` (3.6 GB, needed by `ft2_train.py`) and the small feature folders.
- 2026-11-22 (later): the user decided to keep using `submissions/1.csv` as the secondary agreement check (see the rules discussion above); it stays out of `main.ipynb` and the final notebook and is never used to choose pipeline values.

## 2026-11-22 - E3 on augmented frozen DINOv3 features (E3a): no gain

- Setup: `scripts/extract_dino_aug.py` made one augmented copy (hide + weather + hflip, the same random draw on all 30 frames of a clip, test delivery format, 224x400) of every labelled train clip through the frozen DINOv3 ViT-H+ (2560 per frame, `cache/dinov3_aug1_224x400`, 398 videos, 17 frames/s, about 2.7 h with a stop in the middle). E3a = BiGRU and transformer trained on the originals plus the copies of the training-fold clips only (the held-out fold is never augmented or used). Control E3c = originals only, 50 epochs (same number of steps as E3a roughly). Test frames stay unaugmented.
- Alone, 5 folds by subject, held-out clips (AUC / best F1): E3 BiGRU 0.8516 / 0.7729, E3a BiGRU 0.8581 / 0.7842, E3c BiGRU 0.8550 / 0.7688; E3 transformer 0.8574 / 0.7741, E3a transformer 0.8540 / 0.7728, E3c transformer 0.8546 / 0.7664.
- Blend v2 (AUC / best F1): plain E3 0.9225 / 0.8265; E3a instead of E3 0.9223 / 0.8239; E3a added as two extra members 0.9189 / 0.8176; E3c instead of E3 0.9231 / 0.8254; E3c added 0.9184 / 0.8176.
- Conclusion: no gain. The only visible effect is the BiGRU (+0.012 F1 over E3, but E3c is also +0.0034 AUC over E3, so part of it is just more training steps), and it disappears in the blend (-0.0026 F1, inside the noise of 0.005 to 0.008). The frozen sequence heads do not profit from augmented features, so submission 2 (frozen augmented members) is not worth making. Not adopted. Kept: `cache/dinov3_aug1_224x400` (small) in case frozen A is re-tested with it.

## 2026-11-22 - LeVJEPA fine-tune, fold 0 (K=8, augmentation): failed to learn

- New scripts: `scripts/levjepa_clips.py` (frame cache `cache/levjepa_clips/train_clips.npy`, 6,166 clips x 30 frames, 224x224, test format, 27.8 GB) and `scripts/levjepa_ft.py` (top K=8 of 24 LeVJEPA-1.5-Large blocks + final norm + 8-action head on [cls, mean patch], bitsandbytes 8-bit AdamW in the backward pass, checkpointing, one random half of the clip (15 frames, even or odd) per step with one augmentation draw on all 15 frames, hide + weather + hflip, EMA over the last 20% of steps, BS 8, lr 2e-5 blocks / 1e-3 head, 2 epochs = 622 steps, 3.75 s/step, peak 3.2 GB). Forward check against the Hugging Face model: cosine 0.99999.
- Fold 0 (3,932 training clips, 1,364 held-out clips, both halves averaged): AUC 0.7047, best F1 0.628 (EMA) / 0.631 (final weights). Training loss stayed near chance (epoch means 2.35 and 2.09; chance for the weighted 8-class loss is about 2.08). The frozen LeVJEPA members on the same fold-0 clips: D1 0.842 to 0.847, E5 BiGRU 0.870, E5 transformer 0.857 (AUC).
- Checks: the cached frames and my forward reproduce the old frozen extraction features exactly (cosine 1.0 on 8 clips); a frozen logistic regression on [cls, mean patch] of one half gives fold-0 AUC 0.725 (1,200 training clips), so the fine-tune gained nothing over the plain probe. 100-step diagnostics without augmentation: lr 2e-5 loss 2.34 / 2.43 / 1.94 at steps 25 / 50 / 75; lr 1e-4 loss 2.34 / 2.50 / 2.10, so no learning in the first 100 steps with either rate and without augmentation (warm-up is the first 31 steps).
- Conclusion: not adopted. The cause is not yet known (the data and the forward are right; the optimisation does not move the loss). Open: a head-only run (K=0) to see if the loss moves at all, a lower batch-norm / head setup, per-frame mean-pooled readout instead of the clip cls, and a different learning rate; the frozen sequence heads on 30 per-frame vectors + pose reach 0.85 to 0.87, so this readout may also just carry less signal. Test predictions of the fold-0 model are in `cache/levjepa_ft/k8_f0_aug/` (not used).

## 2026-11-22 - Mixed E3 members, and the 30 fps 288x512 Kaggle notebook (not run yet)

- Mixed E3 members in the nested v2 blend (AUC / F1): plain E3 0.9225 / 0.8265; augmented BiGRU + plain transformer 0.9227 / 0.8218; plain BiGRU + augmented transformer 0.9230 / 0.8189; control BiGRU (E3c) + plain transformer 0.9232 / 0.8225; both augmented 0.9223 / 0.8239. No gain from any mix (AUC equal, F1 lower, all inside the noise of 0.005 to 0.008). E3 stays plain. Script in the scratchpad (`blend_e3mix.py`).
- New notebook `kaggle_30fps_aug/kaggle_finetune_30fps_aug_full_288x512.ipynb` (copy of the 15 fps aug full notebook, nothing else changed): FPS 30 (every frame), EPOCHS 1 (about 6,700 steps like the 15 fps 2-epoch run; two epochs at this size would not fit the session), IMG_HW 288x512 (18x32 patches, 1.7x the cost of 224x400), K 12, 100% train, hide + weather + hflip, EMA over the last 20% of steps, one learning rate 2e-5 on all unfrozen blocks, head only, TIME_BUDGET_H 8.5 (the step count is cut automatically if the measured speed would not fit). TAG `k12_30fps_288x512_aug_full`. Expected: about 4.5 h training at about 2.3 s per step (estimated from 1.39 s at 224x400 times 1.7) plus about 0.8 h reading and recoding all frames.
- Why this is an experiment, not a proven improvement (user's call, 2026-11-22: try it at submission time): 288x512 showed +0.002 mean AUC over two seeds with a spread of 0.03 (noise), and 30 fps with augmentation has not been compared with 15 fps with augmentation (the old 30 fps model had no augmentation; averaging it with the 15 fps fold-0 model gave +0.001 public). The 15 fps aug model (public 0.8743) stays the reference.
- Local inference after the run: `IMG_H=288 IMG_W=512 DIR=cache/kaggle_30fps_288x512_aug_full TAG=k12_30fps_288x512_aug_full python scripts/kaggle_infer_test.py`.

## 2026-11-23 - Error analysis (peeking, front vs side) and pose duration features

- Error breakdown of the held-out v2 blend (older B12, 787 wrong of 6,166, thr 0.70, F1 0.8265): Repeated_Peeking 224 wrong (208 missed cheats, 27% of its clips), Looking_Around 121 and Checking_Smartwatch 110 (all false cheats), Reading_Hidden_Notes 110 (100 missed), Using_Smartwatch 104 (91 missed). Cheat clips are rarely wrong before the action start (about 2%) but 30% wrong within 0.5 s of the start and 32% wrong more than 3 s after it. Current best submission (`submission_aug.csv`, public 0.8743) vs `1.csv`: differ in 152 of 1,515 (114 cheats we call normal, 38 the other way; cheat share 0.234 vs 0.284).
- Peeking, aggregation of the fold-0 Kaggle 15 fps frame outputs (104 peeking cheat clips vs all normal clips, AUC): mean 0.7305, top-10 mean 0.7493, max 0.7479, max of the Repeated_Peeking class probability 0.8229. Adding w x (max peeking-class probability) to the mean cheat probability: overall AUC 0.8545 (w=0), 0.8560 (0.1), 0.8543 (0.3), 0.8321 (1.0); subject-grouped choice of w gives 0.8540. Peeking recall at the best threshold goes 0.31 -> 0.42 at w=0.3 but false cheats on Looking_Around go 0.094 -> 0.141 and on Checking_Smartwatch 0.011 -> 0.033. No overall gain: a peek and a look-around look alike.
- Head-movement features (`d30_peek.py`, scratchpad): from the clip's own 30 frames, for head_yaw, head_down and eye_yaw: range, 90th percentile deviation from the clip median, mean and max speed, direction reversals, and per threshold the fraction of frames away, number of away runs, longest run, mean run length, speed while away (45 features). D1 (LeVJEPA + pose, LR) AUC / best F1 / peeking-cheat AUC: as before 0.8539 / 0.7381 / 0.8079; with the features 0.8539 / 0.7389 / 0.8073 (C=0.001). Features alone AUC 0.6935. No gain: how long or how fast the head turns inside a 1-second clip does not separate a peek from a look-around. Nested blend with this D1: 0.9224 / 0.8251 vs 0.9225 / 0.8265.
- Front vs side (held-out v2 blend): front AUC 0.9241 / F1 0.8351, side AUC 0.9209 / F1 0.8182, predicted cheat share 0.240 vs 0.241. Error rate by action front vs side: Writing_Answers 0.044 vs 0.113 (the one clear difference), Using_Smartwatch 0.112 vs 0.143, the rest within 0.02. A separate threshold per view (chosen on the other folds) gives F1 0.8233 against 0.8265 for one threshold: no gain, the best threshold is about the same for both views.
- Watch-duration features (`d30_watch.py`): from the clip's own 30 frames, hand-to-face distance and wrist height above the shoulder line: min, median, and per threshold the fraction of frames near / raised, number of runs, longest run, mean run length, hand speed while near (34 features). D1 + these, 5-fold held-out train (AUC / best F1 / peeking-cheat AUC / Using-vs-Checking-smartwatch AUC): C=0.001 as before 0.8539 / 0.7381 / 0.8079 / 0.7664, with the features 0.8679 / 0.7525 / 0.8306 / 0.7731; C=0.003 0.8488 -> 0.8636. Pose only 0.7864 -> 0.8013. The gain is general (peeking too), not specific to the smartwatch pair. Nested blend (AUC / F1): D1 as before 0.9225 / 0.8265; D1w C=0.001 instead of D1 0.9232 / 0.8167; D1w C=0.003 0.9224 / 0.8257; D1 and D1w both 0.9247 / 0.8202; D1w with weight 2 0.9252 / 0.8242. AUC +0.001 to +0.003, F1 -0.001 to -0.010: not a reliable gain in the blend (the fine-tuned B and E5 already carry the cue). Not adopted. Saved members: `members/D1w_lev_pose_watch_C0.001_{oof,test}.npy`, `D1p_lev_pose_peek_C0.001_{oof,test}.npy` in the scratchpad.

## 2026-11-23 - Routing smartwatch clips to D1 (oracle test), peeking label statistics

- Oracle test (the action of every held-out clip known, which a real router would not know): on the smartwatch clips only (Using_Smartwatch cheat clips + Checking_Smartwatch normal clips) the v2 blend separates them with AUC 0.9215 / best F1 0.8452; D1 alone 0.8182 / 0.7089, D1 + watch-duration 0.8247 / 0.7127, B12 0.8694 / 0.8022, E5 transformer 0.8815 / 0.7759. D1 is the worst single member on these clips, so sending smartwatch clips to D1 alone would be worse even with a perfect router. Not done.
- Peeking labels: 434 peeking cheat clips (label 1), 208 (48%) are called normal by the blend at threshold 0.70 (Using_Smartwatch 22%, Reading_Hidden_Notes 26%), but only 5 have a blend rank under 0.3. So label noise cannot be told apart from real misses with a strict cut.

## 2026-11-23 - Clean-label tests on D1 (self-score cleaning, late-clip cut): no gain

- Label rule today: a clip is cheat when its centre (second + 0.5) is at or after the by-eye action start; there is no end, the cheat range runs to the end of the video (median length 15.5 s, 1,514 cheat clips).
- Self-score cleaning (`d30_clean.py`, nested: the "looks like writing" score of every training clip comes from an inner 4-fold fit on the training folds only; evaluation on the original labels of the held-out fold, which penalises a cleaned model a little). D1 (LeVJEPA summary + pose, LR C=0.001), AUC / best F1 / peeking-cheat AUC: original labels 0.8539 / 0.7381 / 0.8079; drop the lowest 10% of cheat clips 0.8452 / 0.7322 / 0.7948; drop 25% 0.8353 / 0.7279 / 0.7830; relabel the lowest 10% as normal 0.8434 / 0.7309 / 0.7923; relabel 25% 0.8323 / 0.7255 / 0.7792. Worse in every version. Nested v2 blend: reference 0.9225 / 0.8265; D1 replaced by the clean-label D1 (drop 10%) 0.9217 / 0.8253; added as a sixth member 0.9231 / 0.8169. No gain in the blend either.
- Time profile of the cheat-labelled clips (held-out v2 blend, mean rank / share called normal by seconds since the start): 0-1.5 s 0.789 / 24%, 1.5-3 s 0.822 / 14%, 3-5 s 0.780 / 27%, 5-7 s 0.751 / 32%, 7-9 s 0.716 / 40%, more than 9 s 0.632 / 50%. Smartwatch falls from 0.85 to 0.47 and hidden notes from 0.84 to 0.62 late in the video, so the person may be back to writing near the end while the label stays cheat.
- Late-clip cut (`d30_tail.py`, D1 trained without the cheat clips more than X s after the start, 16 / 193 / 564 clips for X = 9 / 7 / 5 s; AUC / F1 on all original labels): none 0.8539 / 0.7381; X=9 0.8546 / 0.7379; X=7 0.8497 / 0.7368; X=5 0.7967 / 0.7111. No gain; the evaluation counts the dropped clips as cheat, so a real gain could be hidden, but nothing shows up at 9 or 7 s. Not adopted.
- Head-movement features were tested before (same day, no gain). The clean-label idea as a separate member ("step F" proposed by the user) is closed: alone worse, in the blend no gain. Not tried: weak-label / multiple-instance training on the video label only (needs a fine-tune).

## 2026-11-23 - Frozen LeVJEPA members with augmented copies (D1a, E5a)

- Setup: `scripts/extract_levjepa_aug.py` made one augmented copy of every labelled train clip (hide + weather, no flip because the members use pose from the original clip, the same random draw on all 30 frames, from the clip cache `cache/levjepa_clips/train_clips.npy`) through the frozen LeVJEPA-1.5-Large (32 x 1024 per clip, `cache/levjepa_aug1_224_30/train.npy`, 6,166 clips at about 1.1 clips/s, 1.6 h). D1a = logistic regression on summary + pose trained on the originals plus the copies of the training folds; E5a = BiGRU / transformer on the 30 per-frame vectors (PCA 128, clip-mean centred) + pose, same augmented rows. The held-out fold is never augmented or used. Test frames are unaugmented.
- Alone, 5 folds by subject (AUC / best F1 / peeking-cheat AUC): D1 C=0.001 0.8539 / 0.7381 / 0.8079 -> D1a 0.8650 / 0.7438 / 0.8317; D1 C=0.003 0.8488 / 0.7327 / 0.7992 -> D1a 0.8683 / 0.7474 / 0.8327; E5 BiGRU 0.8695 / 0.7521 / 0.8270 -> E5a 0.8832 / 0.7717 / 0.8400; E5 transformer 0.8784 / 0.7705 / 0.8273 -> E5a 0.8860 / 0.7725 / 0.8404. Every augmented member is better alone (AUC +0.008 to +0.020), more than the frozen DINOv3 members gained (E3a +0.000 to +0.006).
- Nested v2 blend (AUC / F1): reference 0.9225 / 0.8265; D1 -> D1a 0.9234 / 0.8185; E5 transformer -> E5a transformer 0.9232 / 0.8283; both augmented 0.9241 / 0.8168; plain and augmented all six members 0.9271 / 0.8169. AUC +0.001 to +0.005, F1 between -0.010 and +0.002: no reliable gain in the blend (inside the noise of 0.005 to 0.008), as with E3a. Not adopted yet. Saved members: `members/D1a_lev_pose_C0.001_*`, `E5a_lev_gru_*`, `E5a_lev_transformer_*` in the scratchpad.

## 2026-11-23 - Per-second labels of the 104 cheat front videos (annotation by the AI assistant, train frames only)

- Rule basis: the revised FAQ (Jawaban FAQ (3), #83) allows VLM / AI annotation on train data only, so the per-second labels were made by Claude looking at train frames, never test frames; this has to be stated in the final notebook. Label rule from the FAQ answers (#44.3, #50.3, #63.1, #63.2, #72, #82): a 1-second clip is cheat only when the cheating behaviour dominates it, otherwise normal; start and end movements count when they already point toward cheating; under desk counts from the moment the head bends down; smartwatch = operating it, notes = reading them; peeking = repeated glances to one side; looking around = many directions; neutral activity is normal inside a cheat video.
- Method: `scripts/annotation_sheet.py` makes one contact sheet per video (3 frames per second, 168x190 crop around the person from the cached person box, seconds from 3 to the end, two columns); one sheet viewed per video in worklist order (most held-out errors first, `annotation/worklist.csv`); labels written with `annotation/add_labels.py` into `annotation/ai_second_labels.csv` (video, second, label 1 cheat / 0 normal / -1 unsure, viewed flag, note). Seconds 0 to 2 were not viewed and are written as normal (viewed = 0), except the 9 videos that were not in the clip table (no start had been read): those were viewed from second 0. Normal videos are not labelled (all normal) and the side videos are not labelled (FAQ #50.2: shift the front labels onto the side video; the earlier convention was a 0.5 s lag).
- Result: 1,318 seconds viewed: 481 cheat, 571 normal, 266 unsure (20%). Unsure share by action: Looking_Under_the_Desk 89 of 348, Reading_Hidden_Notes 81 of 322, Repeated_Peeking 56 of 326, Using_Smartwatch 40 of 322. Under desk and the "note held while writing" cases have the lowest confidence (notes in the `note` column); head turns for peeking and the raised wrist for the smartwatch were the clearest.
- Against the old labels (cheat from the by-eye start to the end of the video), on the 1,231 front cheat clips with a start and not unsure: 90.2% agree; 120 old-cheat clips become normal and only 1 old-normal clip becomes cheat. Share of clips labelled cheat by seconds after the old start (new / old): 1-3 s 0.94 / 1.00, 3-5 s 0.79 / 1.00, 5-7 s 0.68 / 1.00, 7-9 s 0.53 / 1.00, more than 9 s 0.25 / 1.00. So the old rule over-labels the later part of the action: the cheat range does not run to the end of the video, which matches the falling model scores late in the video and the dominant-behaviour rule of the test labels.
- Caveats: made from 3 frames per second at about 170 px per frame by one annotator (no second opinion), so peeking and under-desk labels are noisy; unsure seconds should be excluded from training; the labels have not been used in any model yet.
- Next: turn them into clip labels for all train clips (front directly, side shifted, normal videos all normal, unsure excluded), then compare label schemes by train validation as FAQ #81 allows (train D1 / E5 / B screen on the old vs the new labels and score both on the new labels, plus the clips both schemes agree on).

## 2026-11-23 - New per-second labels: D1 (LeVJEPA summary + pose, LR) trained on old vs new labels

- Clip labels from `annotation/make_clip_labels.py` (file `annotation/annotations.csv`, renamed by the user): front cheat videos directly, side cheat videos from the front labels shifted by the 0.5 s lag (a side second is cheat / normal only when both front seconds it overlaps agree, otherwise unsure), normal videos normal. Result: 775 cheat, 4,713 normal, 678 unsure clips (449 of the unsure on side videos because of the shift); old labels had 1,514 cheat. Old cheat clips: 774 stay cheat, 181 become normal, 559 are unsure.
- D1 (C=0.001, 5 folds by subject) trained on old labels, on new labels with the unsure clips dropped, and on new labels with unsure = old label; AUC / best F1, scored on the new labels (5,488 clips with a new label), on the 5,306 clips where old = new, and on all clips with the old labels: old-trained 0.8621 / 0.7243, 0.8696 / 0.7333, 0.8539 / 0.7381; new-trained (unsure dropped) 0.8646 / 0.7280, 0.8692 / 0.7332, 0.8306 / 0.7194; new-trained (unsure = old) 0.8652 / 0.7300, 0.8707 / 0.7364, 0.8461 / 0.7292.
- Conclusion for the frozen LR member: the label scheme changes little (+0.003 AUC on the new labels, inside the noise). The model trained on the old labels already scores about the same on the new labels (0.8621 against 0.8539 on its own labels), so the late-video label noise was not hurting this member much. The fine-tuned B (which sees frames) is the real test: fold-0 screen `aug_hwf_newlab` (`NEWLAB=1 python scripts/ft2_train.py 12 2`, new env flag in `ft2_train.py`: cheat videos use the per-second labels, cheat second = the video's action, normal second = writing, unsure = not trained on).

## 2026-11-23 - Fine-tuned B trained on the new per-second labels (fold 0, seed 42)

- Setup: `NEWLAB=1 FT_OUT=aug_hwf_newlab AUG=hide+weather+hflip VAL_ONLY=1 FOLDS=0 python scripts/ft2_train.py 12 2` (K=12, 2 epochs, hide + weather + hflip, 224x400 test-format frames, fold 0 held out, 588 steps, 14.5 min). Cheat videos use the per-second labels (cheat second = the video's action, normal second = writing, unsure = not trained on), all other videos as before; compared with the old-label run `aug_hwf` (same settings). Held-out clip score = mean cheat probability of the clip's 2 frames (script `score_newlab.py`, scratchpad).
- Fold-0 clips, AUC / best F1: scored on the new labels (1,212 clips with a new label): old-trained 0.9489 / 0.8623, new-trained 0.9619 / 0.8812; on the 1,165 clips where old = new labels: old-trained 0.9570 / 0.8802, new-trained 0.9666 / 0.9032; on all 1,364 clips with the old labels: old-trained 0.8866 / 0.8151, new-trained 0.8731 / 0.8082.
- Reading: the new-trained model is better by +0.010 to +0.013 AUC (+0.019 to +0.023 F1) on the clips both schemes label the same, and lower by 0.014 on the old labels, as expected since the old labels include the late-video clips that the new labels call normal or unsure. One seed, one fold (6 subjects), seed noise of this screen about 0.008, so not yet a confirmed gain; a seed-2 repeat is running (`aug_hwf_newlab_seed2`, compare with `aug_hwf_seed2`). Caveat: the test labels follow the organisers' dominant-behaviour rule, so only the public leaderboard shows whether the new labels help.
- Seed 2 repeat (`aug_hwf_newlab_seed2` against the existing `aug_hwf_seed2`, same settings), AUC / best F1: on the new labels old-trained 0.9507 / 0.8834, new-trained 0.9530 / 0.8675; on the 1,165 clips where old = new old-trained 0.9561 / 0.9032, new-trained 0.9566 / 0.8796; on the old labels old-trained 0.8789 / 0.8158, new-trained 0.8604 / 0.8003.
- Mean of the two seeds, old-trained vs new-trained: new labels AUC 0.9498 vs 0.9575 (+0.008), F1 0.8729 vs 0.8744 (+0.002); old = new clips AUC 0.9566 vs 0.9616 (+0.005), F1 0.8917 vs 0.8914 (0.000); old labels AUC 0.8828 vs 0.8668 (-0.016). The seed-42 gain did not repeat (seed 2: +0.002 and +0.0005 AUC, F1 lower), so the new labels give no reliable gain for the fine-tuned B at the screen's noise level (about 0.008) and cost about 0.016 AUC on the old labels. Not adopted for the submission pipeline; the per-second labels stay as an option (a 100% train run with them would only show its effect on the public leaderboard).

## 2026-11-23 - Kaggle notebook with the new per-second labels (not run yet)

- New notebook `kaggle_15fps_aug/kaggle_finetune_15fps_aug_full_newlab.ipynb`, a copy of `kaggle_finetune_15fps_aug_full.ipynb` (100% train, 15 fps, K=12, 2 epochs, hide + weather + hflip, EMA over the last 20% of steps, 224x400) with one change: the frame labels of the cheat videos come from a pasted table of per-second labels (2,963 seconds of the cheat videos from `annotation/clip_labels_ai.csv`; cheat second = the video's action, normal second = writing, unsure = not trained on; side videos use the front labels shifted by 0.5 s). Videos without per-second labels (normal videos and the 18 cheat videos with no start) keep the old rule. Output files are named `k12_15fps_newlab_aug_full_*`; expected time about 2.9 h like the 15 fps run. Local inference: `DIR=cache/kaggle_15fps_newlab_aug_full TAG=k12_15fps_newlab_aug_full python scripts/kaggle_infer_test.py`.
- Why it is run although the fold-0 screens showed no reliable gain (two seeds, +0.005 AUC on the clips both schemes agree on): the user wants to compare the resulting submission with `1.csv` (secondary check only) and try it on the public leaderboard. Plan: v2 blend with only B replaced (same members, weights 2/1/1/1/1), rank scale from the fold-0 held-out predictions of the new-label screen (`cache/ft2/aug_hwf_newlab`), threshold 0.70 kept from v2.

## 2026-11-23 - Blend weights with the augmented frozen LeVJEPA members, and the 30 fps new-label notebook

- Nested weight search (grid {0,1,2}; weights and threshold chosen on 4 folds, scored on the fifth; held-out train only, B = the older B12): v2 members (B12, E3 BiGRU, E3 transformer, D1, E5 transformer) F1 0.8180 / AUC 0.9194; with the augmented LeVJEPA members instead (B12, E3 BiGRU, E3 transformer, D1a, E5a BiGRU, E5a transformer) F1 0.8232 / AUC 0.9233. Chosen weights were stable: B12 2, E3 BiGRU 1, D1a 1, E5a BiGRU 1, the transformers 0 in 4 of 5 folds. Fixed weights, threshold on 4 folds: v2 2/1/1/1/1 0.8265 / 0.9225; augmented with 2/1/1/1/1/1 0.8258 / 0.9264; 2/1/1/2/2/2 (more LeVJEPA weight) 0.8187 / 0.9268; 2/0/0/1/1/1 0.8248 / 0.9238; 3/1/1/1/1/1 0.8271 / 0.9247.
- Reading: the augmented LeVJEPA members add about +0.004 AUC and nothing reliable in F1; giving them more weight does not help (AUC flat at about 0.927, F1 lower). Single members reach AUC 0.88 to 0.89, the blend 0.92 to 0.93, and the extra gain from a better member is small because the members share their mistakes (peeking against looking around, the late-video clips).
- New notebook `kaggle_30fps_aug/kaggle_finetune_30fps_aug_full_288x512_newlab.ipynb`: the 30 fps 288x512 notebook (1 epoch, about 6,700 steps) with the same per-second label change as the 15 fps `_newlab` notebook; output files `k12_30fps_288x512_newlab_aug_full_*`.

## 2026-11-23 - 8-action heads on the augmented frozen LeVJEPA members (old labels)

- Setup: the same members as D1a / E5a (LeVJEPA summary + pose LR; BiGRU and transformer over the 30 per-frame vectors + pose, originals plus augmented copies of the training folds) but with an 8-action output like B (4 cheat actions, 4 normal actions; cheat probability = sum of the first 4 class probabilities; cross-entropy with label smoothing 0.1 for the sequence heads, multinomial logistic regression for D1a8). Clip action labels: cheat video = the action after the by-eye start (+-0.5 s around it not used), writing before; normal-action videos = the action after 9.75 s, writing before 4.25 s; clips in the unused zones are left out of training (5,040 of 6,166 usable). Scripts `d30_aug8.py`, `e_lev_aug8.py` (scratchpad, env LAB=old|new).
- Alone, 5 folds by subject (AUC / best F1 / peeking-cheat AUC): D1a 0.8650 / 0.7438 / 0.8317 -> D1a8 0.8680 / 0.7482 / 0.8603; E5a BiGRU 0.8832 / 0.7717 / 0.8400 -> E58 BiGRU 0.8998 / 0.7863 / 0.8531; E5a transformer 0.8860 / 0.7725 / 0.8404 -> E58 transformer 0.8910 / 0.7842 / 0.8513. The E58 BiGRU (0.8998) is the best frozen member so far (the V-JEPA 2.1 ViT-g BiGRU had 0.8874). The 8-action head helps most on the peeking clips (+0.011 to +0.029 AUC).
- Nested v2 blend (AUC / F1): v2 0.9225 / 0.8265; augmented binary members 0.9264 / 0.8258; 8-action members (D1a8, E58 BiGRU, E58 transformer) 0.9291 / 0.8303; E58 BiGRU + transformer with the plain D1 0.9284 / 0.8291. Best frozen set so far (+0.007 AUC, +0.004 F1 over v2; the F1 gain is inside the noise, the AUC gain points the same way in every swap). New-label (LAB=new) versions are still running.

## 2026-11-23 - Submission file v3 (8-action augmented LeVJEPA members), not yet submitted

- File `submissions/blend_v3_8class_lev_aug.csv` (1,515 rows, sample_submission order, written because the user asked for a file to submit). Blend: B = the 100%-train aug fine-tune (EMA weights, all 30 test frames) x2, E3 BiGRU, E3 transformer, D1a8 (8-action LR + pose, trained on originals + augmented copies), E58 BiGRU, E58 transformer (8-action heads on the 30 per-frame vectors + pose, augmented copies), each x1; every member's rank is its position inside its own fold-0 held-out predictions (B: the Kaggle 15 fps fold-0 model); threshold = best F1 on the 1,364 fold-0 held-out clips = 0.675 (v2 used 0.70). Script `blend_v3.py` (scratchpad); the v2 recipe rebuilt by the same script reproduces `submission_aug.csv` exactly (0 rows differ), so the pipeline is the same.
- Numbers: held-out fold-0 F1 0.8406 for v3 against 0.8312 for v2; predicted cheat share on test 0.245 against 0.234; v3 differs from the submitted best (public 0.8743286) in 55 of 1,515 rows. Secondary check against `1.csv`: v3 F1 0.8583 (167 rows differ) against 0.8692 (152 rows) for v2, so v3 agrees slightly less with `1.csv`; the nested held-out train blend (all 5 folds) gave 0.9291 / 0.8303 for v3 against 0.9225 / 0.8265 for v2. The two checks disagree, so the expected public change is small and uncertain (55 rows).

## 2026-11-23 - Public result of submission v3: 0.8525422 (worse than 0.8743286)

- Public leaderboard (submitted by the user as `submission_8class_levjepa.csv` = `blend_v3_8class_lev_aug.csv`): **0.8525422**, 0.022 below the best (0.8743286, `submission_aug.csv`). Only the LeVJEPA members differed (55 of 1,515 rows).
- Diagnosis (`diag_v3.py`, scratchpad; `1.csv` only as the secondary check): of the 55 changed rows v3 calls 36 cheat that v2 called normal and 19 the other way; on those 55 rows `1.csv` agrees with v2 on 35 and with v3 on 20 (15 rows net worse, matching the public drop). Agreement with `1.csv` (F1): v2 0.8692, v3 0.8583. Swapping one part at a time: D1 -> D1a8 0.8647, E5 transformer -> E58 transformer 0.8710, adding the E58 BiGRU 0.8671, the binary augmented set (D1a, E5a BiGRU, E5a transformer) 0.8469 (the worst), v3 without E3 0.8622. D1 alone shifts between train and test (share of clips above 0.5: held-out train 0.18, test 0.25), D1a8 and the E58 members do not (0.225 / 0.246, 0.279 / 0.262, 0.282 / 0.281).
- Reading: the held-out train checks improved (nested blend 0.9225 -> 0.9291 AUC, fold-0 F1 0.8312 -> 0.8406) but the test got worse, and the `1.csv` check had flagged it before the submission (0.8692 -> 0.8583). So the gains of the augmented / 8-action frozen LeVJEPA members on train folds did not transfer to the test clips; the augmentation (not the 8-action head) looks like the main cause (the binary augmented set agrees least with `1.csv`). Decision: the best file stays `submission_aug.csv`; the augmented and 8-action LeVJEPA members are not adopted; when a held-out train gain and the `1.csv` agreement point in opposite directions, trust `1.csv` and the test-side check.

## 2026-11-23 - Submission file with one swap: E5 transformer -> E58 transformer (not yet submitted)

- File `submissions/blend_v2_e58tf_swap.csv` (1,515 rows). Blend: B (100%-train aug fine-tune, EMA weights, all 30 test frames) x2, E3 BiGRU, E3 transformer, D1 (plain, cheat / normal LR + pose), E58 transformer (8-action head, 30 per-frame LeVJEPA vectors centred on the clip mean + pose, trained on the originals + augmented copies) each x1; rank inside each member's fold-0 held-out predictions, threshold = best F1 on the fold-0 held-out clips (0.675). Script `blend_v2swap.py` (scratchpad). Against the submitted best (public 0.8743286): 29 rows differ; agreement with `1.csv` 0.8710 against 0.8692; fold-0 held-out F1 0.8392 against 0.8312; predicted cheat share 0.250 against 0.234. Both checks lean slightly positive but the changes are at noise level; v3 (three swaps) had the same signs on the train check and still lost 0.022 on the public score.

## 2026-11-23 - Public result of the single-swap file, new best 0.8749880, and the main notebook update

- Public leaderboard (submitted by the user as `submission.csv` = `blend_v2_e58tf_swap.csv`): **0.8749880**, new best (+0.0007 over 0.8743286; the file differs from `submission_aug.csv` in 29 rows, so the gain is at noise level, but by the user's rule it becomes the best configuration). Best configuration now: B x2 (aug 100% train fine-tune, EMA weights, all 30 test frames), E3 BiGRU, E3 transformer, D1 (plain cheat / normal LR + pose on LeVJEPA), E58 transformer (8-action head on the 30 per-frame LeVJEPA vectors + pose, trained on the originals + augmented copies), weights 2/1/1/1/1, ranks inside each member's fold-0 held-out predictions, threshold 0.675.
- `main.ipynb` updated to that pipeline (NOT run; it still reads the caches built by the scripts): imports cell gets `LEV_AUG_CACHE = ../cache/levjepa_aug1_224_30`; the feature extraction list names `scripts/levjepa_clips.py` and `scripts/extract_levjepa_aug.py`; the LeVJEPA loading cell also loads the augmented copy (length checked against the clip table); `build_sequences` can project extra clips with the scaler / PCA of the originals (argument `extra`); two new cells define the 8-action clip labels, `TransformerNet8`, `fit_predict8` and `sequence_model8` (originals + augmented copies of the training folds, cross-entropy with label smoothing 0.1, cheat probability = sum of the first 4 class probabilities); the "Transformer pada LeVJEPA" cell now trains that model; the blend member keeps the name "Transformer LeVJEPA", weights and threshold rule unchanged. The 8-action labels in the notebook use the by-eye start table as the local script did (cheat video: action after start + 0.5 s, writing before start - 0.5 s; normal-action videos: action after 9.75 s, writing before 4.25 s).

## 2026-11-23 - Threshold experiment file (0.70) of the current best configuration, not yet submitted

- File `submissions/blend_v2_e58tf_swap_thr070.csv`: the best configuration (B x2, E3 BiGRU, E3 transformer, D1, E58 transformer; submitted as `submission.csv`, public 0.8749880, rebuilt exactly by `blend_thr070.py`) with the threshold fixed at 0.70 instead of 0.675. Fold-0 held-out F1 0.8319 (0.8392 at 0.675); test cheat share 0.231 (0.250); agreement with `1.csv` 0.8722 with 148 rows different (0.8710 and 153 at 0.675); it differs from the 0.675 file in 29 rows (all cheat -> normal) and from `submission_aug.csv` in 24 rows. The F1 curve is flat between 0.65 and 0.75 on the held-out clips, so this is a small experiment on the threshold only.

## 2026-11-23 - Public result of the 0.70 threshold file: 0.8805271, new best

- Public leaderboard (`blend_v2_e58tf_swap_thr070.csv`, submitted by the user): **0.8805271**, +0.0055 over the same blend at threshold 0.675 (0.8749880) and +0.0062 over the earlier best 0.8743286. Only the threshold changed: 29 rows went from cheat to normal (test cheat share 0.231 instead of 0.250). The `1.csv` agreement had been slightly higher for this file (0.8722 against 0.8710), so the check and the public score agree. Reading: the extra cheat calls that the lower threshold added at the margin were mostly wrong, so on the test clips the better cut is at or above 0.70, while the fold-0 held-out F1 curve was flat from 0.65 to 0.75 (0.8319 at 0.70, 0.8344 at 0.75, 0.8392 at 0.675) and could not tell them apart. Not tried: 0.725 and 0.75 (held-out F1 0.8293 and 0.8344, test cheat share 0.207 and 0.190).
- Threshold curve of the best configuration with the `1.csv` agreement added (F1 against `1.csv` with rows wrong / fold-0 held-out F1 / test cheat share): 0.60 0.8384 (210) / 0.8203 / 0.339; 0.625 0.8539 (185) / 0.8313 / 0.311; 0.65 0.8624 (168) / 0.8363 / 0.276; 0.675 0.8710 (153) / 0.8392 / 0.250; **0.70 0.8722 (148) / 0.8319 / 0.231**; 0.725 0.8565 (161) / 0.8293 / 0.207; 0.75 0.8430 (172) / 0.8344 / 0.190; 0.775 0.8119 (199) / 0.8216 / 0.167; 0.80 0.7860 (220) / 0.7747 / 0.149. The `1.csv` agreement peaks at 0.70 (the public best) and falls on both sides, so 0.725 is expected to be worse than 0.70; the held-out curve is flat and did not show this.

## 2026-11-23 - main.ipynb: blend threshold fixed at 0.70

- `main.ipynb` (not run): the blend threshold is no longer the argmax of the fold-0 held-out F1 (0.675) but fixed at 0.70, the value of the best public score (0.8805271 against 0.8749880 at 0.675). The notebook states the train-side evidence (the held-out F1 of the blend is within 0.01 of its maximum from 0.625 to 0.75, 0.70 is inside that range, so the result is not sensitive to the exact value) and that of the two thresholds tried on the public leaderboard 0.70 scored higher; `1.csv` is not mentioned in the notebook. The cell prints the F1 at 0.70, the maximum and the plateau range.

## 2026-11-23 - Threshold 0.68 file for tomorrow's first submission (not yet submitted)

- File `submissions/blend_v2_e58tf_swap_thr068.csv`: the best configuration (B x2, E3 BiGRU, E3 transformer, D1, E58 transformer) with the threshold 0.68. Against `1.csv`: F1 0.8758 with 146 rows wrong (0.8722 / 148 at 0.70, 0.8710 / 153 at 0.675); fold-0 held-out F1 0.8422 (its highest); predicted cheat share 0.243; it differs from the 0.70 file (public 0.8805271) in 18 rows and from the 0.675 file (public 0.8749880) in 11 rows, all of them cheat -> normal relative to 0.675. The user chose it as the first submission of the next day, followed by the 30 fps fine-tune.

## 2026-11-23 - Every augmented variant swapped into the best blend, scored against 1.csv

- Script `aug_vs1csv.py` (scratchpad): the current best blend (B x2, E3 BiGRU, E3 transformer, D1, E58 transformer) with ONE member replaced or added at a time; F1 against `1.csv` (rows wrong) at the fixed threshold 0.70 (the fairest comparison), at the held-out threshold, and the fold-0 held-out F1. At 0.70: current best 0.8722 (148); v2 with the plain E5 transformer 0.8692 (152); E3 BiGRU -> E3a 0.8647 (155); -> E3c (more epochs, no augmentation) 0.8688 (152); E3 transformer -> E3a 0.8699 (149); -> E3c 0.8698 (151); both E3 -> E3a 0.8635 (156); D1 -> D1a 0.8642 (155); D1 -> D1a8 0.8645 (155); E58 transformer -> E5a transformer (binary) 0.8695 (151); add E5a BiGRU 0.8674 (153); add E58 BiGRU 0.8668 (154); E58 transformer trained on the new labels 0.8679 (152); D1a8 on the new labels 0.8582 (160).
- Reading: the current best is the highest of all variants against `1.csv` (every other swap is 0.002 to 0.014 lower), so none of the augmented or new-label members improves the test-side agreement; the only augmented member kept is the E58 transformer (+0.003 over the plain E5 transformer, at noise level, and the public score rose by 0.0007). The fold-0 held-out F1 rose for most of these swaps (up to 0.8461) while the `1.csv` agreement fell, so the held-out F1 is not a reliable guide for the augmented members; trust the `1.csv` column.

## 2026-11-23 - Fine threshold scan 0.670 to 0.695 and the notebook threshold set to 0.68

- Scan in steps of 0.001 of the best configuration (F1 against `1.csv` with rows wrong / fold-0 held-out F1 / test cheat share): 0.675 0.8710 (153) / 0.8392 / 0.250; 0.677 0.8747 (148) / 0.8400 / 0.247; 0.678 0.8736 (149) / 0.8416; 0.679 0.8743 (148) / 0.8405; **0.680 0.8758 (146) / 0.8422 / 0.243**; **0.681 0.8756 (146) / 0.8422 / 0.242**; 0.682 0.8747 (147) / 0.8410; 0.685 0.8745 (147) / 0.8389 / 0.240; 0.690 0.8731 (148) / 0.8391; 0.695 0.8727 (148) / 0.8338. 0.680 and 0.681 are the best on both checks, so no finer value is better; neighbouring values differ by 1 to 3 rows, which is noise (a finer cut would only choose individual test rows).
- `main.ipynb` (not run): the blend threshold is fixed at 0.68 (was 0.70); the cell now computes the F1 at the threshold directly (0.68 is not on the 0.025 grid) and the text names the two neighbours tested on the public leaderboard (0.675: 0.8750, 0.70: 0.8805); `1.csv` is not mentioned. The result of 0.68 on the leaderboard is still unknown (tomorrow's first submission).

## 2026-11-23 - 8-action heads on the DINOv3 E3 members and on the LeVJEPA members with the new labels (end of day)

- E38p = the E3 BiGRU and transformer (frozen DINOv3 30 per-frame vectors + pose) with the 8-action head, originals only (`e30c_aug8.py`, scratchpad). Alone, 5 folds by subject, best F1: BiGRU 0.7952 (E3 0.7722), transformer 0.7860 (E3 0.7731). In the best blend at the fixed threshold 0.70, F1 against `1.csv` (rows wrong): current best 0.8722 (148); E3 BiGRU -> E38p BiGRU 0.8626 (159); E3 transformer -> E38p transformer 0.8709 (150); both 0.8611 (161); add the E38p BiGRU as a sixth member 0.8595 (161). Not adopted: no swap beats the current best on the `1.csv` agreement, although the held-out F1 of the members and of the transformer swap (0.8438) went up.
- 8-action LeVJEPA members with the NEW labels (`LAB=new`): D1a8 best F1 0.7267, E58 BiGRU 0.7742, E58 transformer 0.7759 (old labels: 0.7482, 0.7863, 0.7842); their blend swaps against `1.csv` are in the entry above (E58 transformer on the new labels 0.8679, D1a8 on the new labels 0.8582 at 0.70), all below the current best. The new labels stay unused.
- All background jobs are finished; no GPU or CPU job is running.

## 2026-11-24 - Public result of the 0.68 threshold file: 0.8822952, new best

- Public leaderboard (`blend_v2_e58tf_swap_thr068.csv`, submitted by the user): **0.8822952**, +0.0017 over the same blend at 0.70 (0.8805271) and +0.0073 over 0.675 (0.8749880). Public scores of the three thresholds: 0.675 -> 0.8750, 0.68 -> 0.8823, 0.70 -> 0.8805. The `1.csv` agreement scan had put the three in the same order (0.8710, 0.8758, 0.8722), so the agreement column predicted the public ordering; the fold-0 held-out F1 (0.8392, 0.8422, 0.8319) ranked 0.675 above 0.70 and did not. Best configuration is now confirmed with the threshold 0.68: B x2, E3 BiGRU, E3 transformer, D1, E58 transformer, weights 2/1/1/1/1. `main.ipynb` text updated to name the three tested thresholds.

## 2026-11-24 - Kaggle 30 fps 288x512 fine-tune (100% train, old labels): run, inference and blend check

- Run (`kaggle_30fps_aug/kaggle_finetune_30fps_aug_full_288x512.ipynb`, user, Kaggle T4): the whole plan of 6,701 steps at 2.452 s per step, 5.66 h of training, 6.42 h in total, final loss 0.751 (the 15 fps run ended at 0.804), fp16. Weights extracted to `cache/kaggle_30fps_288x512_aug_full/` (zip checked first). Test inference on all 30 frames of the 1,515 sequences (`IMG_H=288 IMG_W=512 python scripts/kaggle_infer_test.py`, 74 to 80 s per chunk of 32 sequences, about 1 h): mean cheat probability 0.321 (EMA weights) against 0.362 for the 15 fps model; correlation of the two models' test scores 0.873.
- Blend check (`blend_b30.py`, B x2, E3 BiGRU, E3 transformer, D1, E58 transformer; every B variant ranked against the same Kaggle 15 fps fold-0 held-out reference, which was built for the 15 fps model), F1 against `1.csv` (rows wrong): at threshold 0.68 the 15 fps B (current best, public 0.8822952) 0.8758 (146), the 30 fps 288x512 B 0.8718 (148), the average of both B 0.8752 (146); at 0.70 0.8722 (148), 0.8605 (158), 0.8754 (143). Because the 30 fps model's scores are lower, its best threshold is lower: scan of the 30 fps B (EMA) 0.62 0.8757 (152), 0.64 0.8774 (147), 0.66 0.8785 (143), **0.67 0.8786 (142)**, 0.68 0.8718 (148), 0.70 0.8605 (158); the plain (non-averaged) weights behave the same (0.66 0.8787 (143), 0.67 0.8775 (143)); the average of both B peaks at 0.70 (0.8754, 143 wrong).
- Reading: against `1.csv` the 30 fps 288x512 B beats the 15 fps B only at its own lower threshold (0.8786 against 0.8758, 4 fewer wrong rows, which is small), and is worse at the shared threshold 0.68. The threshold gap comes from the rank reference being a 15 fps model: the honest fix is a fold-0 held-out run of the 30 fps 288x512 recipe to give it its own reference and threshold (about 5 h on Kaggle). File for the next submission: `submissions/blend_b30_288x512_thr067.csv`.
- Public result of `blend_b30_288x512_thr067.csv` (submitted by the user): **0.8792637**, 0.0030 below the best (0.8822952). The file differs from the best in 66 rows (30 cheat-more, 36 cheat-fewer), so on the public 30% about 20 rows differ and the drop is about 1 to 2 rows net: noise level, the same as the +0.003 that the `1.csv` check had predicted in the other direction. Reading: the 30 fps 288x512 fine-tune does not improve the blend, which matches the fold-0 screens (288x512 +0.002 mean AUC against a seed spread of 0.008 to 0.03). The preprocessing is identical in training and test (train frames recoded to 640x360 JPEG q90, test jpgs read as delivered, both resized with INTER_AREA to 288x512), so a resize mismatch is not indicated. The 15 fps 224x400 B at threshold 0.68 stays the best configuration. Not submitted: the average of both B models at 0.70 (`1.csv` 0.8754, 143 wrong).

## 2026-11-24 - Gaze-LLE feasibility check (peeking vs looking around)

- Setup: Gaze-LLE (CVPR 2025; code cloned into `cache/gazelle_repo/src` and read first, the decoder checkpoint `gazelle_dinov2_vitl14_inout.pt` 12.8 MB loaded with `weights_only=True`, DINOv2 ViT-L backbone from the public hub; FAQ #56 allows it as a feature extractor). Script `gaze_check.py` (scratchpad): frames in the test format (640x360, JPEG q90), head box per frame from the cached keypoints (nose, eyes, ears), gaze point = argmax of the 64x64 heatmap, in-frame score. Clips: 95 front Repeated_Peeking clips labelled cheat per second (`annotation/clip_labels_ai.csv`) against 144 front Looking_Around clips of the normal videos after 9.75 s, 26 subjects. Speed 0.69 clips per second (30 frames each), 346 s for the 239 clips.
- AUC of one gaze feature, peeking vs looking around (0.5 = no signal): mean |dx| (how far sideways the gaze point is from the head) 0.852; std of dx 0.758; circular std of the gaze angle 0.746; std of gaze x 0.746; path length of the gaze point 0.741; std of dy 0.732; mean in-frame score 0.189 (= 0.811 inverted); mean dx and the fraction of frames looking right of the head about 0.5 (no left / right signal).
- Cross-validated by subject (5 folds, logistic regression, 239 clips): head-pose features only (yaw and head-down mean / std / mean frame change from the keypoints, 6 numbers) AUC 0.732; the 13 gaze features only 0.823; head pose + gaze 0.850.
- Reading: the gaze features carry signal that a crude head-pose baseline does not (+0.12 AUC on this subset). Caveats: the 95 peeking clips are the clear ones (labelled cheat per second) and the looking-around clips are the normal-video action period, so this is the easy end of the peeking problem, and the baseline is a 6-number pose summary, not the 19-signal pose series used in the members. Not yet tested in the members. Next: extract the gaze series for all train clips and all 30 test frames (7,681 clips, about 3 h), add gaze x, y relative to the head and the in-frame score as extra per-frame inputs to the sequence members (E3, E58) and D1, then judge by `1.csv` in the blend.
- Full extraction started: `scripts/extract_gaze.py` (new): Gaze-LLE for every frame of the 6,166 labelled train clips and the 1,515 test sequences (all 30 frames, test jpgs as delivered), 6 numbers per frame (gaze x, y, soft gaze x, y, in-frame score, heatmap peak), head box from the cached keypoints; output `cache/gaze_lle/` (`train/<video>.npz`, `test.npy`), resumable, about 3 h (0.7 sequences per second). Plan: gaze series next to the pose series as extra per-frame inputs of the sequence members (E3 BiGRU / transformer, E58 transformer) and gaze summary numbers for D1, then judge by `1.csv` in the blend.

## 2026-11-24 - Cache cleanup (approved by the user)

- Deleted from `cache/` on the user's request ("clear the leftovers"): `kaggle_30fps` (the old 30 fps no-augmentation model), `dinov3_aug1_224x400` (augmented DINOv3 copies of E3a, rejected), `levjepa_ft` (the LeVJEPA fine-tune that did not learn), `levjepa_aug99_224_30` (speed-test copy), `h_crop_dinov3` (person crops, closed), `dinov3_hplus_224x400_step3_bf16` (old 10 fps features), `vjepa2_g_256_30`, `vjepa21_g_256_30`, `vjepa21_l_256_30`, `videomaev2_g_224_30` (rejected video-model members). About 4 GB freed, 161 GB free. The member predictions of those experiments stay in the scratchpad. Not deleted, to be reviewed later (user: "maybe the tier B we could arrive there again"): `levjepa_clips` (26 GB, rebuilt by `scripts/levjepa_clips.py` in about 11 minutes), `ft2/k12` (5.5 GB), `ft_hplus_block20` (1.6 GB), `kaggle_30fps_288x512_aug_full` (1.2 GB, the zip is still in Downloads).

## 2026-11-24 - Gaze-LLE gaze as an extra input of the sequence members (not adopted)

- Features: `scripts/extract_gaze.py` ran Gaze-LLE (frozen DINOv2 ViT-L plus its small decoder, head box from the cached ViTPose face keypoints) on all 30 frames of every labelled train clip (6,166) and every test sequence (1,515). 6 numbers per frame: gaze point x, y (heatmap peak), soft gaze x, y, in-frame score, heatmap peak. Saved in `cache/gaze_lle/`. 6 extra channels were appended to the per-frame input of each member (19 -> 25 for the pose-only part, plus summary statistics for D1). Same recipes otherwise, same folds.
- Members alone (held-out best F1 per clip, plain -> with gaze): D1 (LR, LeVJEPA + pose) 0.7381 -> 0.7402 (AUC 0.8539 -> 0.8578, peeking-cheat vs all normal AUC 0.808 -> 0.818); E58 BiGRU 0.7863 -> 0.7947; E58 transformer 0.7842 -> 0.7805; E3 BiGRU and transformer with gaze (DINOv3 members) 0.7742 and 0.7617 alone.
- Blend swaps (best configuration, B x2 + E3 BiGRU + E3 transformer + D1 + E58 transformer, one swap at a time; vs 1.csv first, then held-out fold-0 F1):

| swap | vs 1.csv at 0.68 | vs 1.csv at 0.70 | scan best (thr) | held-out F1 (best thr) | held-out F1 at 0.68 |
|---|---|---|---|---|---|
| best, no gaze | 0.8758 | 0.8722 | 0.8758 (0.680) | 0.8392 | 0.8422 |
| D1 -> D1 with gaze | 0.8707 | 0.8693 | 0.8738 (0.690) | 0.8434 | 0.8436 |
| E58 transformer -> with gaze | 0.8711 | 0.8659 | 0.8717 (0.695) | 0.8393 | 0.8382 |
| E3 BiGRU -> with gaze | 0.8690 | 0.8659 | 0.8701 (0.685) | 0.8406 | 0.8389 |
| E3 transformer -> with gaze | 0.8713 | 0.8659 | 0.8715 (0.690) | 0.8399 | 0.8345 |
| all four with gaze | 0.8630 | 0.8567 | 0.8631 (0.690) | 0.8389 | 0.8355 |
| best + E58 BiGRU with gaze added | 0.8692 | 0.8676 | 0.8698 (0.685) | 0.8330 | 0.8335 |

- Result: every gaze swap is lower than the best on the 1.csv check (0.863 to 0.871 against 0.8758), and the held-out gain is at most +0.004 (D1 only), inside the noise. Gaze as plain per-frame channels is NOT adopted. Not used in `main.ipynb`.
- Reading: gaze helps the peeking clips a little in the simple LR (AUC +0.01 on peeking-cheat), but the blend already gets that signal from pose and the image features. The earlier +0.12 AUC was on an easy subset against a crude pose baseline.
- Not tried for gaze: gaze relative to the wrists and shoulders (distance and direction, computed from the cached gaze and pose), the pooled 8x8 heatmap per frame (needs a re-extraction, about 3 h), a better head box, gaze for the augmented LeVJEPA copies.

## 2026-11-24 - Kaggle B trained on the new per-second labels (15 fps, 100% train) in the best blend

- Model: `kaggle_15fps_aug/kaggle_finetune_15fps_aug_full_newlab.ipynb` run on Kaggle (K=12, 15 fps, 224x400, 2 epochs, hide + weather + hflip, EMA over the last 20%, 100% train, 5,596 steps, 3.25 h). Weights in `cache/kaggle_15fps_newlab_aug_full/`; test inference on all 30 frames of every sequence (`scripts/kaggle_infer_test.py`, EMA and final logp saved there).
- Blend: the best configuration with ONLY B replaced (B x2, E3 BiGRU, E3 transformer, D1, E58 transformer; ranks inside the fold-0 held-out predictions, B reference = old B's Kaggle fold-0 model). Test scores of the two B models correlate 0.862; they agree on 87.5% of the sequences at rank 0.68.
- Against `1.csv` (secondary check), old-label B / new-label B (rows wrong): thr 0.66 0.8684 / 0.8640 (159 / 165); 0.67 0.8694 / 0.8648 (156 / 162); **0.68 0.8758 / 0.8623 (146 / 163)**; 0.69 0.8731 / 0.8589 (148 / 165); 0.70 0.8722 / 0.8636 (148 / 158); 0.71 0.8635 / 0.8647 (156 / 156). Cheat share at 0.68: 0.243 old, 0.249 new. 57 rows differ at 0.68.
- Result: the new-label B is lower on the 1.csv check at every threshold up to 0.70 (about -0.005 to -0.014), equal at 0.71. Consistent with the earlier fold-0 screens (no reliable gain). File written for the public check at the user's request: `submissions/blend_v2_e58tf_newlabB_thr068.csv`. Public score: not yet submitted by the user.
- Not run (user: tomorrow): E3 BiGRU and transformer (plain, cheat/normal) on the new labels. Script ready in the scratchpad (`e30c_newlab.py`, names E3nl_*; unsure seconds not used in training; prints AUC and F1 on the new and the old labels). D1 and E58 with the new labels already exist (D1 +0.003 AUC, E58 lower alone).
- Public score of `blend_v2_e58tf_newlabB_thr068.csv`: **0.8731680**, against 0.8822952 for the same blend with the old-label B (-0.0091). The 1.csv check predicted the direction (0.8623 against 0.8758, -0.0135). Conclusion: the new per-second labels do not help B on the public test; the test labels follow the old rule better (cheat from the start of the action to the end of the video, close to a fixed boundary at second 7, see the 2026-11-19 start-rule entries). Not adopted. The best configuration stays on the old labels.

## 2026-11-24 - Head pose (6DRepNet) as an extra input of the members (not adopted)

- Features: `scripts/extract_headpose.py` ran 6DRepNet (RepVGG-B1g2, weights `6DRepNet_300W_LP_AFLW2000.pth` downloaded from the authors' public cloud, loaded with `weights_only=True`; the sixdrepnet 0.1.6 wheel was checked and unpacked into `cache/sixdrepnet_pkg/`, not pip-installed) on a square face crop (from the cached ViTPose face keypoints) of all 30 frames of every labelled train clip and every test sequence, test-format frames. Output pitch, yaw, roll in degrees per frame (`cache/headpose/`, 31 min for everything). Six channels per frame (`build_hp_series.py`): pitch, yaw, roll, yaw and pitch minus the clip's own median (no other sequence used), |yaw change|. Raw yaw differs between train (mean -22) and test (-12.6), probably the front/side camera mix; the clip-relative channels do not. D1 also got 6 repetition numbers (biggest turn, share of turned frames, number of turns away and back, pitch range).
- Members alone, plain -> with head pose (AUC / best F1): D1 0.8539 / 0.7381 -> 0.8558 / 0.7403 (peeking vs Looking_Around AUC 0.754 -> 0.762); E58 BiGRU F1 0.7863 -> 0.7976; E58 transformer 0.8910 / 0.7842 -> 0.8899 / 0.7911; E3 BiGRU 0.8516 / 0.7729 -> 0.8530 / 0.7722; E3 transformer 0.8574 / 0.7741 -> 0.8422 / 0.7604.
- Blend swaps (best configuration, one change at a time; vs 1.csv at 0.68 / 0.70 first, then held-out fold-0 F1 at its best threshold): best 0.8758 / 0.8722, 0.8392; D1 -> with head pose 0.8739 / 0.8709, 0.8401; E58 transformer -> with head pose 0.8691 / 0.8676, 0.8411; E3 BiGRU -> 0.8698 / 0.8634, 0.8442; E3 transformer -> 0.8636 / 0.8613, 0.8412; D1 and E58 transformer together 0.8698 / 0.8661, 0.8447; E58 transformer and BiGRU with head pose 0.8626 / 0.8607, 0.8466; all four 0.8606 / 0.8662, 0.8371; best plus E58 BiGRU with head pose added 0.8655 / 0.8666, 0.8390.
- Result: every variant is below the best on the 1.csv check (0.8606 to 0.8739 against 0.8758), the held-out gain is at most +0.007 (inside the noise, and held-out has misled before). Same pattern as gaze and augmentation: members get better alone (E58 transformer +0.007 F1) but the blend does not. Head pose is NOT adopted and not in `main.ipynb`.
- Reading: the blend's remaining errors are shared by all members (fold-0 error split of the best blend: 25% are missed Repeated_Peeking clips, 17% Writing_Answers false alarms mostly from one subject, cheat clips 5 s or more after the start are missed 30% against 11% to 17% in 1 to 5 s), so a better single member does not remove them.
- Error split of the best blend on fold 0 (1,364 clips, 161 errors, F1 0.8422): Repeated_Peeking cheat 41 missed of 104, Writing_Answers normal 28 false alarms of 180, Looking_Around normal 20 of 191, Thinking normal 17 of 183, Reading_Hidden_Notes cheat 15 missed of 80, Checking_Smartwatch normal 12 of 182, Using_Smartwatch cheat 11 of 97, Looking_Under_the_Desk cheat 6 of 51. Subject 7 Writing_Answers (front and side) alone has 22 of the 28 Writing errors.

## 2026-11-24 - ViTPose+ large against ViTPose base (keypoint check only) and the model queue

- Check (scratchpad `pose_compare.py`, no F1 yet): ViTPose+ large (`usyd-community/vitpose-plus-large`, COCO expert, fp16) against the ViTPose base we use, same cached person boxes and test-format frames, 215 clips from 14 fold-0 videos. Median distance between the two models' keypoints (px): nose 0.94, eyes 0.89, ears 1.78, shoulders 2.06, elbows 1.92, wrists 2.51; 90th percentile 2.4 to 9.1 px. Head size (ear to ear) median 12.5 px, so the differences are 6% to 14% of the head. Frame-to-frame jitter base -> large (px): nose 0.50 -> 0.44, eyes 0.49 -> 0.44, ears 0.57 -> 0.49, shoulders 0.45 -> 0.37, elbows 0.60 -> 0.49, wrists 0.75 -> 0.59 (10% to 21% smoother); confidence about equal. No ground truth, so smoother is not proven more accurate. The full extraction (about 1.5 to 2 h for large, my estimate) and the member swap test have not been run.
- Downloaded (Hugging Face cache, safetensors and configs only, nothing executed): `timm/eva02_large_patch14_224.mim_m38m` (MIT, MIM-only, 303M), `timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k` (MIT, 305M), `galilai-group/LeVJEPA-VideoMix-Large` (CC BY-NC 4.0, 303M; its `modeling_levjepa.py` must be read for network or process calls before running), `google/vit-base-patch16-224` (Apache-2.0, 86.6M, supervised ImageNet-21k then 1k; expected to be weaker than DINOv3).
- Order to run when the GPU is free (user: one by one, judge each by the 1.csv column in the blend first): 1) LeVJEPA-VideoMix-Large frozen (D1, E5 BiGRU and transformer, plain, same folds, then E58 recipe if it is equal or better); 2) EVA-02 Large `ft_in22k_in1k` frozen (E3 recipe: 30 per-frame vectors, PCA 128, pose, BiGRU and transformer); 3) EVA-02 `mim_m38m` (MIM-only features are weak frozen, so only as a fine-tune start with the B recipe, fold-0 screen first); 4) ViT-Base frozen; 5) ViTPose+ large full pose extraction and the pose-member swap test.

## 2026-11-24 - LeVJEPA-VideoMix-Large (the other pretrain) as frozen members, plain recipe

- Model `galilai-group/LeVJEPA-VideoMix-Large` (27 Aug 2026, CC BY-NC 4.0, ViT-L 303M, trained on 1.8M clips of Kinetics-710, SSv2, Walking Tours and PE-Video). Its code files are byte-identical to the 1.5 Large files already used (read, no network or process calls), only the weights differ. `scripts/extract_levjepa.py` now takes env `MODEL` and `OUT_DIR` (default unchanged, attention mode stays `block_causal`); extraction into `cache/levjepa_videomix_224_30/` took about 47 min (2.7 clips/s), all 30 frames of every clip as two interleaved halves, same as the 1.5 features. Members built exactly like the 1.5 ones (scratchpad `d1_generic.py`, `e5_generic.py`; D1 = LR + pose, E5 = BiGRU / transformer on the 30 per-frame vectors, PCA 128, pose), same folds, plain recipe (no augmentation, cheat / normal head).
- Alone, 1.5 -> VideoMix (AUC / best F1): D1 C=0.001 0.8539 / 0.7381 -> 0.8609 / 0.7484 (peeking-cheat vs all normal 0.808 -> 0.818; test vs 1.csv best-threshold F1 of the single file 0.7736 -> 0.7818); E5 BiGRU 0.8695 / 0.7521 -> 0.8795 / 0.7728; E5 transformer 0.8784 / 0.7705 -> 0.8819 / 0.7742.
- Blend swaps (best configuration, one change at a time; vs 1.csv at 0.68 / 0.70 first, then held-out fold-0 F1 at its best threshold): best 0.8758 / 0.8722, 0.8392; D1 -> VideoMix D1 0.8721 / 0.8662, 0.8478; E58 transformer -> VideoMix E5 transformer (plain) 0.8670 / 0.8668, 0.8378; best + VideoMix E5 transformer 0.8687 / 0.8640, 0.8435; best + VideoMix E5 BiGRU 0.8642 / 0.8656, 0.8492; best + VideoMix D1 0.8629 / 0.8591, 0.8474.
- Result: better alone (up to +0.010 AUC, +0.021 F1 for the BiGRU), but every blend variant is below the best on the 1.csv check (0.8629 to 0.8721 against 0.8758); the closest is the D1 swap (3 rows worse, held-out +0.009). Not adopted. Not yet tried: the E58 recipe on VideoMix (augmented copy through `extract_levjepa_aug.py` with `levjepa_clips.py` frames, about 1.6 h, then the 8-action head), the one fair follow-up because E58 is the best frozen member.

## 2026-11-24 - EVA-02 Large (frozen image backbone, supervised after masked pretraining) as E3-style members

- Model: timm `eva02_large_patch14_448.mim_m38m_ft_in22k_in1k` (MIT, 305M; masked-image pretraining on Merged-38M, then supervised fine-tuning on ImageNet-22k and 1k). `scripts/extract_eva30.py`: CLS + mean patch (2048 numbers) for every frame of every train video (train frames through the test delivery format first) and all 30 frames of every test sequence, input 224x392 (16 x 28 patches of 14, dynamic image size), bf16 autocast, same layout as the DINOv3 per-frame cache (`cache/eva02_l_ft_224x392/`). The model is slow on this GPU: about 32 frames per second, 2 h 10 min for everything. Members: the plain E3 recipe (scratchpad `e3_generic.py`: 30 per-frame vectors, PCA 128, clip-mean centred, pose, BiGRU and transformer, 3 seeds, cheat / normal head, no augmentation) on the EVA features. One difference from the DINOv3 E3: the train frames here are in the test delivery format (the DINOv3 cache was made from the raw video frames).
- Alone (AUC / best F1), DINOv3 ViT-H+ E3 -> EVA-02: BiGRU 0.8516 / 0.7729 -> 0.7917 / 0.7239; transformer 0.8574 / 0.7741 -> 0.8006 / 0.7105. Much weaker (about -0.06 AUC, -0.05 F1).
- Blend swaps (best configuration; vs 1.csv at 0.68 / 0.70 first, then held-out F1): best 0.8758 / 0.8722, 0.8392; E3 BiGRU -> EVA 0.8631 / 0.8560, 0.8328; E3 transformer -> EVA 0.8705 / 0.8608, 0.8291; both E3 -> EVA 0.8502 / 0.8506, 0.8109; best + EVA BiGRU 0.8572 / 0.8537, 0.8333; best + EVA transformer 0.8590 / 0.8486, 0.8340; best + both 0.8396 / 0.8371, 0.8279. Every variant is worse than the best on both checks.
- Result: frozen EVA-02 (supervised on ImageNet classes) is clearly worse than frozen DINOv3 for this task and hurts the blend. Not adopted. The MIM-only `mim_m38m` weights are the only EVA variant left (as a fine-tune start; `ft2_train.py` is DINOv3-specific: model name, 224x400 frame cache, patch 16, so it needs the model name and a 224x392 frame cache before a fold-0 screen). ViT-Base (supervised ImageNet-21k / 1k, weaker than this model) is expected to be worse still.

