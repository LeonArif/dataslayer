# Experiment Logs (fayadh)

## Struktur folder
```
fayadh/
├── baseline.ipynb      # motion + LGBM (LB 0.5737)
├── main.ipynb          # pose pretrained + rekonstruksi video test + Hungarian (LB 0.6872)
├── logs.md
├── scripts/            # dijalankan dari mana saja: python scripts/<nama>.py
│   ├── extract_pose.py     # ekstraksi pose RT-DETR + ViTPose ke cache/main (sama dengan main.ipynb, untuk background run)
│   ├── extract_dino.py     # ekstraksi embedding DINOv2 ke cache/dino (eksperimen, tidak dipakai main)
│   ├── test_structure.py   # rekonstruksi rantai video test ke cache/main/test_chains.csv
│   ├── exp_pose.py         # eksperimen CV fitur pose
│   └── exp_dino.py         # eksperimen CV embedding DINOv2
├── models/             # baseline_model.pkl, main_model.pkl
├── submissions/        # submission_*.csv
└── cache/              # git-ignored, berisi fitur turunan data wajah
    ├── baseline/       # fitur motion + HOG (ekstraksi ±43 menit)
    ├── main/           # pose, box, rantai test (ekstraksi ±75 menit)
    └── dino/           # embedding DINOv2-small (ekstraksi ±40 menit)
```

## 2026-11-16 - baseline.ipynb (handcrafted motion + HOG, LGBM)

Notebook: `baseline.ipynb`. Outputs: `submissions/submission_baseline.csv`, `models/baseline_model.pkl`, `cache/baseline/` (fitur hasil ekstraksi, git-ignored).

### Data facts (dicek langsung dari file)
- Train: 416 video `.MOV` = 26 subjek x 2 kategori x 4 perilaku x 2 angle, seimbang sempurna (26 video per perilaku per angle).
- Video train: 1280 x 720, 30 FPS (side kadang 29.97), 407-558 frame (13.6-18.6 detik, rata-rata 16.0 detik).
- Test: 1515 sequence x 30 `.jpg`. **Frame test berukuran 640 x 360**, bukan 1280 x 720 seperti di deskripsi lomba.
- `sample_submission.csv`: `id,label`, id = `test_seq_00001` dst, sama persis dengan nama folder test.
- Contoh frame test menunjukkan subjek yang sama dengan train (kelas, kursi, dan meja sama). Belum dicek apakah subjek test termasuk dalam 26 subjek train.

### Pipeline
- Frame train: resize ke 640 x 360, lalu round-trip JPEG (q=90) supaya mirip test, lalu grayscale. Fitur gerakan dihitung pada 160 x 90.
- Klip train: 30 frame, stride 15, menghasilkan 12.717 klip (26-36 per video). Test: 1 klip per sequence.
- Fitur (908 total):
  - `md_*` energi frame difference (5)
  - `mg_*` grid 6x8 rata-rata/std gerakan (96)
  - `of_*` statistik optical flow Farneback + histogram arah 8 bin (15)
  - `og_*` grid 3x4 dx/dy/mag (36)
  - `hog_*` HOG dari frame rata-rata klip, 128 x 64 (756)
- HOG diimplementasi manual karena `cv2.HOGDescriptor` tidak ada di opencv 5.0.0.
- Model: LGBM binary, 400 tree, lr 0.03, 31 leaves, colsample 0.5. Validasi StratifiedGroupKFold(5) per subjek.
- Waktu ekstraksi: train 2145 detik, test 462 detik (12 core).

### Hasil
"video agg" = rata-rata probabilitas OOF semua klip satu video, lalu F1-macro per video. Ini hanya diagnosis, karena test berbentuk klip terpisah.

| Approach | CV F1 (thr 0.5) | CV F1 (tuned thr) | video agg |
|---|---|---|---|
| Semua fitur (motion + HOG) | 0.5126 | 0.5222 (thr 0.64) | 0.537 |
| **Motion saja, tanpa HOG = baseline final** | **0.5811** | **0.5811 (thr 0.50)** | **0.634** |
| Motion z-score per subjek x angle | 0.532 | 0.532 | 0.616 |

- Dengan HOG: importance HOG 97.8%, fold 0.633 / 0.495 / 0.466 / 0.494 / 0.462. HOG membuat model mengenali subjek atau angle, bukan perilaku, jadi `MODEL_USE_HOG = False`. HOG tetap ada di cache.
- Baseline final (152 fitur): fold 0.641 / 0.585 / 0.573 / 0.498 / 0.568. Importance: grid motion 63.7%, grid flow 32.4%, statistik flow 3.5%.
- Akurasi OOF per perilaku (front / side):
  - Normal: Checking Smartwatch 0.55/0.62, Looking Around 0.58/0.61, Thinking 0.57/0.64, Writing Answers 0.50/0.58
  - Cheat: Looking Under the Desk 0.77/0.62, Reading Hidden Notes 0.63/0.53, Repeated Peeking 0.50/0.49, Using Smartwatch 0.67/0.45
- Subjek tersulit: 6 (0.43), 15 (0.46), 16 (0.50).
- Prediksi test: 854 cheat / 661 normal (56.4% cheat).
- **Belum disubmit ke LB.**

### Kesimpulan
- Baseline ini kerangka pipeline yang lengkap: data loading, penyamaan resolusi dan JPEG train-test, klip 30 frame, CV per subjek, threshold, dan submission. Fitur gerakan global hanya sedikit di atas acak.
- Sinyal per klip lemah, tetapi agregasi per video naik ke 0.634. Artinya konteks lebih dari 1 detik membantu.

### Ide untuk main.ipynb
1. Cek apakah sequence test berurutan berasal dari video yang sama (bandingkan frame terakhir seq i dengan frame pertama seq i+1, atau kemiripan background/subjek). Jika ya, smoothing prediksi antar sequence tetangga bisa memberi keuntungan seperti video agg.
2. Cek apakah subjek test ada di train. Jika ya, validasi random per video lebih representatif daripada per subjek.
3. Model temporal sederhana dari nol (3D-CNN kecil atau CNN 2D + GRU) pada frame grayscale kecil, karena ada 12.7k klip dan GPU (torch 2.6 + cu124).
4. Fitur yang lebih terlokalisasi: crop area subjek (bounding box dari frame difference), lalu motion dan HOG di dalam crop. Ini mengurangi pengaruh background dan angle.
5. Klasifikasi 8 perilaku (multiclass) lalu dipetakan ke 2 kelas, karena perilaku di dalam satu kelas sangat berbeda.
6. Konfirmasi ke panitia apakah pretrained weights (ImageNet / pose estimator) dihitung sebagai "data eksternal".

## 2026-11-17 - main.ipynb (pose estimation pretrained + struktur test + Hungarian)

**LB: 0.6872** (CV 0.8846). Gap besar, lihat bagian diagnosis di bawah.

Notebook: `main.ipynb`. Outputs: `submissions/submission_main.csv` (utama), `submissions/submission_main_video.csv` (cadangan tanpa assignment kelompok), `models/main_model.pkl` (3 LGBM), cache pose dan rantai test di `cache/main/`.
Baseline LB: 0.5737 (CV 0.581), jadi CV per subjek bisa dipercaya.

### Temuan struktur test
- Urutan id test diacak, tetapi setiap sequence punya lanjutan: frame pertama sequence lain hampir identik dengan frame ke-30-nya (median L1 0.36 vs 0.81 antara frame 1 dan 30 dalam satu sequence). Dengan mutual nearest neighbor, 1515 sequence tersusun jadi **97 rantai** sepanjang 14-19 detik (sama seperti durasi video train), ditambah 2 rantai 8 detik yang merupakan satu video terputus dan digabung otomatis.
- Angle per video bisa diprediksi 100% (CV) dari fitur pose. Test: 48 video front + 48 video side.
- Subjek dikelompokkan dari histogram warna HSV crop orang, lalu capacitated clustering (8 video per subjek-angle). Hasilnya 12 kelompok x 8 video, sehingga test = **6 subjek x 16 video**, strukturnya sama dengan train.
- Ini bukan hand labeling: semua dilakukan otomatis dari piksel.

### Model pretrained (Hugging Face)
- Percobaan 1, DINOv2-small frozen (frame penuh 448x252, CLS + mean patch + grid 3x4, 5 fps) + LogReg: clip 0.52-0.55, video+Hungarian terbaik 0.659. Lebih buruk dari baseline motion. Orang hanya mengisi sebagian kecil frame, dan perbedaan antar perilaku ada di gerakan kecil kepala dan tangan.
- GTX 1650: fp16 4x **lebih lambat** dari fp32, jadi semua inferensi memakai fp32.
- **Pose: RT-DETR `PekingU/rtdetr_r50vd_coco_o365` (box orang di frame tengah setiap window 1 detik) + ViTPose `usyd-community/vitpose-base-simple` (17 keypoint, 10 frame per detik).** Ekstraksi train 3530 detik, test 767 detik.
- Masalah teknis yang sudah diatasi: worker loky di Windows mengimpor ulang torch (sangat lambat), jadi diganti thread. Generator joblib tanpa batas buffer membuat RAM habis, jadi dipakai prefetch berbatas dan hanya menyimpan frame yang dipakai.

### Experiments (LGBM multiclass 8 perilaku, StratifiedGroupKFold(5) per subjek, window 30 frame tanpa overlap)
clip = per window thr 0.5; video = rata-rata log-prob per video; bal = 4 cheat teratas per subjek-angle; hung = Hungarian 8 perilaku per subjek-angle.

| Approach | clip | video | bal | hung | beh acc video |
|---|---|---|---|---|---|
| Baseline motion (baseline.ipynb) | 0.581 | 0.634 | - | - | - |
| DINOv2 base+grid+std+diff, LogReg | 0.551 | 0.563 | 0.591 | 0.659 | 0.320 |
| Pose per window (131 fitur) | 0.621 | 0.702 | 0.740 | 0.798 | 0.567 |
| + konteks video (vid_, dev_) | 0.771 | 0.766 | 0.789 | 0.846 | 0.673 |
| **+ vidz_ (rata-rata video di-z-score per subjek-angle)** | 0.798 | 0.795 | 0.808 | **0.880** | 0.743 |
| + winz_ (window di-z-score per subjek-angle) | 0.806 | 0.805 | 0.813 | 0.865 | 0.743 |
| vidz, 600 tree lr 0.02 | 0.800 | 0.798 | 0.813 | 0.880 | 0.740 |
| **Final di notebook (vidz, 300 tree, 3 seed)** | **0.806** | **0.805** | **0.808** | **0.885** | |

- Importance: vidz_ 78%, vid_ 21%, fitur window dan dev hampir 0. Model pada dasarnya mengklasifikasikan video yang dinormalisasi per subjek.
- Confusion utama (video, argmax): Looking Around <-> Repeated Peeking (15 dan 14 video), Checking <-> Using Smartwatch (7 dan 5), Writing Answers -> Looking Under the Desk / Reading Hidden Notes.
- Akurasi biner terendah setelah Hungarian: Repeated Peeking side 0.615, Looking Around side 0.654.
- Test: hung vs video threshold sepakat 82.8%; cheat share 50.2%.
- **Belum disubmit ke LB.** Ekspektasi LB sekitar 0.85-0.88 jika pengelompokan subjek test benar.

### Catatan risiko
- Rules melarang data eksternal. Model pretrained (RT-DETR, ViTPose) dipakai atas keputusan tim. Perlu dikonfirmasi ke panitia sebelum final.
- Hungarian bergantung pada pengelompokan subjek-angle test. Jika ada video yang salah kelompok, efeknya lokal pada kelompok itu. `submission_main_video.csv` adalah cadangan tanpa asumsi kelompok (CV 0.805).

### Ide berikutnya
1. Pisahkan Looking Around vs Repeated Peeking: fitur periodisitas toleh kepala di level video (jumlah toleh, durasi menahan toleh, autokorelasi head_yaw), karena "repeated" baru terlihat dalam beberapa detik.
2. Model sekuens di level video (GRU/1D-CNN atas fitur pose per detik) sebagai tambahan LGBM.
3. Window train overlap (stride 15) untuk menggandakan data window.
4. Crop tangan/pergelangan + DINOv2 untuk smartwatch (Checking vs Using).

### Diagnosis gap CV 0.885 vs LB 0.687
- Pengelompokan subjek-angle test dicek visual untuk ke-96 video: 12 kelompok x 8 video, masing-masing satu orang. **Benar.**
- Tempo frame test vs train setara (selisih antar frame berurutan, median 0.37 vs 0.34). Frame rate bukan penyebab.
- Kepercayaan model di test (max prob per video 0.89) setara dengan OOF (0.85). Di banyak kelompok, argmax sudah memberi 8 perilaku berbeda tanpa Hungarian. Tidak terlihat domain shift yang merusak.
- Adversarial validation fitur window AUC 0.999 (box_h, box_cy, posisi tangan). Ini wajar karena subjek baru, dan sebagian besar dinetralkan oleh vidz_.
- Hipotesis tersisa:
  1. Public LB hanya berisi sedikit subjek (6 subjek di test), jadi variansinya besar. Kelompok yang paling tidak yakin: side_2 dan side_5.
  2. Label test per detik, bukan per video.
- Probe berikutnya: submit `submissions/submission_main_video.csv` (beda 17% dari main, tanpa Hungarian).
