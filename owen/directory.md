# Dataset Overview — Data Slayer 4.0

## 1. Competition Information

**Task:** Binary Video Classification — Exam Cheating Detection

Dataset digunakan untuk mengklasifikasikan perilaku peserta ujian menjadi dua kategori:

- **0 — Normal:** Perilaku normal selama ujian.
- **1 — Cheat:** Perilaku yang menunjukkan kecurangan.

Dataset terdiri dari video training (`.mov` / `.MOV`) dan kumpulan sequence frame gambar (`.jpg`) untuk testing.

---

## 2. Dataset Directory Structure

Dataset terverifikasi berada pada direktori input Kaggle berikut:

```text
/kaggle/input/competitions/data-slayer-4/
│
├── train/
│   ├── normal/
│   │   ├── subject_1/
│   │   ├── subject_2/
│   │   ├── ...
│   │   └── subject_26/
│   │
│   └── cheat/
│       ├── subject_1/
│       ├── subject_2/
│       ├── ...
│       └── subject_26/
│
├── test/
│   ├── test_seq_00001/
│   │   ├── 01.jpg
│   │   ├── 02.jpg
│   │   ├── ...
│   │   └── 30.jpg
│   │
│   ├── test_seq_00002/
│   │   ├── 01.jpg
│   │   ├── ...
│   │   └── 30.jpg
│   │
│   └── ...
│
└── sample_submission.csv
```

---

## 3. Training Dataset

**Path:** `/kaggle/input/competitions/data-slayer-4/train/`

Training dataset terdiri dari:

- **26 subjects**
- **2 classes:** normal dan cheat
- **4 behaviors** per class
- **2 camera angles:** front dan side
- **16 videos** per subject
- **416 videos** secara keseluruhan berdasarkan spesifikasi

### Normal Class (Label 0)

| Behavior | Description |
|---|---|
| Writing Answers | Menulis jawaban ujian |
| Thinking | Berpikir saat mengerjakan soal |
| Looking Around | Melihat ke sekitar |
| Checking Smartwatch | Memeriksa smartwatch |

### Cheat Class (Label 1)

| Behavior | Description |
|---|---|
| Looking Under the Desk | Melihat ke bawah meja |
| Reading Hidden Notes | Membaca catatan tersembunyi |
| Repeated Peeking | Berulang kali mengintip |
| Using Smartwatch | Menggunakan smartwatch untuk kecurangan |

Setiap behavior memiliki rekaman dari sudut kamera **Front** dan **Side**.

---

## 4. Testing Dataset

**Path:** `/kaggle/input/competitions/data-slayer-4/test/`

Testing dataset disusun berdasarkan sequence folder:

```text
test/
└── test_seq_XXXXX/
    ├── 01.jpg
    ├── 02.jpg
    ├── ...
    └── 30.jpg
```

Setiap sequence memiliki karakteristik:

| Property | Value |
|---|---|
| Estimated sequences | $\approx 1.515$ sequence ($45.450$ frame / 30) |
| Frames per sequence | 30 |
| Duration | 1 second |
| Frame rate | 30 FPS |
| Resolution | 1280 × 720 |
| Camera angle | Single angle per sequence (Front atau Side, unlabelled) |
| Image format | JPG |
| Target | Normal (0) / Cheat (1) |

Model harus menghasilkan **satu prediksi kelas untuk setiap test sequence**, bukan satu prediksi untuk setiap frame.

---

## 5. Submission File

**Path:** `/kaggle/input/competitions/data-slayer-4/sample_submission.csv`

File ini menyediakan format referensi submission.

Prediksi yang dihasilkan harus mengikuti jumlah baris, nama kolom, dan urutan sequence pada `sample_submission.csv`.

---

## 6. Dataset Summary

| Property | Value |
|---|---|
| Competition | Data Slayer 4.0 |
| Verified Kaggle Path | `/kaggle/input/competitions/data-slayer-4/` |
| Task | Binary Classification (`0` = normal, `1` = cheat) |
| Evaluation Metric | F1-Macro Score |
| Training subjects | 26 subjects |
| Training videos | 416 videos |
| Test sequences | $\approx 1.515$ sequence |
| Classes | 2 (Normal, Cheat) |
| Behaviors per class | 4 behaviors |
| Camera angles | Front, Side |
| Test frames per sequence | 30 frames (1 second @ 30 FPS) |
| Test resolution | 1280 × 720 |
| Dataset size | 6.88 GB |
| Total files | 45,867 files |
| File formats | MOV (train), JPG (test), CSV (sample submission) |

---

## 7. Important Notes & Modeling Recommendations

1. **Root Directory:** Path dataset terverifikasi di Kaggle adalah `/kaggle/input/competitions/data-slayer-4/`.
2. **Case Sensitivity:** File video train berformat `.MOV` (huruf kapital) atau `.mov`. Pastikan pembacaan file bersifat *case-insensitive* (`file.lower().endswith(('.mov', '.mp4'))`).
3. **Format Data:** 
   - Training menggunakan video `.mov`.
   - Testing menggunakan sequence gambar `.jpg` yang sudah diekstrak per 30 frame (1 detik).
4. **Temporal Alignment (Slicing 1 Detik):** 
   - Video train berdurasi multi-detik, sedangkan test sequence berdurasi tepat 1 detik (30 frame).
   - Dianjurkan memotong video train menjadi klip 30 frame (1 detik) agar kecepatan gerak (*temporal velocity*) konsisten 1:1 antara train dan test.
5. **Validasi (GroupKFold by Subject):**
   - Pembagian cross-validation **wajib dikelompokkan berdasarkan subject** (`GroupKFold(n_splits=5, groups=subjects)`). Hal ini krusial untuk mencegah model menghafal wajah/baju peserta ujian antara data train dan validasi.
6. **Optimasi Metrik:**
   - Evaluasi menggunakan **F1-Macro**. Ambang batas prediksi (*threshold*) default `0.5` sebaiknya dioptimasi melalui grid search pada prediksi Out-of-Fold (OOF).
