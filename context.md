# Data Slayer 4.0: Machine Learning Competition
**Tema:** *Honest Score, Equal Chance: Exam Cheating Detection*
**Host:** DATA SLAYER (Kaggle Community Prediction Competition, Private)
**Metrik:** F1-Macro Score
**Status (16 Nov 2026):** mulai 19 jam lalu, tutup sekitar 13 hari lagi (kurang lebih 29 Nov 2026)

---

## 1. Overview
Data Slayer 4.0 adalah kompetisi data science tingkat nasional untuk mahasiswa Indonesia. Tugasnya membuat model machine learning / computer vision yang mengklasifikasikan perilaku peserta ujian pada rekaman video menjadi **normal** atau **cheat** (menyontek), untuk membantu pengawas menjaga ujian tetap adil.

## 2. Latar Belakang
- Kemendikbud mencatat 126 kasus kecurangan terverifikasi pada UN SMA/SMK/MA 2019.
- Sejalan dengan SDG 4 (Quality Education): setiap peserta berhak mendapat nilai yang mencerminkan kemampuannya sendiri.
- Tantangan utama:
  - Membedakan perilaku mencurigakan dari aktivitas normal yang mirip. Contoh: menoleh karena berpikir vs mengintip jawaban; cek waktu di smartwatch vs memakai smartwatch untuk mencari jawaban.
  - Variasi sudut kamera (front dan side).
  - Karakteristik gerakan yang berbeda di setiap peserta (subjek).
  - Meminimalkan false positive, yaitu peserta jujur yang dituduh curang.

## 3. Evaluasi
**F1-Macro**: rata-rata F1 kelas 0 (normal) dan kelas 1 (cheat) dengan bobot sama.

$$F1_c = \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}, \qquad \text{F1-Macro} = \frac{F1_0 + F1_1}{2}$$

Implikasi: kedua kelas sama penting, jadi threshold probabilitas perlu dituning untuk F1-Macro (bukan default 0.5) dan prediksi tidak boleh condong ke satu kelas.

## 4. Dataset
Total 45.867 file, 6,88 GB (jpg, mov, csv).

### 4.1 Label
| Label | Kategori | Perilaku (masing-masing direkam dari angle Front dan Side) |
|---|---|---|
| 0 | normal | Writing Answers, Thinking, Looking Around, Checking Smartwatch |
| 1 | cheat | Looking Under the Desk, Reading Hidden Notes, Repeated Peeking, Using Smartwatch |

Pasangan yang mirip secara visual (sumber kebingungan utama):
- Looking Around (normal) vs Repeated Peeking (cheat)
- Checking Smartwatch (normal) vs Using Smartwatch (cheat)
- Thinking / Writing Answers (normal) vs Reading Hidden Notes / Looking Under the Desk (cheat)

### 4.2 Struktur
- **train/**: 26 subjek, masing-masing 16 video (2 kategori x 4 perilaku x 2 angle) = 416 video `.MOV`.
  - Path: `train/{cheat|normal}/subject_{i}/{front|side}_{Behavior}.MOV`
  - Contoh: `train/cheat/subject_1/front_Looking_Under_the_Desk.MOV`
- **test/**: folder per *sequence*. Setiap sequence berisi 30 frame `.jpg` (1 detik pada 30 FPS) dari **satu** angle kamera, resolusi 1280 x 720.
- **sample_submission.csv**: contoh format submission (2 kolom: id sequence dan label).

### 4.3 Catatan penting dari struktur data
- Train berbentuk video panjang, sedangkan test berupa potongan 1 detik (30 frame). Train harus dipotong menjadi klip 30 frame agar distribusinya sama dengan test.
- Validasi harus di-group per **subjek** (GroupKFold by subject), karena kemungkinan besar subjek test berbeda dari subjek train dan klip dari video yang sama sangat mirip satu sama lain.
- Angle (front/side) tidak diberikan secara eksplisit di test, kecuali bisa disimpulkan dari gambar.

## 5. Rules
- Kerja sama antar tim dilarang dalam bentuk apa pun.
- Boleh memakai tools, library, dan algoritma model apa pun selama tidak melanggar aturan.
- **Dilarang melakukan prediksi secara manual** (termasuk hand labeling data test).
- **Dilarang menggunakan data di luar kompetisi / data eksternal.**
  - Catatan tim: pretrained weights (mis. ImageNet, pose estimator) belum jelas statusnya. Perlu dikonfirmasi ke panitia sebelum dipakai di submission final.
- Panitia berhak mendiskualifikasi tim yang curang atau melanggar aturan sebelum, selama, atau setelah kompetisi. Keputusan juri final.

### 5.1 Privasi & Data
Dataset berisi wajah manusia (data sensitif):
- Dilarang menggunakan, membagikan, mempublikasikan, atau mendistribusikan dataset di luar kepentingan kompetisi tanpa izin resmi panitia.
- Dilarang mengunggah ulang dataset ke platform publik mana pun (termasuk GitHub; folder `data/` sudah di-ignore di `.gitignore`).
- Penggunaan dataset harus mengikuti prinsip keamanan dan privasi.

### 5.2 Kaggle Foundational Rules (ringkasan yang relevan)
- Satu akun Kaggle per orang; tim maksimal sesuai ketentuan di website kompetisi.
- Tidak boleh berbagi kode atau data secara privat di luar tim. Kode boleh dibagikan publik hanya lewat forum/notebook kompetisi di Kaggle.
- Open source yang dipakai harus berlisensi OSI-approved yang tidak membatasi penggunaan komersial.
- Pemenang ditentukan dari **Private Leaderboard**. Jika seri, submission yang lebih dulu masuk menang.
- Calon pemenang dapat diminta menyerahkan kode dan dokumentasi untuk verifikasi.

### 5.3 Perubahan aturan
Informasi dan perubahan aturan diumumkan lewat Instagram **@dataslayer.tup** dan WhatsApp Group Competition.

## 6. Struktur Repo
```
dataslayer/
├── context.md        # file ini
├── data/             # dataset (git-ignored)
│   ├── train/{cheat,normal}/subject_*/{front,side}_*.MOV
│   ├── test/<sequence>/*.jpg
│   └── sample_submission.csv
├── fayadh/           # baseline.ipynb, main.ipynb, logs.md
└── leon/
```
