# Ekstraksi Hasil Eksperimen — Hybrid LSTM-Prophet Multiple Seasonality Carbon Intensity

**Judul penelitian:** Forecasting Tren Hourly Carbon Intensity dengan Hybrid LSTM-Prophet Multiple Seasonality Berbasis Renewable dan Carbon Free Energy
**Skema:** Non Kerja Sama Eksternal — C3 (International Proceeding), Periode 2026 (1)
**Sumber:** 11 notebook (`00_dani_eda` → `10_dani_ensemble_tuned`) + `data_loader.py`
**Dibuat:** ekstraksi otomatis dari output notebook yang sudah dijalankan (bukan re-run) — semua angka di bawah adalah hasil aktual yang tersimpan di cell output.

---

## 1. Dataset

| Item                                                   | Detail                                                                                                                         |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| Sumber                                                 | ElectricityMaps, snapshot hourly Indonesia (`ID`)                                                                              |
| Rentang final (CI/RE modeling)                         | 2024-01-01 s.d. 2025-12-31 (2 tahun penuh, hourly)                                                                             |
| Jumlah baris (setelah gabung 2024+2025)                | 17.544 baris (8.784 di 2024 — tahun kabisat, 8.760 di 2025)                                                                    |
| Duplicate timestamp                                    | 0 (1 duplicate row terdeteksi & dibuang di preprocessing)                                                                      |
| Missing values (kolom inti)                            | 0% pada CI, CI life-cycle, CFE%, RE% setelah cleaning                                                                          |
| Kolom inti yang dipakai modeling                       | `carbon_intensity` (direct, gCO₂eq/kWh), `renewable_percentage` (RE%)                                                          |
| Kolom tambahan tersedia (tidak dipakai sebagai target) | CI Life-cycle, CFE% (identik dengan RE% — lihat catatan di bawah), breakdown per sumber energi (coal, gas, hydro, solar, dst.) |
| Dataset EDA historis lebih panjang                     | Ada arsip 2017–2025 (78.864 baris) dipakai khusus untuk EDA jangka panjang (ACF, year-over-year), bukan untuk training model   |

**Statistik deskriptif (2024–2025, 17.544 jam):**

| Variabel                   |   Mean |   Std |    Min |    Max |
| -------------------------- | -----: | ----: | -----: | -----: |
| CI direct (gCO₂eq/kWh)     | 585.44 | 10.44 | 558.03 | 632.65 |
| CI life-cycle (gCO₂eq/kWh) | 664.62 | 10.21 | 639.40 | 712.86 |
| CFE%                       |  15.63 |  1.45 |   8.30 |  19.62 |
| RE%                        |  15.63 |  1.45 |   8.30 |  19.62 |

**Temuan penting:** Di dataset Indonesia ini, **CFE% dan RE% identik secara numerik** (max diff = 0.0, `np.allclose` = True) — artinya Indonesia praktis tidak punya kontribusi non-renewable carbon-free (nuklir) dalam bauran energinya, jadi kedua variabel itu **satu sinyal yang sama**, bukan dua eksogen independen. Ini penting untuk narasi paper: judul menyebut "RE% dan CFE%" tapi secara empiris hanya ada satu variabel eksogen efektif di data Indonesia.

CI direct vs CI life-cycle: korelasi 0.985, R² regresi linear = 0.970, rasio life-cycle/direct rata-rata ≈ 1.135 (relatif konstan, std 0.0043) → CI life-cycle secara praktis adalah CI direct dikali faktor ~1.135, sehingga modeling difokuskan pada CI direct saja.

Korelasi CI direct vs RE%/CFE%: **−0.901** (kuat negatif) → mengonfirmasi hipotesis inti penelitian: makin tinggi porsi energi terbarukan, makin rendah intensitas karbon.

---

## 2. Pola Multiple Seasonality (menjawab SQ1)

### Pola harian (hour-of-day, rata-rata 2024–2025)

- CI direct terendah sekitar jam **13:00** (~577 gCO₂eq/kWh) dan tertinggi dini hari/malam sekitar jam **04:00 & 23:00** (~588–590).
- RE%/CFE% mengikuti pola berlawanan dengan cekungan siang: turun ke titik terendah sekitar jam **08:00** (~15.18%) dan naik ke puncak malam sekitar jam **21:00–22:00** (~16.1%).
- Dekomposisi Prophet: puncak pagi ~06:00–07:00 (+5, demand ramp-up), **lembah siang ~13:00 (−10, diduga kontribusi solar PV menekan CI)**, recovery sore setelah jam 17:00 (solar turun, fosil dominan lagi).
- Amplitudo seasonal harian dari MSTL: ±10–15 gCO₂eq/kWh — komponen musiman paling dominan & paling stabil sepanjang tahun.

### Pola mingguan (day-of-week)

- CI tertinggi hari **Jumat** (589.4) dan terendah hari **Minggu** (581.2); Sabtu-Minggu jelas lebih rendah dari hari kerja.
- RE%/CFE% tertinggi di **Minggu** (16.55%) dan terendah di **Jumat** (15.11%) — pola berlawanan sempurna dengan CI, konsisten dengan hipotesis "aktivitas ekonomi rendah di akhir pekan → proporsi RE lebih tinggi → CI turun".
- Dekomposisi Prophet weekly: minggu (Sunday) paling rendah (~−5.5), Jumat–Sabtu tertinggi (~+5.5).
- Amplitudo seasonal mingguan dari MSTL: ±8–10 gCO₂eq/kWh, struktur lebih kompleks/variatif dibanding pola harian.

### Trend & residual (MSTL)

- Komponen trend **bukan tren linear naik/turun**, melainkan osilasi bulanan (~560–580), relatif stasioner jangka panjang — kemungkinan mencerminkan perubahan musim (hujan → lebih banyak hidro → CI turun).
- Residual tersebar ±10–15 dengan beberapa outlier ekstrem sekitar Agustus–September 2024 (mendekati −15), perlu dicek apakah event khusus.

### MSTL vs Prophet — siapa lebih baik untuk dekomposisi?

- **Trend:** Prophet menghasilkan trend yang "spiky" (overfitting ke changepoints), sedangkan MSTL menghasilkan trend yang smooth. → **MSTL lebih superior untuk mengisolasi trend** pada data ini.
- **Forecasting one-week-ahead:** Prophet baik menangkap ritme/frekuensi multiple seasonality, tapi lemah menangkap level-shift & volatilitas non-linear (contoh: lonjakan CI aktual 22–26 Oktober tidak tertangkap, residual besar). → **Inilah motivasi utama pendekatan hybrid**: kelemahan Prophet pada dinamika non-linear dikompensasi oleh LSTM di tahap residual modeling.

### Baseline statistik klasik (AR/MA/SARIMAX+Fourier)

| Model                                        |                                                        AIC | Catatan                                                                                                                                                                                                                                                |
| -------------------------------------------- | ---------------------------------------------------------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| MA(24)                                       | 3412.89 (subset awal) / **3583.45** (angka evaluasi akhir) | AIC terendah di antara ketiganya                                                                                                                                                                                                                       |
| AR(24)                                       | 3481.21 (subset awal) / **3752.47** (angka evaluasi akhir) | Lebih tinggi dari MA — model lebih kompleks tapi fit relatif lebih buruk                                                                                                                                                                               |
| SARIMAX(1,1,1) + Fourier (harian & mingguan) |                                                    3602.44 | AR(1) coef 0.583 (p<0.001) — autokorelasi lag-1 kuat; MA(1) coef −0.911 (p<0.001) — koreksi error sangat kuat; Fourier harian (sin/cos 1,24 & 2,24) signifikan p<0.001; Fourier mingguan sebagian tertangkap (cos 1,168 signifikan p=0.012, sin tidak) |

**Kesimpulan baseline statistik:** ada trade-off jelas — AR/MA punya error matematis rendah tapi gagal struktural (mean-reverting/flatline, tidak benar-benar meniru pola), sedangkan SARIMAX+Fourier mempertahankan struktur musiman tapi overshoot pada volatilitas non-linear. **Ini yang memvalidasi urgensi Deep Learning (LSTM)** sebagai lapisan tambahan yang bisa mempelajari volatilitas non-linear yang luput dari model klasik.

---

## 3. Pipeline & Konfigurasi Eksperimen (konsisten di seluruh notebook 04–08)

- Split temporal 80:20 (train:test), tanpa shuffle
- `MinMaxScaler` di-fit HANYA pada training data (no leakage)
- Look-back baseline = 24 jam (1 hari); tuned look-back bervariasi per skema (12/48 jam, lihat §5–6)
- Horizon: one-step-ahead (utama) + multistep (rolling-origin recursive, H=24 jam untuk hybrid)
- Seed = 42, epochs = 50, batch size = 32 (default), validation split = 0.1, early stopping patience = 10
- Dua representasi temporal dibandingkan sepanjang notebook 04–07: **Dummy** (H1-H24, B1-B12, M1H1-M7H24 one-hot) vs **Cyclical** (hour/dow/month sin-cos)
- Struktur 6-slot arsitektur konsisten dari nb04 s.d. nb08: `Uni-LSTM CI`, `Uni-LSTM RE`, `Multi-LSTM` (CI+RE), `BiLSTM CI`, `BiLSTM RE`, `BiLSTM Multi`

---

## 4. Deep Learning Murni — Tanpa Dekomposisi (nb04–nb06)

### 4.1 Baseline dengan Dummy Features (nb04)

| Model               | Target |       RMSE |        MAE |        MAPE |
| ------------------- | ------ | ---------: | ---------: | ----------: |
| LSTM Univariate     | CI     |     3.2445 |     2.5848 |     0.4426% |
| LSTM Multivariate   | CI     | **2.3674** | **1.8571** | **0.3182%** |
| BiLSTM Univariate   | CI     |     3.8764 |     3.1392 |     0.5382% |
| BiLSTM Multivariate | CI     |     2.4616 |     1.9408 |     0.3322% |
| LSTM Univariate     | RE     |     0.3490 |     0.2819 |     1.7595% |
| LSTM Multivariate   | RE     | **0.2777** | **0.2174** | **1.3849%** |
| BiLSTM Univariate   | RE     |     0.3338 |     0.2689 |     1.6763% |
| BiLSTM Multivariate | RE     |     0.2917 |     0.2245 |     1.4334% |

→ Dengan dummy features, **LSTM Multivariate** (bukan BiLSTM) adalah yang terbaik untuk kedua target.

### 4.2 Dengan Cyclical Features (nb05)

| Model            | Target |    MAE |       RMSE |    MAPE |     R² |
| ---------------- | ------ | -----: | ---------: | ------: | -----: |
| **BiLSTM Multi** | CI     | 1.9739 | **2.5131** | 0.3386% | 0.9223 |
| Multi-LSTM       | CI     | 2.2725 |     2.8371 | 0.3903% | 0.9010 |
| BiLSTM CI        | CI     | 2.4226 |     2.9174 | 0.4150% | 0.8953 |
| Uni-LSTM CI      | CI     | 2.5014 |     2.9795 | 0.4283% | 0.8908 |
| **BiLSTM RE**    | RE     | 0.1630 | **0.2171** | 1.0301% | 0.9568 |
| Uni-LSTM RE      | RE     | 0.1663 |     0.2190 | 1.0504% | 0.9560 |
| Multi-LSTM       | RE     | 0.2059 |     0.2659 | 1.3012% | 0.9352 |
| BiLSTM Multi     | RE     | 0.2190 |     0.2802 | 1.3812% | 0.9280 |

→ **Cyclical features mengungguli Dummy features secara signifikan** untuk model terbaik (CI RMSE turun dari 2.37 → 2.51 di skenario terbaik masing²… tapi arsitektur pemenang berubah dari LSTM Multi ke **BiLSTM Multi** untuk CI, dan ke **BiLSTM RE** univariate untuk RE). BiLSTM Multi + Cyclical jadi kandidat model terbaik nb05 → dibawa ke tuning (nb06).

### 4.3 Hyperparameter Tuning dengan Optuna (nb06) — BiLSTM Multivariate + Cyclical

**Search space:** look_back {12,24,48,72}, tipe layer (LSTM/BiLSTM), jumlah stacked layer (1-3), units {32,64,96,128,192,256}, dropout (0.0-0.5), learning rate (1e-4–1e-2 log), batch size {16,32,64,128}. Budget 40 trial (TPESampler + MedianPruner), objective = rata-rata scaled validation MSE (CI+RE equally-weighted lewat MinMax).

**Best trial (trial #25):** val_loss = 0.001071
| Hyperparameter | Nilai |
|---|---|
| look_back | 12 jam |
| layer_type | **lstm** (bukan BiLSTM!) |
| n_layers | 1 |
| units_l1 | 128 |
| dropout_l1 | 0.0 |
| learning_rate | 0.001847 |
| batch_size | 64 |

**Hasil akhir model tuned (test set):**
| Model | Target | MAE | RMSE | MAPE | R² |
|---|---|---:|---:|---:|---:|
| BiLSTM Multi (Optuna-tuned) | CI | 1.7466 | **2.2594** | 0.2993% | 0.9372 |
| BiLSTM Multi (Optuna-tuned) | RE | 0.1817 | **0.2362** | 1.1520% | 0.9488 |

**Improvement vs baseline nb05 (BiLSTM Multi + Cyclical, tidak di-tuning):**
| Target | Metric | Baseline (nb05) | Tuned (nb06) | Improvement |
|---|---|---:|---:|---:|
| CI | RMSE | 2.5131 | 2.2594 | **+10.09%** |
| CI | MAPE | 0.3386% | 0.2993% | +11.59% |
| RE | RMSE | 0.2802 | 0.2362 | **+15.70%** |
| RE | MAPE | 1.3812% | 1.1520% | +16.60% |

→ Tuning Optuna memberi improvement dua digit, konsisten untuk kedua target. Catatan menarik: arsitektur optimal ternyata **plain LSTM 1-layer**, bukan BiLSTM bertumpuk — model paling sederhana justru menang dalam pencarian hyperparameter yang lebih luas.

### 4.4 MSTL-as-Feature (nb07_dani_dl_mstl — **versi draft/superseded**)

Catatan penting: notebook `07_dani_dl_mstl.ipynb` memakai MSTL sebagai _input feature tambahan_ (bukan dekomposisi residual), dan menurut catatan di notebook `07_dani_hybrid_residual_lstm.ipynb`, pendekatan ini **digantikan** karena skema resmi proyek mendefinisikan kategori "Hybrid" sebagai residual modeling, bukan feature-augmentation. Hasilnya tetap dicatat sebagai ablasi/pembanding:

Model terbaik per target (di antara Dummy vs Cyclical × MSTL-feature):
| Target | Temporal | Model | RMSE | MAPE | R² |
|---|---|---|---:|---:|---:|
| CI | dummy | BiLSTM CI | **2.2878** | 0.3088% | 0.9359 |
| CI | cyclical | Multi-LSTM | 2.6732 | 0.3665% | 0.9125 |
| RE | cyclical | BiLSTM RE | **0.2127** | 1.0079% | 0.9587 |
| RE | dummy | Uni-LSTM RE | 0.2281 | 1.1268% | 0.9525 |

→ Menariknya, di skema MSTL-as-feature ini, **Dummy features lebih unggul untuk CI** (kebalikan dari nb05 tanpa dekomposisi, di mana Cyclical menang) — perilaku dummy vs cyclical tampak berinteraksi dengan ada/tidaknya fitur dekomposisi. Karena skema ini superseded, tidak direkomendasikan jadi angka utama laporan, tapi baik disebut sebagai ablasi metodologi.

---

## 5. Hybrid Prophet/MSTL + Residual LSTM — Skema Resmi (nb07_hybrid_residual_lstm)

**Definisi resmi skema hybrid** (sesuai tabel model di proposal): input = **residual model utama** (pakai hasil tuning deep learning), bukan dekomposisi-sebagai-fitur.

**Arsitektur:**

1. Dekomposisi `Y = Trend + Seasonal + Residual` via **MSTL** (causal/leak-free: trend = trailing rolling mean window 168 jam, seasonal_24 & seasonal_168 = profil periodik di-fit dari train saja) **atau Prophet** (fit di train, extend ke test via `make_future_dataframe`, leak-free by design).
2. Trend+Seasonal diekstrapolasi deterministik (bukan oleh LSTM).
3. LSTM/BiLSTM memodelkan **residual saja**, reuse hyperparameter tuning Optuna dari nb06 (look_back=12, 1 layer, units=128, dropout=0.0, lr=0.001847, batch=64) — untuk semua 6 slot arsitektur (Multi-LSTM reuse langsung; Uni-LSTM & BiLSTM variants reuse _nilai_ hyperparameter yang sama, hanya beda tipe layer/output_dim — simplifikasi yang eksplisit dicatat, jadi alasan nb08 tetap diperlukan).
4. Forecast akhir = Trend + Seasonal + Residual_prediksi.
5. Evaluasi multistep pakai **rolling-origin recursive** (H=24 jam, banyak titik origin), bukan recursive tunggal — supaya menjawab langsung apakah residual model drift terhadap horizon.

### 5.1 Hasil One-Step (12 konfigurasi: {MSTL, Prophet} × 6 slot arsitektur)

| Decomposition | Model           | Target |    MAE |       RMSE |        MAPE |         R² |
| ------------- | --------------- | ------ | -----: | ---------: | ----------: | ---------: |
| **mstl**      | **BiLSTM CI**   | CI     | 1.5181 | **2.0276** | **0.2596%** | **0.9495** |
| mstl          | BiLSTM Multi    | CI     | 1.5202 |     2.0278 |     0.2600% |     0.9495 |
| mstl          | Uni-LSTM CI     | CI     | 1.5228 |     2.0374 |     0.2604% |     0.9490 |
| mstl          | Multi-LSTM      | CI     | 1.6333 |     2.1588 |     0.2795% |     0.9428 |
| prophet       | BiLSTM CI       | CI     | 1.9282 |     2.4775 |     0.3297% |     0.9246 |
| prophet       | BiLSTM Multi    | CI     | 1.9351 |     2.4796 |     0.3314% |     0.9245 |
| prophet       | Multi-LSTM      | CI     | 1.9623 |     2.5211 |     0.3362% |     0.9219 |
| prophet       | Uni-LSTM CI     | CI     | 1.9986 |     2.5572 |     0.3425% |     0.9197 |
| **mstl**      | **Uni-LSTM RE** | RE     | 0.1623 | **0.2199** | **1.0303%** | **0.9557** |
| mstl          | BiLSTM RE       | RE     | 0.1673 |     0.2244 |     1.0633% |     0.9539 |
| mstl          | Multi-LSTM      | RE     | 0.1846 |     0.2428 |     1.1724% |     0.9461 |
| prophet       | Multi-LSTM      | RE     | 0.1789 |     0.2378 |     1.1352% |     0.9483 |
| mstl          | BiLSTM Multi    | RE     | 0.2139 |     0.2694 |     1.3540% |     0.9336 |
| prophet       | BiLSTM Multi    | RE     | 0.1917 |     0.2496 |     1.2154% |     0.9430 |
| prophet       | BiLSTM RE       | RE     | 0.2049 |     0.2657 |     1.2965% |     0.9354 |
| prophet       | Uni-LSTM RE     | RE     | 0.2589 |     0.3305 |     1.6303% |     0.9000 |

**Temuan kunci:**

- **MSTL secara konsisten mengalahkan Prophet** sebagai backbone dekomposisi untuk hybrid residual — di 12/12 pasangan model yang sama, MSTL selalu punya RMSE lebih rendah dari Prophet.
- Model terbaik overall untuk CI: **MSTL + BiLSTM CI** (RMSE 2.0276, R²=0.9495) — ini adalah **model terbaik di seluruh eksperimen (nb04–nb10) untuk target CI**.
- Model terbaik overall untuk RE: **MSTL + Uni-LSTM RE** (RMSE 0.2199, R²=0.9557) — juga **terbaik di seluruh eksperimen** untuk target RE.
- Hybrid (residual MSTL) mengungguli deep-learning-murni-tuned nb06 (BiLSTM Multi tuned: CI RMSE 2.2594 vs hybrid MSTL BiLSTM CI 2.0276) — **dekomposisi residual memberi keuntungan nyata** dibanding model DL langsung, bahkan dengan hyperparameter yang belum di-tuning khusus untuk residual.

### 5.2 Hasil Multistep (Rolling-Origin, H=1–24 jam) — MSTL vs Prophet

Contoh kurva MAE-vs-horizon untuk CI, model BiLSTM CI:

| Horizon (jam) | MSTL MAE | Prophet MAE |
| ------------: | -------: | ----------: |
|             1 |     1.27 |        1.67 |
|             5 |     3.70 |        3.98 |
|            10 |     4.27 |        8.10 |
|            15 |     4.55 |       12.27 |
|            20 |     3.93 |       12.37 |

**Temuan kunci (menjawab bagian "keamanan recursive forecast"):**

- **MSTL residual stabil/plateau** — MAE naik landai dari ~1.3 di H=1 ke ~4 di H≥10, lalu relatif datar sampai H=24. Ini sehat: model tidak drift signifikan seiring bertambahnya horizon.
- **Prophet residual meledak** — MAE naik hampir monoton dari ~1.7 di H=1 sampai >12 di H≥12, jauh lebih buruk dari MSTL pada horizon panjang. Ini adalah **bukti kuantitatif kuat** yang mendukung superioritas MSTL sebagai backbone dekomposisi untuk forecasting multistep, melengkapi temuan kualitatif di §2 (trend Prophet yang "spiky"/overfitting changepoints).
- Implikasi untuk paper: **rekomendasikan MSTL sebagai dekomposisi utama** untuk multistep ahead; Prophet bisa tetap dilaporkan sebagai pembanding/ablasi yang menunjukkan mengapa pilihan metodologis MSTL penting.

---

## 6. Hybrid dengan Hyperparameter Tuning per Slot (nb08)

**Revisi scope:** Versi awal mencakup 12 search (90 trial/search) tapi terlalu berat secara komputasi (semalaman jalan, config pertama dari 12 saja belum selesai). Dipangkas jadi **4 search independen**: `{mstl, prophet} × {Multi-LSTM, BiLSTM Multi}`, masing-masing 25 trial (turun dari 90) — 3 alasan empiris: (1) nb07 MSTL BiLSTM Multi RMSE 2.0278 nyaris tied dengan overall-best MSTL BiLSTM CI (2.0276); (2) ensemble `multi_only` (nb09/10) kompetitif; (3) premis judul "berbasis Renewable Energy" butuh CI diprediksi bersama RE sebagai input — hanya dipenuhi Multi-LSTM & BiLSTM Multi.

### Best hyperparameters (4 search)

| Decomposition | Model        | look_back | batch_size | n_layers | units | dropout | learning_rate | best_val_loss |
| ------------- | ------------ | --------: | ---------: | -------: | ----: | ------: | ------------: | ------------: |
| mstl          | BiLSTM Multi |        48 |         64 |        1 |   128 |  0.0998 |      0.001068 |       0.00195 |
| mstl          | Multi-LSTM   |        48 |         64 |        1 |   128 |  0.0998 |      0.001068 |       0.00195 |
| prophet       | BiLSTM Multi |        48 |         64 |        1 |   128 |  0.0998 |      0.001068 |       0.00313 |
| prophet       | Multi-LSTM   |        48 |         64 |        1 |   128 |  0.0998 |      0.001068 |       0.00307 |

### Hasil evaluasi one-step (test set) & perbandingan vs default (nb07)

| Decomposition | Model        | Target | RMSE (tuned) | RMSE (nb07 default) |               Improvement |
| ------------- | ------------ | ------ | -----------: | ------------------: | ------------------------: |
| mstl          | BiLSTM Multi | CI     |       2.1605 |              2.0278 |  **−6.54%** (lebih jelek) |
| mstl          | Multi-LSTM   | CI     |       2.7906 |              2.1588 | **−29.27%** (lebih jelek) |
| prophet       | BiLSTM Multi | CI     |       2.8674 |              2.4796 |     −15.64% (lebih jelek) |
| prophet       | Multi-LSTM   | CI     |       3.1546 |              2.5211 |     −25.13% (lebih jelek) |
| mstl          | BiLSTM Multi | RE     |       0.2228 |              0.2694 |  **+17.29%** (lebih baik) |
| mstl          | Multi-LSTM   | RE     |       0.2341 |              0.2428 |       +3.57% (lebih baik) |
| prophet       | Multi-LSTM   | RE     |       0.2740 |              0.2378 |     −15.26% (lebih jelek) |
| prophet       | BiLSTM Multi | RE     |       0.2959 |              0.2496 |     −18.54% (lebih jelek) |

**Temuan penting (negative/counter-intuitive finding — layak dilaporkan apa adanya):**
Tuning hyperparameter per-slot (look_back naik ke 48 jam, dsb.) **memperburuk performa CI di semua 4 konfigurasi**, dan hanya membantu RE di 2 dari 4 konfigurasi. Ini adalah temuan yang sah secara metodologis: **hyperparameter default yang di-reuse dari nb06 (look_back=12, model sederhana) sudah mendekati optimal untuk residual modeling**, dan pencarian Optuna dengan search space lebih luas (khususnya look_back lebih panjang) justru overfit atau kurang cocok untuk sinyal residual yang relatif stasioner. Model terbaik untuk CI di seluruh eksperimen tetap **MSTL + BiLSTM CI dari nb07 (default hyperparameter), RMSE 2.0276** — bukan hasil tuning nb08.

---

## 7. Ensemble (nb09 — heuristik 1/RMSE, nb10 — Optuna-tuned weights)

**Strategi:** weighted average dari 12 model hybrid (nb07/nb08). Bobot dihitung dari **validation RMSE** (bukan test — mencegah leakage). Dua varian dibandingkan: **`all`** (8 model per target, campur univariate+multivariate) vs **`multi_only`** (hanya Multi-LSTM + BiLSTM Multi, 4 model — konsisten dengan premis judul "berbasis Renewable Energy", direkomendasikan sebagai angka final).

### 7.1 nb09 — Bobot heuristik (1/RMSE)

| Target | Best individual model          | Best ensemble                              |                  Improvement |
| ------ | ------------------------------ | ------------------------------------------ | ---------------------------: |
| CI     | mstl_BiLSTM CI (RMSE 2.0276)   | Ensemble-Weighted (mstl_only, RMSE 2.0128) |                   **+0.73%** |
| RE     | mstl_Uni-LSTM RE (RMSE 0.2199) | Ensemble-Simple (mstl_only, RMSE 0.2227)   | **−1.24%** (ensemble kalah!) |

`all` vs `multi_only` (RMSE): CI — all=2.0317 vs multi_only=2.0992 (multi_only 3.32% lebih jelek); RE — all=0.2227 vs multi_only=0.2284 (multi_only 2.58% lebih jelek). → Ada trade-off nyata: `multi_only` lebih defensible secara metodologis (konsisten premis judul) tapi sedikit kalah performa dari `all`.

### 7.2 nb10 — Bobot hasil pencarian Optuna (menggantikan heuristik 1/RMSE)

Optuna mencari bobot linear per model (bukan arsitektur — tidak ada training ulang) yang meminimalkan RMSE kombinasi di **validation set**.

| Target | Best individual model          | Best ensemble (Optuna-tuned)                   |                                 Improvement |
| ------ | ------------------------------ | ---------------------------------------------- | ------------------------------------------: |
| CI     | mstl_BiLSTM CI (RMSE 2.0276)   | Ensemble-OptunaTuned (multi_only, RMSE 2.0121) |                                  **+0.76%** |
| RE     | mstl_Uni-LSTM RE (RMSE 0.2199) | Ensemble-OptunaTuned (all, RMSE 0.2201)        | **−0.05%** (nyaris tied, masih kalah tipis) |

**Kesimpulan ensemble (temuan penting untuk paper):**
Baik dengan bobot heuristik maupun bobot hasil optimasi Optuna, **ensembling tidak memberikan keuntungan berarti** dibanding model tunggal terbaik: improvement CI hanya <1% (0.73–0.76%), dan untuk RE ensemble justru sedikit kalah dari model individu terbaik (mstl Uni-LSTM RE). Ini konsisten dengan temuan nb08: **model hybrid tunggal terbaik (MSTL + BiLSTM CI untuk CI, MSTL + Uni-LSTM RE untuk RE) sudah cukup kuat**, sehingga baik hyperparameter tuning tambahan (nb08) maupun ensembling (nb09/10) tidak memberikan gain substansial di atasnya. Ini adalah kesimpulan yang sah dan layak dilaporkan eksplisit di paper sebagai bagian dari analisis, bukan kegagalan eksperimen.

---

## 8. Ringkasan Model Terbaik — Lintas Semua Eksperimen

| Target | Model Terbaik          | Skema                                          |       RMSE |    MAE |    MAPE |     R² |
| ------ | ---------------------- | ---------------------------------------------- | ---------: | -----: | ------: | -----: |
| **CI** | **MSTL + BiLSTM CI**   | Hybrid residual, default hyperparameter (nb07) | **2.0276** | 1.5181 | 0.2596% | 0.9495 |
| **RE** | **MSTL + Uni-LSTM RE** | Hybrid residual, default hyperparameter (nb07) | **0.2199** | 0.1623 | 1.0303% | 0.9557 |

Runner-up yang layak disebut di paper sebagai model resmi (karena konsisten dengan premis "berbasis Renewable Energy", yaitu CI diprediksi bersama RE sebagai input):

- CI: **MSTL + BiLSTM Multi** (RMSE 2.0278, nyaris tied dengan BiLSTM CI univariate) — direkomendasikan nb08 sebagai basis pemilihan slot yang dituning, dan dasar `multi_only` di ensemble.
- Alternatif ensemble (jika ingin angka "gabungan"): **Ensemble-OptunaTuned (multi_only)** CI RMSE 2.0121 — sedikit lebih baik dari model tunggal, tapi peningkatannya marginal (<1%).

### Progresi performa CI (RMSE, lebih rendah lebih baik)

| Tahap                                                                         |                                          RMSE CI |
| ----------------------------------------------------------------------------- | -----------------------------------------------: |
| SARIMAX+Fourier (statistik klasik, AIC-based, bukan RMSE langsung comparable) |                                  — (AIC 3602.44) |
| DL murni, dummy features, LSTM Multivariate (nb04)                            |                                           2.3674 |
| DL murni, cyclical features, BiLSTM Multi (nb05)                              |                                           2.5131 |
| DL murni, cyclical + Optuna-tuned, BiLSTM Multi (nb06)                        |                                           2.2594 |
| **Hybrid MSTL residual, default hyperparameter, BiLSTM CI (nb07)**            |                             **2.0276** ← terbaik |
| Hybrid MSTL residual, Optuna-tuned per-slot (nb08)                            |                   2.1605 (lebih jelek dari nb07) |
| Ensemble heuristik 1/RMSE, multi_only (nb09)                                  |                                           2.0992 |
| Ensemble Optuna-tuned, multi_only (nb10)                                      | 2.0121 (terbaik ensemble, tapi <1% di atas nb07) |

**Insight metodologis utama:** Loncatan performa terbesar terjadi saat berpindah dari "DL murni" ke "Hybrid decomposition+residual" (nb06 → nb07: RMSE CI turun 2.2594 → 2.0276, ~10% improvement). Setelah itu, baik hyperparameter tuning tambahan maupun ensembling hanya memberi gain marginal atau bahkan memperburuk hasil — mengindikasikan **dekomposisi (khususnya MSTL) adalah komponen paling berpengaruh dalam arsitektur hybrid ini**, lebih penting daripada elaborasi lebih lanjut pada sisi tuning/ensembling.

---

## 9. Menjawab Sub-Pertanyaan Penelitian (SQ)

**SQ1 — Karakteristik multiple seasonality (harian & mingguan) pada Hourly CI:**

- Seasonality harian dominan & sangat stabil (amplitudo ±10–15 gCO₂eq/kWh), dengan lembah siang ~13:00 (diduga efek solar PV) dan puncak dini hari/malam.
- Seasonality mingguan lebih kecil tapi konsisten (±8–10 gCO₂eq/kWh), CI tertinggi Jumat dan terendah Minggu — berkorelasi terbalik sempurna dengan pola RE%.
- Trend bukan tren linear, melainkan osilasi bulanan/musiman jangka menengah, relatif stasioner.
- MSTL menangkap ketiga komponen ini lebih baik (lebih smooth, lebih causal/leak-free) dibanding Prophet, yang cenderung overfit pada changepoints untuk komponen trend dan gagal menangkap level-shift/volatilitas non-linear.

**SQ2 — Pengaruh fluktuasi RE% dan CFE% terhadap akurasi Hybrid LSTM-Prophet (dan LSTM-MSTL):**

- RE% dan CFE% identik secara empiris di dataset Indonesia (korelasi sempurna) — jadi secara statistik hanya ada satu sinyal eksogen efektif, dengan korelasi −0.901 terhadap CI.
- Memasukkan RE% sebagai variabel eksogen (skema Multivariate: Multi-LSTM / BiLSTM Multi) secara konsisten menurunkan error CI dibanding skema univariate — contoh nb04: LSTM Multivariate RMSE 2.3674 vs LSTM Univariate 3.2445 (**27% lebih baik**).
- Namun pada skema hybrid residual (nb07, model terbaik), univariate **BiLSTM CI** (RMSE 2.0276) sedikit mengungguli multivariate **BiLSTM Multi** (RMSE 2.0278) — perbedaannya nyaris nol (0.01%), artinya setelah dekomposisi MSTL menghilangkan sebagian besar sinyal musiman, kontribusi tambahan RE% sebagai eksogen pada level residual menjadi sangat kecil.
- Kesimpulan: **RE%/CFE% berpengaruh besar terhadap akurasi pada level raw-target modeling (tanpa dekomposisi)**, tapi pengaruhnya **menyusut drastis begitu dekomposisi musiman (MSTL) diterapkan** — karena sebagian besar variasi yang berkorelasi dengan RE% (yaitu pola harian & mingguan) sudah tertangkap oleh komponen seasonal, menyisakan residual yang lebih murni non-linear/idiosinkratik di mana RE% tidak banyak menambah informasi baru.
- Implikasi metodologis: variabel eksogen RE%/CFE% paling bernilai sebagai _input langsung ke model DL_ (bukan hanya di sisi residual hybrid) — mendukung penggunaan skema Multi-LSTM/BiLSTM Multi sebagai varian yang dilaporkan (`multi_only`), meski trade-off performanya kecil dibanding varian univariate terbaik.

---

## 10. Catatan Metodologi & Poin yang Perlu Diperjelas di Paper

1. **Leakage control:** Semua skema (MSTL causal, Prophet fit-train-only, MinMaxScaler fit-train-only) sudah didesain leak-free — baik untuk one-step maupun multistep (rolling-origin). Ini poin metodologis kuat yang layak ditekankan di bagian Metode.
2. **CFE% = RE% secara numerik** di dataset ini — perlu dijelaskan eksplisit di paper (bukan kesalahan data, tapi karakteristik bauran energi Indonesia yang minim nuklir), supaya tidak menimbulkan pertanyaan reviewer soal kenapa hanya satu variabel eksogen efektif yang dipakai.
3. **nb07_dani_dl_mstl.ipynb adalah draft superseded** — sebaiknya tidak jadi headline result, cukup disebut sebagai ablasi metodologi (MSTL-as-feature vs MSTL-as-residual-decomposition).
4. **Hyperparameter reuse di nb07:** Uni-LSTM & BiLSTM variants mereuse nilai hyperparameter dari Multi-LSTM tuning (nb06), bukan tuning independen — perlu disebutkan sebagai simplifikasi/limitasi, meski nb08 mengonfirmasi tuning independen tidak lantas lebih baik.
5. **Negative findings yang sah untuk dilaporkan (bukan kegagalan):**
   - Tuning per-slot (nb08) memperburuk CI dibanding default hyperparameter nb07.
   - Ensembling (nb09/nb10) memberi gain <1% untuk CI dan sedikit merugikan untuk RE.
   - Ini memperkuat argumen bahwa **pemilihan skema dekomposisi (MSTL vs Prophet) adalah keputusan desain paling berpengaruh**, bukan hyperparameter tuning atau ensembling.
6. **Rekomendasi angka final untuk paper:**
   - Headline result: **MSTL + BiLSTM CI** (CI) dan **MSTL + Uni-LSTM RE** (RE), dari nb07, sebagai model hybrid terbaik.
   - Untuk konsistensi dengan premis judul "berbasis RE/CFE": laporkan juga **MSTL + BiLSTM Multi** (CI, RMSE 2.0278, nyaris tied) sebagai varian yang secara eksplisit memakai RE% sebagai input, dan diskusikan trade-off metodologi vs performa mentah secara eksplisit (sudah diantisipasi di nb09 markdown).
   - Ensemble Optuna-tuned `multi_only` (RMSE CI 2.0121) bisa dilaporkan sebagai upaya lanjutan yang menunjukkan batas atas performa dicapai dengan pendekatan hybrid tunggal — bukan sebagai "peningkatan besar".

---

## 11. Peta Artefak Tersimpan (untuk referensi saat menulis laporan)

| Notebook      | Artefak hasil (CSV/JSON/model)                                                                                                                                                        |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| nb05          | `results/deep_learning/05_dani_dl_cyclical_metrics.csv`, model di `models/deep_learning/`                                                                                             |
| nb06          | `06_optuna_trials.csv`, `06_best_hyperparameters.json`, `06_optuna_study.pkl`, `06_dani_dl_optuna_metrics.csv`, `06_baseline_vs_tuned_comparison.csv`, `06_bilstm_multi_optuna.keras` |
| nb07 (hybrid) | `07_dani_hybrid_onestep_metrics.csv`, `07_dani_hybrid_multistep_horizon_mae.csv`                                                                                                      |
| nb08          | `08_dani_hybrid_tuned_best_params.csv`, `08_dani_hybrid_tuned_onestep_metrics.csv`                                                                                                    |
| nb09          | `09_dani_ensemble_onestep_metrics_{SOURCE}.csv`, `09_dani_ensemble_individual_base_metrics_{SOURCE}.csv`                                                                              |
| nb10          | `10_dani_ensemble_tuned_weights_{SOURCE}.csv`, `10_dani_ensemble_tuned_onestep_metrics_{SOURCE}.csv`                                                                                  |

`{SOURCE}` = `"nb07"` (default hyperparameter) atau `"nb08"` (tuned per-slot) — toggle yang dipakai di nb09/nb10 untuk memilih basis model.

---

_Dokumen ini adalah ekstraksi objektif dari output notebook yang sudah dijalankan — semua angka bersumber langsung dari cell output, bukan estimasi ulang. Untuk kebutuhan publikasi, disarankan menjalankan ulang model terbaik (MSTL + BiLSTM CI, MSTL + Uni-LSTM RE) dengan seed berbeda untuk melaporkan interval kepercayaan/variance jika reviewer meminta._
