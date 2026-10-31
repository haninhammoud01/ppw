# Evaluation

Tahap ini membandingkan semua representasi teks yang sudah dicoba pada tahap Modeling. Semua model diuji pada **40 dokumen uji** (20% dari 200 dokumen, dibagi dengan `stratify=label` dan `random_state=42`), sehingga setiap representasi dinilai dengan soal yang sama.

## Ringkasan Akurasi

| Representasi | Algoritma | Jumlah fitur | Akurasi data uji |
|---|---|:---:|:---:|
| TF-IDF | MultinomialNB | 2.556 | 0,975 |
| PCA | GaussianNB | 104 | 0,925 |
| Truncated SVD | GaussianNB | 105 | 0,925 |
| Skip-gram manual (baseline, window 2, 50 dimensi) | GaussianNB | 50 | 0,950 |
| Skip-gram manual (window 5, 50 dimensi) | GaussianNB | 50 | 0,975 |

Sumber angka: notebook **Modeling** (TF-IDF, PCA, SVD) dan **Klasifikasi Skip-gram vs TF-IDF** (skip-gram dan perbandingannya dengan TF-IDF).

## Setelah Grid Search

| Model | Parameter terbaik | Akurasi CV (5-fold) | Akurasi data uji |
|---|---|:---:|:---:|
| TF-IDF + MultinomialNB | `alpha = 0.01` | 1,000 | 0,950 |
| Skip-gram + GaussianNB | `var_smoothing = 1e-12` | 1,000 | 0,975 |

Model dengan akurasi uji tertinggi, yaitu **Skip-gram manual + GaussianNB**, disimpan dengan `joblib` dan dipakai pada aplikasi Streamlit.

## Cara Membaca Hasil

- **Selisih kecil bukan selisih nyata.** Dengan 40 dokumen uji, satu dokumen yang salah menggeser akurasi sebesar 2,5 poin. Selisih antara 0,950 dan 0,975 hanya satu dokumen.
- **Akurasi CV 1,000 pada data latih** menunjukkan bahwa dua kategori ini mudah dipisahkan (kosakata finance dan sport sangat berbeda). Angka ini tidak dipakai untuk memilih model; pemilihan memakai data uji yang tidak ikut dilatih.
- **Skip-gram memakai 50 fitur**, jauh lebih sedikit dari TF-IDF (2.556 fitur), tetapi akurasinya setara. Artinya vektor kata hasil neural network sudah cukup menangkap perbedaan topik.
- **Keterbatasan:** model hanya mengenal kategori finance dan sport, dan ukuran data masih kecil (200 artikel). Berita dari kanal lain akan tetap dipaksa masuk ke salah satu dari dua kategori itu.

## Kesimpulan

Kriteria keberhasilan pada tahap Business Understanding (akurasi di atas 90%) terpenuhi: model dapat membedakan artikel finance dan sport dengan akurasi 0,95 sampai 0,975 pada data uji. Model terpilih diimplementasikan pada tahap [Deployment](Production).
