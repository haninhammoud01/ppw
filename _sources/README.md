<div align="center">

# 📰 PPW - Web Search & Mining

**Klasifikasi Berita Indonesia: Sport vs Finance**

Tugas mata kuliah Pencarian dan Penambangan Web - disusun dengan metodologi CRISP-DM

[![GitHub Pages](https://img.shields.io/badge/docs-GitHub%20Pages-blue?logo=github)](https://haninhammoud01.github.io/ppw/)
[![Python](https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter Book](https://img.shields.io/badge/built%20with-Jupyter%20Book-orange?logo=jupyter)](https://jupyterbook.org/)

</div>

---

## 📖 Tentang Proyek

Proyek ini mengumpulkan dan menyiapkan dataset **200 artikel berita** berbahasa Indonesia dari [detik.com](https://detik.com), untuk klasifikasi teks berita ke dalam dua kategori: **Sport** dan **Finance**, dengan representasi TF-IDF, PCA, SVD, dan skip-gram yang dibangun manual, serta aplikasi Streamlit.

**📚 Baca dokumentasi lengkap →** [haninhammoud01.github.io/ppw](https://haninhammoud01.github.io/ppw/)

## 🗂️ Struktur Proyek

```
ppw/
├── CRISP-DM/
│   ├── Business.ipynb                → Business Understanding
│   ├── EDA.ipynb                     → Data Understanding
│   ├── Input.ipynb                   → Data Preparation (crawling)
│   ├── Preprocessing.ipynb           → Text Preprocessing
│   ├── Preprocessing_Ablation.ipynb  → Perbandingan tahap preprocessing
│   ├── Vectorization.ipynb           → TF-IDF, PCA, SVD
│   ├── Skipgram.ipynb                → Skip-gram manual (demo 3 kalimat)
│   ├── Modeling.ipynb                → Naive Bayes pada TF-IDF/PCA/SVD
│   ├── Skipgram_Classification.ipynb → Skip-gram vs TF-IDF + Grid Search
│   ├── Output.md                     → Evaluation
│   └── Production.md                 → Deployment
├── data/                             → dataset dan model hasil training
├── deployment/                       → aplikasi Streamlit (app.py + model/)
├── _config.yml
├── _toc.yml
└── intro.md
```

## 🚀 Aplikasi

```bash
pip install -r requirements.txt
streamlit run deployment/app.py
```

## 📊 Ringkasan Dataset

| Kategori | Jumlah Artikel | ID Range |
|----------|:---:|:---:|
| ⚽ Sport | 100 | 1–100 |
| 💰 Finance | 100 | 101–200 |

**Kolom:** `id` · `isi_berita` · `label` · `url`

## 🛠️ Tools

`Python` · `Trafilatura` · `Pandas` · `scikit-learn` · `NumPy` · `Streamlit` · `Jupyter Book`

---

<div align="center">

*Hanin Hammoud*

</div>
