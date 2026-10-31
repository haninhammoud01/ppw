<div style="border-left: 4px solid #4C72B0; padding: 12px 20px; background-color: rgba(76,114,176,0.08); border-radius: 6px; margin-bottom: 24px;">

**👤 Profil Penulis**

| | |
|---|---|
| **NIM/NPM** | 230411100005 |
| **Nama** | HANIN |
| **Email** | 230411100005@student.trunojoyo.ac.id |
| **Program Studi** | Teknik Informatika |
| **Mata Kuliah** | Pencarian dan Penambangan Web |

</div>

# PPW - Web Search & Mining

Dokumentasi ini disusun untuk tugas mata kuliah **Pencarian dan Penambangan Web (PPW)**, membahas proses pengumpulan, pengolahan, pemodelan, dan penerapan klasifikasi berita berbahasa Indonesia dari [detik.com](https://detik.com) mengikuti kerangka kerja **CRISP-DM**.

**Topik:** Klasifikasi Berita Indonesia - membedakan artikel kategori **Sport** dan **Finance** berdasarkan isi teksnya.

## Pengantar Web Mining

**Web Mining** adalah proses penerapan teknik penambangan data (*data mining*) untuk menemukan pola, informasi, dan pengetahuan yang berguna dari data yang tersedia di web, baik berupa teks, struktur tautan, maupun catatan perilaku pengguna.

Secara umum, web mining terbagi menjadi tiga kategori:

- **Web Content Mining** - menggali informasi dari isi/konten halaman web (teks, gambar, dll). Proyek ini termasuk kategori ini, karena berfokus pada ekstraksi teks artikel berita.
- **Web Structure Mining** - menganalisis struktur tautan (hyperlink) antar halaman web.
- **Web Usage Mining** - mempelajari pola perilaku pengguna dari log aktivitas di suatu situs web.

Dalam proyek ini, teknik **web content mining** diterapkan melalui proses *crawling* untuk mengumpulkan artikel berita dari Detik.com, dilanjutkan dengan ekstraksi teks menggunakan Trafilatura, kemudian teks diolah, diubah menjadi vektor (TF-IDF, PCA, SVD, dan skip-gram), diklasifikasikan dengan Naive Bayes, dan dipasang pada aplikasi Streamlit.

## Struktur Dokumen

- **Business Understanding** - latar belakang masalah dan kriteria keberhasilan proyek
- **Data Understanding** - eksplorasi awal terhadap 200 artikel yang telah dikumpulkan
- **Data Preparation** - crawling, ekstraksi isi artikel, pelabelan, *text preprocessing*, perbandingan tahap preprocessing, vektorisasi (TF-IDF, PCA, SVD), dan *word embedding* skip-gram
- **Modeling** - klasifikasi Naive Bayes pada TF-IDF, PCA, dan SVD, serta klasifikasi dengan skip-gram yang dibandingkan dengan TF-IDF, lengkap dengan Grid Search
- **Evaluation** - perbandingan akurasi semua representasi dan pemilihan model terbaik
- **Deployment** - aplikasi Streamlit untuk mengklasifikasikan berita dari link detik.com

*Hanin Hammoud*
