# Deployment

Model terbaik dari tahap Evaluation dipasang pada aplikasi web sederhana dengan **Streamlit**. Pengguna menempelkan link artikel dari `finance.detik.com` atau `sport.detik.com`, lalu aplikasi menampilkan kategorinya: **Finance** atau **Sport**.

## Alur Aplikasi

```
Link artikel  ->  ambil halaman (requests + BeautifulSoup)  ->  preprocessing
              ->  kata menjadi indeks (word2idx)  ->  vektor kata (W1)
              ->  rata-rata vektor = vektor dokumen  ->  GaussianNB  ->  Finance / Sport
```

Aplikasi juga punya tab **Tempel Teks** untuk menempelkan isi berita langsung, dan menolak link dari kanal selain finance dan sport.

## File yang Dipakai

Semuanya ada di folder `deployment/`.

| File | Fungsi |
|---|---|
| `app.py` | Kode aplikasi Streamlit |
| `model/model_final.joblib` | Model GaussianNB hasil Grid Search |
| `model/embedding_W1.npy` | Matriks embedding skip-gram (4.697 kata x 50 dimensi) |
| `model/word2idx.joblib` | Kamus kata ke nomor baris pada `W1` |
| `.streamlit/config.toml` | Pengaturan tampilan |

Ketiga file di `model/` harus berasal dari satu kali run notebook **Klasifikasi Skip-gram vs TF-IDF**. Aplikasi memeriksa ukurannya dan menampilkan pesan jika tidak cocok.

## Menjalankan di Komputer Sendiri

```bash
pip install -r requirements.txt
streamlit run deployment/app.py
```

## Deploy ke Streamlit Community Cloud

1. Push repo ke GitHub (cabang `main`).
2. Buka [share.streamlit.io](https://share.streamlit.io) dan login dengan akun GitHub.
3. Klik **Create app**, pilih repo `haninhammoud01/ppw`, cabang `main`, dan **Main file path** `deployment/app.py`.
4. Klik **Deploy**. Link aplikasi berbentuk `https://<nama-app>.streamlit.app`.
