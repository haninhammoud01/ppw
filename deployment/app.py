"""
Aplikasi Klasifikasi Berita detik.com: Sport vs Finance
Model: Skip-gram manual (embedding W1) + Gaussian Naive Bayes

Jalankan:  streamlit run app.py
"""
import re
import string
import unicodedata
from pathlib import Path

from urllib.parse import urlparse

import joblib
import numpy as np
import requests
import streamlit as st
from bs4 import BeautifulSoup

FOLDER_MODEL = Path(__file__).parent / "model"


# ------------------------------------------------------------------
# 1. MUAT MODEL (sekali saja, di-cache)
# ------------------------------------------------------------------
@st.cache_resource
def muat_model():
    model = joblib.load(FOLDER_MODEL / "model_final.joblib")
    word2idx = joblib.load(FOLDER_MODEL / "word2idx.joblib")
    W1 = np.load(FOLDER_MODEL / "embedding_W1.npy")
    return model, word2idx, W1


# ------------------------------------------------------------------
# 2. PREPROCESSING (sama persis dengan notebook Preprocessing, tahap 1, 3, 4, 5)
# ------------------------------------------------------------------
POLA_EMOJI = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U0001F900-\U0001F9FF"
    "\U00002600-\U000026FF"
    "]+",
    flags=re.UNICODE,
)

POLA_ARTEFAK = {
    "mata_uang": r"\bRp\.?\s?[\d.,]+(?:\s?(?:juta|miliar|triliun|ribu))?\b|\bUS\$\s?[\d.,]+\b",
    "komposit_angka": r"\b(?=[A-Za-z0-9-]*[A-Za-z])(?=[A-Za-z0-9-]*[0-9])[A-Za-z0-9-]{2,}\b",
    "angka_romawi": r"\b(?=[MDCLXVI])M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\b",
    "url_inline": r"https?://\S+|www\.\S+",
}


def normalisasi_non_ascii(teks):
    teks = POLA_EMOJI.sub(" ", teks)
    return unicodedata.normalize("NFKD", teks).encode("ascii", "ignore").decode("utf-8")


def angka_ke_kode_huruf(n):
    huruf = ""
    n += 1
    while n > 0:
        n -= 1
        huruf = string.ascii_lowercase[n % 26] + huruf
        n //= 26
    return huruf


def lindungi_artefak(teks):
    ditemukan = {}
    urutan = 0

    def buat_pengganti(kategori):
        def pengganti(m):
            nonlocal urutan
            asli = m.group(0)
            nilai_akhir = "urltautan" if kategori == "url_inline" else re.sub(r"[^a-zA-Z0-9]", "", asli).lower()
            kode = "protectedtoken" + angka_ke_kode_huruf(urutan)
            urutan += 1
            ditemukan[kode] = nilai_akhir
            return f" {kode} "
        return pengganti

    for kategori, pola in POLA_ARTEFAK.items():
        teks = re.sub(pola, buat_pengganti(kategori), teks)
    return teks, ditemukan


def pulihkan_artefak(teks, ditemukan):
    for kode, nilai in ditemukan.items():
        teks = teks.replace(kode, nilai)
    return teks


def case_folding_dan_bersihkan(teks):
    teks = teks.lower()
    teks = re.sub(r"[^a-z\s]", " ", teks)
    return re.sub(r"\s+", " ", teks).strip()


def buang_token_terlalu_pendek(teks, panjang_minimum=2):
    return " ".join(w for w in teks.split() if len(w) >= panjang_minimum)


def bersihkan_teks(teks):
    teks = normalisasi_non_ascii(teks)
    teks, artefak = lindungi_artefak(teks)
    teks = case_folding_dan_bersihkan(teks)
    teks = pulihkan_artefak(teks, artefak)
    return buang_token_terlalu_pendek(teks)


# ------------------------------------------------------------------
# 3. TEKS -> VEKTOR DOKUMEN (rata-rata vektor kata, seperti saat training)
# ------------------------------------------------------------------
def teks_ke_vektor(teks_bersih, word2idx, W1):
    token = teks_bersih.split()
    indeks = [word2idx[k] for k in token if k in word2idx]
    if len(indeks) == 0:
        return None, 0, len(token)
    return W1[indeks].mean(axis=0), len(indeks), len(token)


# ------------------------------------------------------------------
# 4. AMBIL ARTIKEL DARI LINK detik.com
# ------------------------------------------------------------------
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
KANAL_URL = {"finance.detik.com": "finance", "sport.detik.com": "sport"}
AWALAN_NOISE = ("advertisement", "scroll to continue", "baca juga", "simak video", "saksikan live", "baca selengkapnya")


def cek_host_detik(url):
    """Kembalikan nama host jika URL valid dan milik detik.com, selain itu None."""
    p = urlparse(url.strip())
    host = (p.hostname or "").lower()
    if p.scheme in ("http", "https") and (host == "detik.com" or host.endswith(".detik.com")):
        return host
    return None


def host_dari_teks(teks):
    """Cari asal berita di teks tempelan. Salinan berita detik.com biasanya memuat baris
    'Baca artikel detik... selengkapnya https://<kanal>.detik.com/...' di bagian bawah."""
    m = re.search(r"baca artikel detik[^\n]*?(https?://\S+)", teks, flags=re.IGNORECASE)
    if m:
        return cek_host_detik(m.group(1))
    semua = re.findall(r"https?://(?:[a-z0-9-]+\.)*detik\.com", teks, flags=re.IGNORECASE)
    semua = [urlparse(u).hostname for u in semua]
    semua = [h for h in semua if h and h != "apps.detik.com"]  # apps.detik.com = promosi aplikasi
    return semua[0].lower() if semua else None


def buang_baris_promosi(teks):
    awalan = ("baca artikel detik", "download apps detik")
    baris = [b for b in teks.splitlines() if not b.strip().lower().startswith(awalan)]
    return "\n".join(baris).strip()


def ambil_artikel(url):
    """Unduh halaman artikel, lalu kembalikan (judul, isi_berita)."""
    url = url.strip()
    if "single=1" not in url:  # detik memecah artikel panjang jadi beberapa halaman
        url += ("&" if "?" in url else "?") + "single=1"

    respons = requests.get(url, headers=HEADERS, timeout=15)
    respons.raise_for_status()
    soup = BeautifulSoup(respons.text, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()

    judul_tag = soup.select_one("h1.detail__title") or soup.find("h1")
    judul = judul_tag.get_text(" ", strip=True) if judul_tag else "(judul tidak ditemukan)"

    wadah = soup.select_one("div.detail__body-text") or soup.find("article") or soup
    paragraf = [p.get_text(" ", strip=True) for p in wadah.find_all("p")]
    paragraf = [p for p in paragraf if p and not p.lower().startswith(AWALAN_NOISE)]
    isi = " ".join(paragraf)

    if len(isi) < 100:
        raise ValueError("Isi artikel tidak ditemukan di halaman ini.")
    return judul, isi


# ------------------------------------------------------------------
# 5. TAMPILAN
# ------------------------------------------------------------------
NAMA_TAMPIL = {
    "sport": ("⚽", "Sport"),
    "finance": ("💰", "Finance"),
}

st.set_page_config(page_title="Klasifikasi Berita detik.com", page_icon="📰", layout="wide")
st.title("📰 Klasifikasi Berita detik.com: Finance atau Sport?")
st.write("Tempel link artikel dari **finance.detik.com** atau **sport.detik.com**, lalu klik **Klasifikasikan**.")

try:
    model, word2idx, W1 = muat_model()
except FileNotFoundError as e:
    st.error(f"File model belum ada: {e.filename}. Salin semua file ke folder `model/`.")
    st.stop()

if W1.shape != (len(word2idx), model.theta_.shape[1]):
    st.error(
        f"Ukuran embedding {W1.shape} tidak cocok dengan word2idx ({len(word2idx)} kata) "
        f"dan model ({model.theta_.shape[1]} fitur). Pastikan ketiga file berasal dari satu kali run notebook."
    )
    st.stop()


def tampilkan_hasil(judul, isi, kanal_url=None):
    teks_bersih = bersihkan_teks(isi)
    vektor, dikenal, total = teks_ke_vektor(teks_bersih, word2idx, W1)

    kiri, kanan = st.columns([3, 2], gap="large")

    with kiri:
        if judul:
            st.subheader(judul)
        with st.container(height=360, border=True):
            st.write(isi)

    with kanan:
        if vektor is None or dikenal < 5:
            st.warning("Teksnya terlalu pendek atau banyak kata yang tidak dikenali, jadi hasilnya tidak bisa dipercaya.")
            return

        x = vektor.reshape(1, -1)
        prediksi = model.predict(x)[0]
        probabilitas = dict(zip(model.classes_, model.predict_proba(x)[0]))
        emoji, nama = NAMA_TAMPIL.get(prediksi, ("", prediksi))
        st.success(f"## {emoji} Berita {nama}")

        if kanal_url:
            if kanal_url == prediksi:
                st.info(f"✅ Sesuai dengan kanal di link ({NAMA_TAMPIL[kanal_url][1]}).")
            else:
                st.info(f"⚠️ Kanal di link adalah {NAMA_TAMPIL[kanal_url][1]}, tetapi model memprediksi {nama}.")

        with st.expander("Detail teknis"):
            for kelas, p in probabilitas.items():
                st.write(f"Peluang {NAMA_TAMPIL.get(kelas, ('', kelas))[1]}: {p * 100:.1f}%")
            st.write(f"Kata dikenali model: {dikenal} dari {total}")
            st.write("Teks setelah preprocessing:")
            st.write(teks_bersih)


tab_link, tab_teks = st.tabs(["🔗 Dari Link", "✍️ Tempel Teks"])

with tab_link:
    # st.form: tombol selalu aktif, dan menekan Enter di kotak link langsung mengirim
    with st.form("form_link"):
        url = st.text_input(
            "Link artikel",
            placeholder="https://finance.detik.com/... atau https://sport.detik.com/...",
        )
        kirim_link = st.form_submit_button("Klasifikasikan", type="primary")

    if kirim_link:
        if not url.strip():
            st.warning("Tempel link artikelnya dulu ya.")
        elif cek_host_detik(url) is None:
            st.error("Link harus berasal dari detik.com (finance.detik.com atau sport.detik.com).")
        elif cek_host_detik(url) not in KANAL_URL:
            st.error(
                f"Link dari **{cek_host_detik(url)}** tidak didukung. Model ini hanya dilatih untuk berita "
                "**Finance** dan **Sport**, jadi gunakan link dari finance.detik.com atau sport.detik.com."
            )
        else:
            host = cek_host_detik(url)
            try:
                with st.spinner("Mengambil artikel..."):
                    judul, isi = ambil_artikel(url)
            except requests.exceptions.RequestException:
                st.error("Gagal membuka link. Periksa linknya dan koneksi internet, atau coba tab **Tempel Teks**.")
            except ValueError as e:
                st.error(f"{e} Coba tab **Tempel Teks**.")
            else:
                tampilkan_hasil(judul, isi, KANAL_URL.get(host))

with tab_teks:
    with st.form("form_teks"):
        teks_manual = st.text_area(
            "Isi berita",
            height=200,
            placeholder="Tempel teks berita finance atau sport detik.com di sini...",
            label_visibility="collapsed",
        )
        kirim_teks = st.form_submit_button("Klasifikasikan", type="primary")

    if kirim_teks:
        if not teks_manual.strip():
            st.warning("Tempel teks beritanya dulu ya.")
        else:
            host_teks = host_dari_teks(teks_manual)
            if host_teks is not None and host_teks not in KANAL_URL:
                st.error(
                    f"Teks ini berasal dari **{host_teks}**, bukan finance/sport. Model ini hanya dilatih untuk "
                    "berita **Finance** dan **Sport**, jadi teks ini tidak diklasifikasikan."
                )
            else:
                tampilkan_hasil(None, buang_baris_promosi(teks_manual), KANAL_URL.get(host_teks))

st.caption("Model hanya mengenal dua kategori: finance dan sport.")
