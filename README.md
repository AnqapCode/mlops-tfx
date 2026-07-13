# [End-to-End Machine Learning Pipeline]

Proyek ini adalah pipeline Machine Learning (MLOps) modular yang mencakup tahap persiapan data, pelatihan model, evaluasi, hingga pembungkusan model ke dalam container Docker agar siap dideploy secara online.

## 🚀 Struktur Repositori

Berikut adalah penjelasan fungsi dari masing-masing folder dan file dalam proyek ini:

*   **`arifcahyo27-pipeline/`**: Folder utama yang berisi konfigurasi pipa (pipeline) otomatisasi proyek, seperti alur CI/CD atau orkestrasi pipeline.
*   **`modules/`**: Folder modular yang berisi skrip Python terpisah untuk setiap tahapan ML (misalnya: data preprocessing, feature engineering, training, dan evaluation).
*   **`serving_model/`**: Folder yang menyimpan model yang sudah terlatih beserta skrip untuk melayani permintaan prediksi (*inference API/serving*).
*   **`Dockerfile`**: Berisi instruksi untuk membungkus seluruh aplikasi dan model ke dalam Docker image agar bisa dijalankan secara konsisten di lingkungan mana pun.
*   **`requirements.txt`**: Daftar pustaka (*dependencies*) Python yang dibutuhkan untuk menjalankan proyek ini.
*   **`setup_data.py`**: Skrip untuk mengunduh, mengekstrak, atau menyiapkan dataset awal sebelum masuk ke pipeline training.
*   **`notebook.ipynb` & `arifcahyo27-testing.ipynb`**: File Jupyter Notebook yang digunakan untuk eksperimen awal, analisis data eksploratif (EDA), pengujian kode, dan validasi model secara interaktif.

## 🛠️ Alur Kerja (Workflow)

1.  **Persiapan Data**: Menjalankan `setup_data.py` untuk menyiapkan dataset.
2.  **Eksperimen**: Melakukan eksplorasi data dan pembuatan prototipe model di dalam file Notebook (`.ipynb`).
3.  **Modularisasi & Pipeline**: Membagi kode eksperimen menjadi komponen-komponen terpisah di dalam folder `modules/` dan menjalankannya melalui pipeline otomatis.
4.  **Containerization**: Mengemas model dan layanan prediksi (`serving_model/`) menggunakan `Dockerfile` menjadi sebuah Docker image untuk deployment.

## 📦 Cara Menjalankan Proyek

### 1. Instalasi Dependensi
Pastikan Anda sudah menginstal Python, lalu jalankan perintah berikut:
```bash
pip install -r requirements.txt
```

### 2. Menyiapkan Data
```bash
python setup_data.py
```

### 3.Membangun Docker Image
Untuk membungkus proyek ini ke dalam Docker container, jalankan perintah:
```bash
docker build -t nama-image-proyek:latest .
```
