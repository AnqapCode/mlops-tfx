# 1. Menggunakan image dasar resmi dari TensorFlow Serving
FROM tensorflow/serving:latest

# 2. Mengatur variabel lingkungan (Environment Variable)
# Kita menamai model ini 'spam_model' agar mudah dipanggil nanti
ENV MODEL_NAME=spam_model

# 3. Menyalin model dari komputer lokalmu ke dalam sistem Docker
# PENTING: Ganti 'arifcahyo27-pipeline-v6' dengan nama folder pipeline terakhirmu yang berhasil
COPY ./serving_model/arifcahyo27-pipeline /models/spam_model

# 4. Membuka port 8501 (Port standar TF Serving untuk menerima request dari luar)
EXPOSE 8501