import urllib.request
import pandas as pd
import os

DATA_ROOT = 'data'
if not os.path.exists(DATA_ROOT):
    os.makedirs(DATA_ROOT)

print("Mengunduh dataset SMS Spam...")
url = "https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv"
urllib.request.urlretrieve(url, 'raw_spam.csv')

print("Merapikan format data...")
df = pd.read_csv('raw_spam.csv', encoding='latin-1')

df = df[['v1', 'v2']]
df.columns = ['label', 'text']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Menghapus baris yang kosong (NaN) agar tidak membuat error di komponen Transform
df = df.dropna()

df.to_csv(f'{DATA_ROOT}/data.csv', index=False)
os.remove('raw_spam.csv')

print("Selesai! Dataset berhasil disiapkan di folder 'data/data.csv'")