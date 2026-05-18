import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras import layers

# Nama kolom harus sama persis dengan yang ada di transform.py
FEATURE_KEY = 'text'
LABEL_KEY = 'label'

def transformed_name(key):
    """Menambahkan akhiran '_xf' pada fitur agar sesuai dengan output Transform."""
    return f"{key}_xf"

def gzip_reader_fn(filenames):
    """Fungsi untuk membaca file data yang dikompresi oleh TFX."""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def input_fn(file_pattern, tf_transform_output, num_epochs, batch_size=64):
    """Fungsi untuk menyuplai data ke dalam model secara bertahap (batch)."""
    transform_feature_spec = tf_transform_output.transformed_feature_spec().copy()
    
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY)
    )
    return dataset

def build_keras_model():
    """Membangun arsitektur model Neural Network."""
    # Lapisan Input: Menerima teks
    inputs = tf.keras.Input(shape=(1,), name=transformed_name(FEATURE_KEY), dtype=tf.string)
    
    # Lapisan Vektorisasi: Mengubah teks menjadi angka
    vectorize_layer = layers.TextVectorization(
        max_tokens=5000,
        output_mode='int',
        output_sequence_length=100
    )
    
    # Meratakan input agar sesuai dengan format TextVectorization
    x = tf.reshape(inputs, [-1])
    x = vectorize_layer(x)
    
    # Lapisan Hidden (Otak Model): Mencari pola dari angka-angka
    x = layers.Embedding(5000, 64, name="embedding")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation='relu')(x)
    
    # Lapisan Output: Mengeluarkan probabilitas (0 hingga 1)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    return model, vectorize_layer

def run_fn(fn_args):
    """
    Fungsi untuk TFX Trainer.
    Fungsi ini akan dipanggil otomatis oleh pipeline untuk memulai pelatihan.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    
    # Menyiapkan data training dan evaluasi
    train_dataset = input_fn(fn_args.train_files, tf_transform_output, num_epochs=10)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, num_epochs=1)
    
    # Membangun model
    model, vectorize_layer = build_keras_model()
    
    # Beradaptasi dengan kata-kata yang ada di data training
    train_text_dataset = train_dataset.map(lambda x, y: x[transformed_name(FEATURE_KEY)])
    vectorize_layer.adapt(train_text_dataset)
    
    # Memulai proses pelatihan
    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps
    )
    
    # Menyimpan hasil model agar bisa digunakan di tahap Pusher
    model.save(fn_args.serving_model_dir, save_format='tf')