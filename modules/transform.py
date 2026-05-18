import tensorflow as tf
import tensorflow_transform as tft

FEATURE_KEY = 'text'
LABEL_KEY = 'label'

def transformed_name(key):
    return f"{key}_xf"

def preprocessing_fn(inputs):
    outputs = {}

    # Mengambil data teks
    text = inputs[FEATURE_KEY]
    # Jika TFX mendeteksinya sebagai SparseTensor, ubah menjadi Dense (padat)
    if isinstance(text, tf.SparseTensor):
        text = tf.sparse.to_dense(text, default_value='')
        
    outputs[transformed_name(FEATURE_KEY)] = tf.strings.lower(text)

    # Mengambil data label
    label = inputs[LABEL_KEY]
    # Sama seperti teks, pastikan label tidak berupa SparseTensor
    if isinstance(label, tf.SparseTensor):
        label = tf.sparse.to_dense(label, default_value=0)
        
    outputs[transformed_name(LABEL_KEY)] = tf.cast(label, tf.int64)

    return outputs