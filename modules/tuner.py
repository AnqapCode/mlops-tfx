import tensorflow as tf
import tensorflow_transform as tft
import keras_tuner as kt
from tensorflow.keras import layers
from typing import NamedTuple, Dict, Any
from tfx.components.trainer.fn_args_utils import FnArgs

# KITA MENAMBAHKAN IMPORT INI UNTUK MEMPERBAIKI ERROR
from keras_tuner.engine.base_tuner import BaseTuner 

FEATURE_KEY = 'text'
LABEL_KEY = 'label'

def transformed_name(key):
    return f"{key}_xf"

def gzip_reader_fn(filenames):
    """Membaca file data yang dikompresi."""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def input_fn(file_pattern, tf_transform_output, num_epochs=1, batch_size=64):
    """Menyuplai data dalam bentuk batch."""
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

TunerFnResult = NamedTuple('TunerFnResult', [('tuner', BaseTuner), ('fit_kwargs', Dict[str, Any])])

def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    """Fungsi utama yang akan dieksekusi oleh komponen Tuner."""
    
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    train_dataset = input_fn(fn_args.train_files, tf_transform_output, num_epochs=5)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, num_epochs=1)

    vectorize_layer = layers.TextVectorization(
        max_tokens=5000, 
        output_mode='int', 
        output_sequence_length=100
    )
    train_text_dataset = train_dataset.map(lambda x, y: x[transformed_name(FEATURE_KEY)])
    vectorize_layer.adapt(train_text_dataset)

    def build_model(hp):
        inputs = tf.keras.Input(shape=(1,), name=transformed_name(FEATURE_KEY), dtype=tf.string)
        x = tf.reshape(inputs, [-1])
        x = vectorize_layer(x)
        
        embed_dim = hp.Int('embed_dim', min_value=16, max_value=64, step=16)
        x = layers.Embedding(5000, embed_dim, name="embedding")(x)
        x = layers.GlobalAveragePooling1D()(x)
        
        dense_units = hp.Int('dense_units', min_value=16, max_value=64, step=16)
        x = layers.Dense(dense_units, activation='relu')(x)
        
        outputs = layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        model.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=['accuracy']
        )
        return model

    tuner = kt.RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=3,
        directory=fn_args.working_dir,
        project_name='sms_spam_tuning'
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            'x': train_dataset,
            'validation_data': eval_dataset,
            'steps_per_epoch': fn_args.train_steps,
            'validation_steps': fn_args.eval_steps
        }
    )