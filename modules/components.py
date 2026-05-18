import os
import tensorflow_model_analysis as tfma
from tfx.components import (
    CsvExampleGen,
    StatisticsGen,
    SchemaGen,
    ExampleValidator,
    Transform,
    Tuner,
    Trainer,
    Evaluator,
    Pusher
)
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing

def init_components(data_dir, transform_module, trainer_module, tuner_module, training_steps, eval_steps, serving_model_dir):
    """
    Fungsi untuk menginisialisasi 9 Komponen Wajib TFX + Tuner
    """
    example_gen = CsvExampleGen(input_base=data_dir)
    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
    schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'])
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema']
    )
    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=transform_module
    )
    tuner = Tuner(
        module_file=tuner_module,
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        train_args={'num_steps': training_steps},
        eval_args={'num_steps': eval_steps}
    )
    trainer = Trainer(
        module_file=trainer_module,
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        hyperparameters=tuner.outputs['best_hyperparameters'],
        train_args={'num_steps': training_steps},
        eval_args={'num_steps': eval_steps}
    )
    model_resolver = Resolver(
        strategy_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing)
    ).with_id('latest_blessed_model_resolver')
    
    eval_config = tfma.EvalConfig(
        model_specs=[
            # Beri tahu evaluator bahwa kolom target kita bernama 'label_xf'
            tfma.ModelSpec(label_key='label')
        ],
        slicing_specs=[
            # Evaluasi dilakukan secara menyeluruh (tanpa dipecah berdasarkan kategori tertentu)
            tfma.SlicingSpec()
        ],
        metrics_specs=[
            tfma.MetricsSpec(metrics=[
                tfma.MetricConfig(class_name='ExampleCount'),
                # Menilai akurasi. Harus > 50% dan lebih baik dari model sebelumnya
                tfma.MetricConfig(class_name='BinaryAccuracy',
                    threshold=tfma.MetricThreshold(
                        value_threshold=tfma.GenericValueThreshold(
                            lower_bound={'value': 0.5}
                        ),
                        change_threshold=tfma.GenericChangeThreshold(
                            direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                            absolute={'value': -1e-10}
                        )
                    )
                )
            ])
        ]
    )

    # Memasukkan eval_config ke dalam komponen Evaluator
    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config
    )

    pusher = Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination={'filesystem': {'base_directory': serving_model_dir}}
    )
    
    return (
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        tuner,
        trainer,
        model_resolver,
        evaluator,
        pusher
    )