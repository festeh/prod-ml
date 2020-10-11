from tensorflow import TensorSpec
from tensorflow.python.data.experimental import make_batched_features_dataset
from tensorflow_transform import TFTransformOutput
import tensorflow as tf

from preprocessing.complaints_preporcessing import LABEL_KEY, transformed_name
from training.model import get_model


def read_gz(files):
    return tf.data.TFRecordDataset(filenames=files, compression_type="GZIP")


def input_fn(file_pat, transform_graph: TFTransformOutput, batch_size=32):
    feature_spec = transform_graph.transformed_feature_spec().copy()
    print(file_pat)
    dataset = make_batched_features_dataset(
        file_pat,
        batch_size=batch_size,
        features=feature_spec,
        reader=read_gz,
        label_key=transformed_name(LABEL_KEY),
    )
    return dataset


def get_serve_fn(model, transform_graph):
    features_layer = transform_graph.transform_features_layer()

    @tf.function
    def serve_fn(serialized_examples):
        feature_spec = transform_graph.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(serialized_examples, feature_spec)
        transformed_features = features_layer(parsed_features)
        return {"outputs": model(transformed_features)}

    return serve_fn


def run_fn(args):
    transform_graph = TFTransformOutput(args.transform_output)
    train_dataset = input_fn(args.train_files, transform_graph)
    val_dataset = input_fn(args.eval_files, transform_graph)

    model = get_model()
    model.fit(
        train_dataset,
        steps_per_epoch=args.train_steps,
        validation_data=val_dataset,
        validation_steps=args.eval_steps,
    )
    signatures = {
        "serving_default": get_serve_fn(model, transform_graph).get_concrete_function(
            TensorSpec(shape=[None], dtype=tf.string, name="example")
        )
    }
    model.save(args.serving_model_dir, save_format="tf", signatures=signatures)
