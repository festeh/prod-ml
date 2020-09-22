from tensorflow_transform.tf_metadata.dataset_metadata import DatasetMetadata
from tensorflow_transform.tf_metadata.schema_utils import schema_from_feature_spec
import tensorflow as tf
import apache_beam as beam
import tensorflow_transform.beam as tft_beam
from tfx.components import ImportExampleGen
from tfx.utils.dsl_utils import external_input


def convert_image_to_grayscale(inputs):
    img = inputs['img']
    img = tf.sparse.to_dense(img)
    raw_image = tf.reshape(img, [-1])
    return {"img_gs": raw_image}


def save_images_to_tfrecord(images_paths, save_path):
    with tf.io.TFRecordWriter(save_path) as writer:
        for img_path in images_paths:
            img = tf.io.read_file(img_path.resolve().as_posix())
            example = tf.train.Example(features=tf.train.Features(feature={
                'img': tf.train.Feature(bytes_list=tf.train.FloatList(value=[img]))
            }))
            writer.write(example.SerializeToString())


class ImageConverter:
    def __init__(self):
        self.metadata = DatasetMetadata(
            schema_from_feature_spec({
                'img': tf.io.VarLenFeature(tf.int64)
            }
            ))

    def convert(self, images):
        dataset = [{"img": img for img in images}]
        with beam.Pipeline() as pipeline:
            with tft_beam.Context(temp_dir="tmp"):
                transformed_dataset, transform_fn = (
                        (dataset, self.metadata) | tft_beam.AnalyzeAndTransformDataset(convert_image_to_grayscale))
                return transformed_dataset
