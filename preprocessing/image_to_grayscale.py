from tensorflow.python.ops.map_fn import map_fn
from tensorflow_transform.tf_metadata.dataset_metadata import DatasetMetadata
from tensorflow_transform.tf_metadata.schema_utils import schema_from_feature_spec
import tensorflow as tf
import apache_beam as beam
import tensorflow_transform.beam as tft_beam
import tensorflow_transform as tft


# tf.function
def convert_image_to_grayscale(img_bytes):
    image = tf.io.decode_jpeg(img_bytes, channels=3)
    gayscale_image = tf.image.rgb_to_grayscale(
        image, name=None
    )
    return tf.io.encode_jpeg(gayscale_image)


def preprocessing_fn(inputs):
    return {"img_gs": map_fn(convert_image_to_grayscale, inputs['img'])}


def save_images_to_tfrecord(images_paths, save_path):
    with tf.io.TFRecordWriter(save_path) as writer:
        for img_path in images_paths:
            img = tf.io.read_file(img_path.resolve().as_posix())
            example = tf.train.Example(features=tf.train.Features(feature={
                'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.numpy()]))
            }))
            writer.write(example.SerializeToString())


class ImageConverter:
    def __init__(self):
        self.metadata = DatasetMetadata(
            schema_from_feature_spec({
                'img': tf.io.FixedLenFeature([], tf.string)
            }
            ))

    def convert(self, images):
        dataset = [{"img": img} for img in images]
        with beam.Pipeline() as pipeline:
            with tft_beam.Context(temp_dir="tmp"):
                transformed_dataset, transform_fn = (
                        (dataset, self.metadata) | tft_beam.AnalyzeAndTransformDataset(preprocessing_fn))
                return transformed_dataset[0]
