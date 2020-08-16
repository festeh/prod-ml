import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow.python.lib.io.tf_record import TFRecordWriter
import tensorflow as tf

if __name__ == '__main__':
    records_path = "data/tf_records/test.tfrecord"
    with TFRecordWriter(records_path) as w:
        w.write(b"First")
        w.write(b"Second")

    for record in tf.data.TFRecordDataset(records_path):
        print(record)
