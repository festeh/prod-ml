import os

from constants import DATA_DIR

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow.python.lib.io.tf_record import TFRecordWriter
import tensorflow as tf

if __name__ == '__main__':
    records_path = DATA_DIR / "tf_records/test.tfrecord"
    with TFRecordWriter(records_path.as_posix()) as w:
        w.write(b"First")
        w.write(b"Second")

    for record in tf.data.TFRecordDataset(records_path.as_posix()):
        print(record)
