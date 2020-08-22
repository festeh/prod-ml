from csv import DictReader

from tensorflow.python.lib.io.tf_record import TFRecordWriter
import tensorflow as tf

from constants import BASE_DIR, DATA_DIR


def bytes_feature(data):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[data.encode()]))


def int_feature(data):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[data]))


if __name__ == '__main__':
    data_dir = DATA_DIR / "complaints"
    sample_path = data_dir / "sample.csv"
    tfrecord_filename = "sample.tfrecord"
    writer = TFRecordWriter((data_dir / tfrecord_filename).as_posix())

    with open(sample_path) as f:
        reader = DictReader(f, quotechar='"')
        for line in reader:
            example = tf.train.Example(
                features=tf.train.Features(
                    feature=dict(
                        product=bytes_feature(line["product"]),
                        sub_product=bytes_feature(line["sub_product"]),
                        issue=bytes_feature(line["issue"]),
                        sub_issue=bytes_feature(line["sub_issue"]),
                        state=bytes_feature(line["state"]),
                        zipcode=int_feature(int(line["zipcode"])),
                        company=bytes_feature(line["company"]),
                        company_response=bytes_feature(line["company_response_to_consumer"]),
                        consumer_complaint_narrative=bytes_feature(line["consumer_complaint_narrative"]),
                        timely_response=bytes_feature(line["timely_response"]),
                        consumer_disputed=bytes_feature(line["consumer_disputed?"])
                    ))
            )
            writer.write(example.SerializeToString())
            writer.close()
