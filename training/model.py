from logging import basicConfig, DEBUG

from tensorflow.python.keras import Input
import tensorflow as tf
import tensorflow_hub as hub
from preprocessing.complaints_preporcessing import ONE_HOT_FEATURES, transformed_name, BUCKET_FEATURES, TEXT_FEATURES


def get_model():
    input_features = []
    for name, dim in {**ONE_HOT_FEATURES, **BUCKET_FEATURES}.items():
        input_features.append(Input(shape=(dim + 1),
                                    name=transformed_name(name)))
    input_texts = []
    for name in TEXT_FEATURES:
        input_texts.append(Input(shape=(1,),
                                 name=transformed_name(name),
                                 dtype=tf.string))

    model_inputs = input_features + input_texts
    MODULE_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"
    embed = hub.KerasLayer(MODULE_URL)
    return model_inputs


if __name__ == '__main__':
    basicConfig(level=DEBUG)
    print(get_model())
