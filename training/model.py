from os import environ
from pathlib import Path

environ["TFHUB_CACHE_DIR"] = (Path(__file__).parent / "model_weights").as_posix()

from logging import basicConfig, DEBUG

from tensorflow.python.keras import Input
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.python.keras.layers import Dense, concatenate
from tensorflow.python.keras.models import Model

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

    inputs = input_features + input_texts
    MODULE_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"
    embed = hub.KerasLayer(MODULE_URL)
    embed_narrative = embed(tf.reshape(input_texts[0], [-1]))
    deep = Dense(256, activation='relu')(embed_narrative)
    deep = tf.keras.layers.Dense(64, activation='relu')(deep)
    deep = tf.keras.layers.Dense(16, activation='relu')(deep)
    wide_ff = concatenate(input_features)
    wide = tf.keras.layers.Dense(16, activation='relu')(wide_ff)
    both = concatenate([deep, wide])

    output = Dense(1, activation="sigmoid")(both)
    model = Model(inputs, output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=0.001),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.BinaryAccuracy(),
                 tf.keras.metrics.TruePositives()]
    )
    return model


if __name__ == '__main__':
    basicConfig(level=DEBUG)
    print(get_model())
