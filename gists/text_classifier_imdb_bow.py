# Copyright 2019 Jakub Glinka. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
IMDB Sentiment classifier using bag of words features extraction.

    Dependencies:
        libraries:
            tensorflow==2.0.0
            tensorflow-datasets
            scikit-learn==0.21.2
"""

from absl import app
from absl import flags
from absl import logging
import tqdm

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.feature_extraction.text import CountVectorizer

from typing import List, Generator, Tuple


class BoWFeaturesExtractor:
    """BoW Feature Encoder."""
    def __init__(self, texts: Generator[str, None, None]):
        self.cv = CountVectorizer().fit(texts)

    def __call__(self, texts: Generator[str, None, None]):
        return self.cv.transform(texts)


def gather_data(data: Generator[Tuple[str, int], None, None]
                ) -> Tuple[List[str], List[int]]:
    """Transform List of Tuples into Tuple of Lists."""

    labels = []
    texts = []
    for txt, lab in tqdm.tqdm(data, total=None, ncols=100):
        labels.append(lab)
        texts.append(txt)

    return texts, labels


def prepare_dataset(texts: List[str],
                    labels: List[int],
                    bfe: BoWFeaturesExtractor) -> tf.data.Dataset:
    """Prepare dataset for TensorFlow model."""
    mat = bfe(texts).toarray()
    mat = np.array(mat, dtype="float64")
    dataset = tf.data.Dataset.from_tensor_slices((mat, labels))
    return dataset


class GLM(tf.keras.Model):
    def __init__(self, lambda_: float = 0.0):
        super(GLM, self).__init__()
        self.dense = tf.keras.layers.\
            Dense(units=1,
                  kernel_regularizer=tf.keras.regularizers.l1(lambda_),
                  activation="sigmoid")

    def call(self, x):
        out = self.dense(x)
        return out


def main(__):
    logging.set_verbosity(logging.ERROR)

    logging.info("Preparing data...")
    train_validation_split = tfds.Split.TRAIN.subsplit([6, 4])
    imdb = tfds.load(name="imdb_reviews",
                     as_supervised=True,
                     split=(train_validation_split, tfds.Split.TEST))

    (train, valid), test = tfds.as_numpy(imdb)
    train = gather_data(train)
    valid = gather_data(valid)
    test = gather_data(test)

    logging.info("BoWFeaturesExtractor...")
    bfe = BoWFeaturesExtractor(train[0])

    n = len(bfe.cv.vocabulary_)
    logging.info("Detected vocabulary with {} unique tokens..."
                 .format(n))

    train_dataset = prepare_dataset(*train, bfe)
    train_dataset = train_dataset.shuffle(100000).batch(32)

    valid_dataset = prepare_dataset(*valid, bfe)
    valid_dataset = valid_dataset.shuffle(100000).batch(32)

    model = GLM(0.000001)
    model.compile(loss=tf.losses.BinaryCrossentropy(), optimizer="Adam", metrics=["binary_accuracy"])
    model.fit(train_dataset, epochs=10, validation_data=valid_dataset)

    print(model.evaluate(valid_dataset))

if __name__ == "__main__":
    app.run(main)

