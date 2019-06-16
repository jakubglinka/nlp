import tensorflow as tf
from typing import Iterator, List, Tuple, Any, Dict
import tqdm
from absl import logging
import numpy as np
from nlp.tokenizer.sentencepiece import Tokenizer

SentenceIterator = Iterator[str]
TextIterator = Iterator[Tuple[Any, SentenceIterator]]
CBOW_TUPLE = Tuple[np.ndarray, np.ndarray, np.ndarray]
# _unk_, _bos_, _eos_, _upp_, _maj_


class BoWEmbeddingMean(tf.keras.layers.Layer):
    def __init__(self, embedding_shape, **kwargs):
        super(BoWEmbeddingMean, self).__init__(**kwargs)
        self.embedding_shape = embedding_shape

    def build(self, inputs_shape):
        self.embedding = self.add_weight(
                shape=self.embedding_shape,
                initializer='glorot_uniform',
                trainable=True,
                dtype=tf.float32,
                name="embedding"
                )
        self._padding_embedding = [tf.zeros(shape=[
                                self.embedding_shape[1]],
                                dtype=tf.float32)]

    def call(self, inputs):
        self.embedding.assign(tf.tensor_scatter_nd_update(
                self.embedding,
                [[0]],
                self._padding_embedding))
        out = tf.nn.embedding_lookup(self.embedding, inputs['bow'])
        out = tf.reduce_sum(out, axis=1, keepdims=False)
        out = out / tf.cast(tf.reshape(inputs['context_size'], [-1, 1]),
                            tf.float32)

        return out


class CBOW_NN(tf.keras.Model):
    """ Continuous Bag of Words model."""

    def __init__(self,
                 embedding_shape: int,
                 name="cbow",
                 **kwargs):
        super(CBOW_NN, self).__init__(name=name, **kwargs)
        self.embedding_shape = embedding_shape
        self.num_tokens = embedding_shape[0]
        self.embedding = BoWEmbeddingMean(embedding_shape=self.embedding_shape,
                                          name="embedding")
        self.nce_initializer = tf.initializers.glorot_normal()
        self.nce_weights = tf.Variable(
                initial_value=self.nce_initializer(shape=self.embedding_shape))
        self.nce_biases = tf.Variable(tf.zeros(shape=self.num_tokens))

    def call(self, inputs):
        out = self.embedding(inputs)
        return out


def create_cbow_tensor_from_sequence(seq: List[int],
                                     window_size: int=10) -> CBOW_TUPLE:

    bows, cws, lens = [], [], []

    for ind_w in range(len(seq)):
        center_word = seq[ind_w]
        bag_of_words = seq[(ind_w - window_size):ind_w] + \
            seq[(ind_w + 1):(ind_w + window_size + 1)]
        bag_len = len(bag_of_words)
        bag_of_words = np.pad(bag_of_words,
                              pad_width=[0, 2*window_size - bag_len],
                              mode="constant")

        cws.append(center_word)
        bows.append(bag_of_words)
        lens.append(bag_len)

    bows = np.vstack(bows)
    cws = np.vstack(cws)
    lens = np.vstack(lens)

    return bows, cws, lens


def create_cbow_tensor_from_corpus(corpus,
                                   tokenizer: Tokenizer,
                                   window_size: int=10,
                                   num_texts=None) -> Dict:

    bows_list, cws_list, lens_list = [], [], []

    n = len(corpus.headers)
    if num_texts is None:
        num_texts = n
    inc = 0

    for info, sents in tqdm.tqdm(corpus.sentences(), total=num_texts):
        text = " ".join(list(sents))
        inc += 1
        if inc > num_texts:
            break
        enc_text = tokenizer.encode_as_ids(text)
        bows, cws, lens = \
            create_cbow_tensor_from_sequence(seq=enc_text,
                                             window_size=window_size)
        bows_list.append(bows)
        cws_list.append(cws)
        lens_list.append(lens)

    bows = np.vstack(bows_list)
    cws = np.vstack(cws_list)
    lens = np.vstack(lens_list)

    return {"bow": bows, "cw": cws, "context_size": lens}
