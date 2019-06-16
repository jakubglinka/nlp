from nlp.corpus import nkjp
from nlp.tokenizer import sentencepiece
from nlp.lm import word2vec
import pandas as pd
import absl
# from nlp.models import word2vec
import hashlib
import importlib as imp
import tensorflow as tf

absl.logging.set_verbosity(absl.logging.INFO)
imp.reload(nkjp)
imp.reload(sentencepiece)
imp.reload(word2vec)

corpus = nkjp.NKJP(dir="/Users/qba/tmp/nkjp/").download().parse_headers()
df = pd.DataFrame(corpus.headers)
print(df.head(10))

tokenizer = sentencepiece.Tokenizer(1000)
tokenizer.fit(corpus.sentences(), 100)
tokenizer.encode_as_pieces("Jakub kupił nową grę na PS4")

tokenizer.encode_as_pieces("Pysza kocha Mysza! Ale nie lubi!")
ids = tokenizer.encode_as_ids("Pysza kocha Mysza! Ale nie lubi!")
tokenizer.decode_ids(ids)

import numpy as np

seq = tokenizer.encode_as_ids("Pysza kocha Mysza!")
bows, cws, lens = word2vec.create_cbow_tensor_from_sequence(seq, window_size=2)

for cw, bow in zip(cws, bows):
    tokens = " ".join([tokenizer.id_to_piece(int(x)) for x in bow])
    target = tokenizer.id_to_piece(int(cw))
    print(tokens, "-->", target)


res = word2vec.create_cbow_tensor_from_corpus(corpus, tokenizer, num_texts=50)
res["cw"].shape

tf.random.set_seed(2014)
tf.Variable(initial_value=tf.initializers.glorot_normal()(shape=[10, 10]))


dataset = tf.data.Dataset.from_tensor_slices(res)
batch = dataset.batch(64)
inputs = next(iter(batch))

imp.reload(word2vec)
model = word2vec.CBOW_NN(embedding_shape=[1000, 10])
model(inputs)
model.summary()

# Compute the NCE loss, using a sample of the negative labels each time.
loss = tf.reduce_mean(
  tf.nn.nce_loss(weights=model.nce_weights, # this could be embedding
                 biases=model.nce_biases,
                 labels=inputs["cw"],
                 inputs=model(inputs),
                 num_sampled=1000,
                 num_classes=1000))

loss

