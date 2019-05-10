from nlp.corpus import nkjp
import pandas as pd
import absl
from nlp.models import word2vec
absl.logging.set_verbosity(absl.logging.INFO)

corpus = nkjp.NKJP(dir="/tmp/nkjp/").download().parse_headers()

df = pd.DataFrame(corpus.headers)
print(df.head(10))

import hashlib

def sample_dir(dir, frac=1.0):
    return int(hashlib.md5(dir.encode()).hexdigest()[:2], 16) / 255 <= frac

gen = lambda: corpus.tokenized_sentences(filter=lambda d: sample_dir(d["dir"], .95))

MAX = 100
inc = 0
for _, text in gen():
    for sent in text:
        print(sent)
        inc += 1

    if inc > MAX:
        break

vocab = word2vec.create_vocabulary(gen())
vocab = word2vec.prune_vocabulary(vocab, 10)
vocab.most_common(100)

sum(vocab.values())
len(vocab.keys())

vocab["miał"]

sent


encode_sentence(sent, vocab)

sent


keys = ["a", "b"]
values = [1, 2]

table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(keys, values), -1)

import tensorflow as tf

out = table.lookup(tf.convert_to_tensor(["a", "a", "b"]))
out
