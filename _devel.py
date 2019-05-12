from nlp.corpus import nkjp
import pandas as pd
import absl
from nlp.models import word2vec
import hashlib

absl.logging.set_verbosity(absl.logging.INFO)

corpus = nkjp.NKJP(dir="/Users/qba/tmp/nkjp/").download().parse_headers()
df = pd.DataFrame(corpus.headers)
print(df.head(10))


def sample_dir(dir, frac=1.0):
    return int(hashlib.md5(dir.encode()).hexdigest()[:2], 16) / 255 <= frac


def gen():
    return lambda: corpus.tokenized_sentences(
        filter=lambda d: sample_dir(d["dir"], .01))

MAX = 100
inc = 0
for _, text in gen():
    for sent in text:
        print(sent)
        inc += 1

    if inc > MAX:
        break

vocab = word2vec.Vocabulary(gen())
