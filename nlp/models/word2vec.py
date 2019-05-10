from collections import Counter
from typing import Iterator, List, Tuple, Any
import tqdm

SentenceIterator = Iterator[List[str]]
TextIterator = Iterator[Tuple[Any, SentenceIterator]]

# _bos_, _unk_, _eos_
# _up_, _maj_

def encode_sentence(s: List[str], vocab) -> List[str]:

    es = []
    es.append("_bos_")
    for token in s:
        if token.istitle():
            es.append("_maj_")
        elif token.isupper():
            es.append("_up_")
        token = token.lower()
        if vocab[token]:
            es.append(token)
        else:
            es.append("_unk_")

    es.append("_eos_")
    return es


def create_vocabulary(gen: TextIterator) -> Counter:
    cnt = Counter()
    for _, text in tqdm.tqdm(gen):
        for sent in text:
            for token in sent:
                cnt[token.lower()] += 1

    return cnt

def prune_vocabulary(cnt: Counter, min_count=1) -> Counter:
    return Counter({k: c for k, c in cnt.items() if c >= min_count})


# def cbow_tf_record(gen: TextIterator) -> List[Tuple[Context, Labels]]