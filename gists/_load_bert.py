import tensorflow as tf
print(tf.__version__)
from models.official.bert import modeling
from models.official.bert import tokenization


bert_pretrained_path = "/Users/Qba/Downloads/cased_L-12_H-768_A-12/bert_model.ckpt"
bert_vocab = "/Users/Qba/Downloads/cased_L-12_H-768_A-12/vocab.txt"
bert_config = modeling.BertConfig.from_json_file("/Users/Qba/Downloads/cased_L-12_H-768_A-12/bert_config.json")
bert_config.to_dict()

# The convention in BERT is:
# (a) For sequence pairs:
#  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
#  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
# (b) For single sequences:
#  tokens:   [CLS] the dog is hairy . [SEP]
#  type_ids: 0     0   0   0  0     0 0
#
# Where "type_ids" are used to indicate whether this is the first
# sequence or the second sequence. The embedding vectors for `type=0` and
# `type=1` were learned during pre-training and are added to the wordpiece
# embedding vector (and position vector). This is not *strictly* necessary
# since the [SEP] token unambiguously separates the sequences, but it makes
# it easier for the model to learn the concept of sequences.
#
# For classification tasks, the first vector (corresponding to [CLS]) is
# used as the "sentence vector". Note that this only makes sense because
# the entire model is fine-tuned.

tokenizer = tokenization.FullTokenizer(vocab_file=bert_vocab, do_lower_case=False)
tokens = tokenizer.tokenize("Jacob went for a hike!")
tokens = ["[CLS]"] + tokens + ["[SEP]"]
tokens

input_word_ids = tokenizer.convert_tokens_to_ids(tokens)
tokenizer.convert_ids_to_tokens(input_word_ids)

# model = tf.keras.Model()


def get_bert_encoder_model(input_word_ids,
                   input_mask,
                   input_type_ids,
                   config=None,
                   name=None,
                   float_type=tf.float32):
  """Wraps the core BERT model as a keras.Model."""
  bert_model_layer = modeling.BertModel(config=config, float_type=float_type, name=name)
  output = bert_model_layer(input_word_ids, input_mask, input_type_ids, mode="encoder")
  bert_model = tf.keras.Model(
      inputs=[input_word_ids, input_mask, input_type_ids],
      outputs=output)
  return bert_model


input_word_ids = tf.keras.Input(shape=(6,), name='tokens', dtype="int32")
input_mask = tf.keras.Input(shape=(6,), name='mask', dtype="int32")
input_type_ids = tf.keras.Input(shape=(6,), name='type', dtype="int32")

model = get_bert_encoder_model(input_word_ids, input_mask, input_type_ids, config=bert_config)
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore(bert_pretrained_path)

ids1 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("Jacob went for a hike."))
ids1
ids2 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("Alan is on a bike."))
ids2

input_word_ids = tf.constant([ids1, ids2])
input_mask = tf.constant([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
input_type_ids = tf.constant([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])

model

res = model(inputs=[input_word_ids, input_mask, input_type_ids])
res = res[0]
res = np.sum(res, axis=1)

import numpy as np
np.dot(res, np.transpose(res))

