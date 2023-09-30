import numpy as np

import tensorflow as tf
from model import Transformer

from dataclasses import dataclass

input_vect = tf.keras.models.load_model('./text vectorizer/input_vectorizer')
target_vect = tf.keras.models.load_model('./text vectorizer/target_vectorizer')

@dataclass
class Config:
    n_layers: int = 6
    input_vocab_size: int = 26673
    target_vocab_size: int = 14163
    d_model: int = 256
    num_heads: int = 6
    ffd_units: int = 256
    dropout: float = .3

args = Config()

vocab = target_vect.layers[0].get_vocabulary()

idx_to_wrd = {idx:wrd for idx, wrd in enumerate(vocab)}

model = Transformer(args)
model((input_vect(tf.constant(['hi'])), target_vect(tf.constant(['by']))))
model.load_weights('./weights/model_weights.h5')

input_lang = '[SOS] Yo soy humano . [EOS]'
input_prepared = tf.constant([input_lang])

encoded_input = input_vect(input_prepared)
tokens = '[SOS]'
encoded_target = target_vect(tf.constant([tokens]))
for i in range(5):
    outputs = model((encoded_input, encoded_target))
    preds = outputs[:, -1, :]
    idx = tf.argmax(preds, axis=-1)[:, None]
    wrd = idx_to_wrd[idx.numpy()[0][0]]
    print(wrd, end=' ')
    if wrd == '[EOS]':
        break
    encoded_target = tf.concat([encoded_target, idx], axis=1)
    tokens = tokens + ' ' + wrd
print()
print(tokens)




