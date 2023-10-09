import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras.layers import Layer, MultiHeadAttention, Dense, Add, LayerNormalization, Embedding, Dropout


class PositionalEmbedding(Layer):
    def __init__(self, vocab_size, em_units):
        super().__init__()
        self.em_units = em_units
        self.embedding = Embedding(vocab_size, em_units, mask_zero=True)
        self.pos_encoding = self.positional_encoding(
            length=100, depth=em_units)

    def positional_encoding(self, length, depth):
        depth = depth/2

        positions = np.arange(length)[:, np.newaxis]  # shape => (seq, 1)
        depths = np.arange(depth)[np.newaxis, :]/depth  # shape => (1, dim)

        angle_rates = 1 / (10000**depths)  # shape => (1, dim)
        angle_rads = positions * angle_rates  # shape => (pos, dim)

        pos_encoding = np.concatenate(
            [np.sin(angle_rads), np.cos(angle_rads)],
            axis=-1)

        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.em_units, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]

        return x


class BaseAttention(Layer):
    def __init__(self, **kwargs):
        super(BaseAttention, self).__init__()
        self.mha = MultiHeadAttention(**kwargs)
        self.nrm = LayerNormalization()
        self.add = Add()


class CrossAttention(BaseAttention):
    def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query=x,
            key=context,
            value=context,
            return_attention_scores=True)
        self.last_attn_scores = attn_scores
        x = self.add([attn_output, x])
        x = self.nrm(x)

        return x


class SelfAttention(BaseAttention):
    def call(self, x):
        atten_output = self.mha(
            query=x,
            key=x,
            value=x
        )
        x = self.add([x, atten_output])
        x = self.nrm(x)

        return x


class MaskedAttention(BaseAttention):
    def call(self, x):
        atten_output = self.mha(
            query=x,
            key=x,
            value=x,
            use_causal_mask=True
        )
        x = self.add([x, atten_output])
        x = self.nrm(x)

        return x


class FeedForwordLayer(Layer):
    def __init__(self, ffd_units, em_units, drop_out=.1):
        super(FeedForwordLayer, self).__init__()
        self.ffd = keras.models.Sequential([
            Dense(ffd_units, activation='relu'),
            Dense(em_units, activation='relu'),
            Dropout(drop_out)
        ])

        self.add = Add()
        self.nrm = LayerNormalization()

    def call(self, x):
        output = self.ffd(x)
        x = self.add([x, output])
        x = self.nrm(x)

        return x


class EncoderLayer(Layer):
    def __init__(self, num_heads, dim, ffd_units, drop_out):
        super(EncoderLayer, self).__init__()
        self.atn = SelfAttention(num_heads=num_heads, key_dim=dim)
        self.ffd = FeedForwordLayer(ffd_units, dim, drop_out)

    def call(self, x):
        x = self.atn(x)
        x = self.ffd(x)

        return x


class Encoder(Layer):
    def __init__(self, n_layers, vocab_size, dim, num_heads, ffd_units, drop_out):
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.pos = PositionalEmbedding(vocab_size, dim)
        self.layers = [
            EncoderLayer(num_heads, dim, ffd_units, drop_out) for _ in range(n_layers)
        ]

    def call(self, x):
        x = self.pos(x)
        for i in range(self.n_layers):
            x = self.layers[i](x)

        return x


class DecoderLayer(Layer):
    def __init__(self, num_heads, dim, ffd_units, dropout):
        super(DecoderLayer, self).__init__()
        self.mal = MaskedAttention(num_heads=num_heads, key_dim=dim)
        self.cal = CrossAttention(num_heads=num_heads, key_dim=dim)
        self.ffl = FeedForwordLayer(ffd_units, dim, dropout)

    def call(self, x, context):
        x = self.mal(x)
        x = self.cal(x, context)
        x = self.ffl(x)

        return x


class Decoder(Layer):
    def __init__(self, n_layers, vocab_size, dim, num_heads, ffd_units, dropout):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.pel = PositionalEmbedding(vocab_size, dim)
        self.dec = [DecoderLayer(num_heads, dim, ffd_units, dropout)
                    for _ in range(n_layers)]

    def call(self, x, context):
        x = self.pel(x)
        for layer in self.dec:
            x = layer(x, context)

        return x


class Transformer(keras.Model):
    def __init__(self, args):
        super(Transformer, self).__init__()

        self.encoder = Encoder(args.n_layers,
                               args.input_vocab_size,
                               args.d_model,
                               args.num_heads,
                               args.ffd_units,
                               args.dropout)

        self.decoder = Decoder(args.n_layers,
                               args.target_vocab_size,
                               args.d_model,
                               args.num_heads,
                               args.ffd_units,
                               args.dropout)

        self.out = Dense(args.target_vocab_size)

    def call(self, inputs):
        context, x = inputs
        context = self.encoder(context)
        x = self.decoder(x, context)
        logits = self.out(x)

        try:
            del logits._keras_mask
        except AttributeError:
            pass

        return logits
