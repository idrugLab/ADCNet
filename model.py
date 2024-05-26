from cProfile import label
import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt



def gelu(x):
    return 0.5 * x * (1.0 + tf.math.erf(x / tf.sqrt(2.)))


def scaled_dot_product_attention(q, k, v, mask,adjoin_matrix):

    matmul_qk = tf.matmul(q, k, transpose_b=True)


    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)


    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    if adjoin_matrix is not None:
        scaled_attention_logits += adjoin_matrix

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1) 

    output = tf.matmul(attention_weights, v) 

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask,adjoin_matrix):
        batch_size = tf.shape(q)[0]

        q = self.wq(q) 
        k = self.wk(k) 
        v = self.wv(v)  

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size) 

        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask,adjoin_matrix)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model)) 

        output = self.dense(concat_attention)

        return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation=gelu),
        tf.keras.layers.Dense(d_model) 
        ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask,adjoin_matrix):
        attn_output, attention_weights = self.mha(x, x, x, mask,adjoin_matrix)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1) 
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output) 

        return out2,attention_weights



class Encoder(tf.keras.Model): 
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                    maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask,adjoin_matrix):
        seq_len = tf.shape(x)[1]
        adjoin_matrix = adjoin_matrix[:,tf.newaxis,:,:]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x,attention_weights = self.enc_layers[i](x, training, mask,adjoin_matrix)
        return x

class Encoder_test(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Encoder_test, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask,adjoin_matrix):
        seq_len = tf.shape(x)[1]
        adjoin_matrix = adjoin_matrix[:,tf.newaxis,:,:]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        x = self.dropout(x, training=training)
        attention_weights_list = []
        xs = []

        for i in range(self.num_layers):
            x,attention_weights = self.enc_layers[i](x, training, mask,adjoin_matrix)
            attention_weights_list.append(attention_weights)
            xs.append(x)

        return x,attention_weights_list,xs

class BertModel_test(tf.keras.Model):
    def __init__(self,num_layers = 6,d_model = 256,dff = 512,num_heads = 8,vocab_size = 18,dropout_rate = 0.1):
        super(BertModel_test, self).__init__()
        self.encoder = Encoder_test(num_layers=num_layers,d_model=d_model,
                        num_heads=num_heads,dff=dff,input_vocab_size=vocab_size,maximum_position_encoding=200,rate=dropout_rate)
        self.fc1 = tf.keras.layers.Dense(d_model, activation=gelu)
        self.layernorm = tf.keras.layers.LayerNormalization(-1)
        self.fc2 = tf.keras.layers.Dense(vocab_size)
    def call(self,x,adjoin_matrix,mask,training=False):
        x,att,xs = self.encoder(x,training=training,mask=mask,adjoin_matrix=adjoin_matrix)
        x = self.fc1(x)
        x = self.layernorm(x)
        x = self.fc2(x)
        return x,att,xs


class BertModel(tf.keras.Model):
    def __init__(self,num_layers = 6,d_model = 256,dff = 512,num_heads = 8,vocab_size = 18,dropout_rate = 0.1):
        super(BertModel, self).__init__()
        self.encoder = Encoder(num_layers=num_layers,d_model=d_model,
                        num_heads=num_heads,dff=dff,input_vocab_size=vocab_size,maximum_position_encoding=200,rate=dropout_rate)
        self.fc1 = tf.keras.layers.Dense(d_model, activation=gelu)
        self.layernorm = tf.keras.layers.LayerNormalization(-1)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

    def call(self,x,adjoin_matrix,mask,training=False):
        x = self.encoder(x,training=training,mask=mask,adjoin_matrix=adjoin_matrix)
        x = self.fc1(x)
        x = self.layernorm(x)
        x = self.fc2(x)
        return x


class PredictModel(tf.keras.Model):
    def __init__(self,num_layers = 6,d_model = 256,dff = 512,num_heads = 8,vocab_size =18, a=2, dropout_rate = 0.1,dense_dropout=0.1):
        super(PredictModel, self).__init__()
        self.encoder = Encoder(num_layers=num_layers,d_model=d_model,
                        num_heads=num_heads,dff=dff,input_vocab_size=vocab_size,maximum_position_encoding=200,rate=dropout_rate)
        self.fc1 = tf.keras.layers.Dense(256,activation=tf.keras.layers.LeakyReLU(0.1))
        self.dropout1 = tf.keras.layers.Dropout(dense_dropout)
        self.fc2 = tf.keras.layers.Dense(a)

    def call(self,x1,adjoin_matrix1,mask1,x2,adjoin_matrix2,mask2,t1,t2,t3,t4,training=False):
        x1 = self.encoder(x1,training=training,mask=mask1,adjoin_matrix=adjoin_matrix1)
        x1 = x1[:,0,:]
        x2 = self.encoder(x2,training=False,mask=mask2,adjoin_matrix=adjoin_matrix2)
        x2 = x2[:,0,:]
        x = tf.concat([x1,x2], axis=1)
        x = tf.concat([x, t1], axis=1)
        x = tf.concat([x, t2], axis=1)
        x = tf.concat([x, t3], axis=1)
        x = tf.concat([x, t4], axis=1)
        x = self.fc1(x)
        x = self.dropout1(x,training=training)
        x = self.fc2(x)
        return x
