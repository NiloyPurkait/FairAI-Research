import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Dropout


# Sampling class for KLD 
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding of an observation."""

    def call(self, inputs):
        z_mu, z_log_sigma = inputs
        batch = tf.shape(z_mu)[0]
        dim = tf.shape(z_mu)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mu + tf.exp(0.5 * z_log_sigma) * epsilon


# 2 Layer Encoder
class Encoder(keras.layers.Layer):
    def __init__(self, input_dim, latent_dim, act):
        super(Encoder, self).__init__()
        self.dense_1 = Dense(input_dim*2, activation=act, input_shape=(input_dim,))
        self.dense_2 = Dense(input_dim/4, activation=act)
        self.z_mean =  Dense(latent_dim, name='mu')
        self.z_log_sigma = Dense(latent_dim, name='log_sigma')
        
    def call(self, x):        
        x = self.dense_1(x)
        x = self.dense_2(x)
        
        mean = self.z_mean(x)
        log_sigma = self.z_log_sigma(x)
        z = Sampling()([mean, log_sigma])
        
        return mean, log_sigma, z



# 2 Layer Decoder
class Decoder(keras.layers.Layer):
    def __init__(self, latent_dim, act):
        super(Decoder, self).__init__()
        self.dense_1 = Dense(input_dim/4, activation=act, input_shape=(latent_dim,))
        self.dense_2 = Dense(input_dim*2, activation=act)
        self.output_layer = Dense(input_dim)
        
    def call(self, x):
        x = self.dense_1(x)
        x = self.dense_2(x)
        return self.output_layer(x)



# 2 Layer MLP Classifier
class Classifier(keras.layers.Layer):
    def __init__(self, latent_dim, act, n_sens = 0):
        super(Classifier, self).__init__()
        self.dense_1 = Dense(64, activation=act, input_shape=(latent_dim+n_sens,))
        self.dense_2 = Dense(32, activation=act)
        self.dropout =  Dropout(0.2)
        self.output_layer = Dense(1)
        
    def call(self, x):
        x = self.dense_1(x)
        x = self.dropout(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return self.output_layer(x)