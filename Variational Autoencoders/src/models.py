import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Mean
from tensorflow.keras import backend as K
from losses import *




#### Variational Fair Autoencoder
class VFAE(keras.Model):
    def __init__(self,
                 encoder,
                 encoder_z,
                 reconstructor_z,
                 decoder,
                 classifier,
                 feature_dim,
                 loss_type,
                 **kwargs):
        
        super(VFAE, self).__init__(**kwargs)
        
        self.eps = tf.constant([10e-25])
        self.beta=1.
        
        self.encoder = encoder
        self.encoder_z = encoder_z
        self.reconstructor_z = reconstructor_z
        self.decoder = decoder
        self.classifier = classifier
        
        self.loss_type = loss_type
        self.total_loss_tracker = Mean(name="total_loss")
        self.prediction_loss_tracker = Mean(name="pred_loss")
        self.kl_loss_tracker = Mean(name="kl_loss")
        self.mmd_loss_tracker = Mean(name="mmd_loss")
        self.reconst_loss_tracker = Mean(name="reconst_loss")
        self.reconst_z_loss_tracker = Mean(name="reconst_z_loss")


    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.prediction_loss_tracker,
            self.kl_loss_tracker,
            self.mmd_loss_tracker,
            self.reconst_loss_tracker,
            self.reconst_z_loss_tracker
        ]
    
    
    def call(self, inputs):
        X, y = inputs
        y = tf.reshape(y, (-1,1))
        
        sens, _ = split_sensitive_X(X, 0, 1)
        
        z_mean, z_log_sigma, z = self.encoder(X)
        q_z_1_mean, q_z_1_log_sigma, z_1 = self.encoder_z(tf.concat([z,y], axis=1))
        
        reconst = self.decoder(tf.concat([z, sens], axis=1))
        z_reconst_mean, z_reconst_log_sigma, _ = self.reconstructor_z(tf.concat([z_1, y], axis=1))

        preds = self.classifier(z)

        return z_mean, z_log_sigma, z,                      \
                reconst, q_z_1_mean, q_z_1_log_sigma, z_1,   \
                z_reconst_mean, z_reconst_log_sigma, preds
    

        
    def train_step(self, data):
        X, y = data
        with tf.GradientTape() as tape:
            
            z_mean, z_log_sigma, z,                                       \
            reconst, q_z_1_mean, q_z_1_log_sigma, z_1,                    \
            z_reconst_mean, z_reconst_log_sigma, preds = self.call(data)
    
            reconst_loss = neg_log_bernoulli(X, reconst, rec=1)
            reconst_z_loss = negative_log_gaussian(z, z_reconst_mean, z_reconst_log_sigma)
            classifier_loss = neg_log_bernoulli(y, preds)
            kl_loss = KL(q_z_1_mean, q_z_1_log_sigma)
            mmd_loss = mmd_loss(X, z)
            entropy_z = entropy_gaussian(z_mean, z_log_sigma)
            
            total_loss = reconst_loss + kl_loss + reconst_z_loss - entropy_z + self.beta*classifier_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.prediction_loss_tracker.update_state(classifier_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.mmd_loss_tracker.update_state(mmd_loss)
        self.reconst_loss_tracker.update_state(reconst_loss)
        self.reconst_z_loss_tracker.update_state(reconst_z_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "classification_loss": self.prediction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "mmd_loss": self.mmd_loss_tracker.result(),
            "reconst_loss": self.reconst_loss_tracker.result(),
            "reconst_z_loss": self.reconst_z_loss_tracker.result()
        }




####Variational Fair Information Bottleneck
class VFIB(keras.Model):
    def __init__(self, encoder, predictor, feature_dim,loss_type,  **kwargs):
        super(VFIB, self).__init__(**kwargs)
        self.encoder = encoder
        self.classifier = predictor
        self.loss_type = loss_type
        self.total_loss_tracker = Mean(name="total_loss")
        self.prediction_loss_tracker = Mean(name="prediction_loss")
        self.kl_loss_tracker = Mean(name="kl_loss")
        self.mmd_loss_tracker = Mean(name="mmd_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.prediction_loss_tracker,
            self.kl_loss_tracker,
            self.mmd_loss_tracker
        ]
    
    def call(self, inputs):
        # 0 refers to first column with sensitive feature 'Age'
        sens, _ = split_sensitive_X(inputs, 0, 1)
        mu, sig, z = self.encoder(inputs)
        preds = self.classifier(tf.concat([z, sens], 1))
        return mu, sig, z, preds
        
        
    def train_step(self, data):
        X, y = data
        with tf.GradientTape() as tape:
            
            z_mean, z_log_sigma, z, preds = self.call(X)

            prediction_loss = neg_log_bernoulli(y, preds)
            kl_loss = KL(z_mean, z_log_sigma)
            mmd_loss = mmd_loss(X, z)
            
            if self.loss_type=='all':
                total_loss =  prediction_loss+ kl_loss + mmd_loss
            elif self.loss_type=='kl':
                total_loss =  prediction_loss+ kl_loss
            else:
                total_loss =  prediction_loss
                
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.prediction_loss_tracker.update_state(prediction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.mmd_loss_tracker.update_state(mmd_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "classification_loss": self.prediction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "mmd_loss": self.mmd_loss_tracker.result()
        }
