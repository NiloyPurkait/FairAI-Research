import tensorflow as tf
from tensorflow.keras import backend as K


@tf.function
def entropy_gaussian(self, mu, sigma, mean=True):
    msigma = tf.reshape(sigma, (K.shape(sigma)[0], -1))
    return tf.reduce_mean(0.5*msigma)



@tf.function
def negative_log_gaussian(self, data, mu, sigma, mean=True):
    mdata = tf.reshape(data, (K.shape(data)[0], -1))
    mmu = tf.reshape(mu, (K.shape(data)[0], -1))
    msigma = tf.reshape(sigma, (K.shape(data)[0], -1))
    return 0.5 * tf.reduce_mean((mdata-mmu)**2/(K.exp(msigma)+self.eps) + msigma)



@tf.function
def neg_log_bernoulli(self, true, pred, mean=True, clamp=True, rec=False):
    if clamp:
        pred = K.clip(pred, -9.5, 9.5)
        
    batch_size = K.shape(true)[0]
    dim = 1 if not rec else 110

    mdata = tf.reshape( true, (batch_size,dim) )
    mmu = tf.reshape( pred, (batch_size,dim) )

    log_prob_1 = tf.math.log_sigmoid(mmu)
    log_prob_2 = tf.math.log_sigmoid(-mmu)
    return -tf.reduce_mean((mdata*log_prob_1)+(1-mdata)*log_prob_2)



@tf.function
def KL(mu, log_sigma):
    kl_loss = 0.5 * tf.reduce_mean(( - log_sigma + K.square(mu) + K.exp(log_sigma)))
    return kl_loss


@tf.function
def md(t, l):
    s_0 = tf.where(t[:,0]==0)
    s_1 =tf.where(t[:,0]==1)

    z_0 = tf.gather(l, s_0)
    z_1 = tf.gather(l, s_1)

    z_0 = tf.reshape(z_0, (K.shape(z_0)[0], K.shape(z_0)[-1]))
    z_1 = tf.reshape(z_1, (K.shape(z_1)[0], K.shape(z_1)[-1]))
    return z_0, z_1



@tf.function
def kernel(a,b):
    dist1 = tf.expand_dims(tf.math.reduce_sum((a**2), axis=1), axis=1) * tf.ones(shape=(1,K.shape(b)[0]))
    dist2 = tf.expand_dims(tf.math.reduce_sum((b**2), axis=1), axis=0)* tf.ones(shape=(K.shape(a)[0], 1))
    dist3 = tf.matmul(a, tf.transpose(b, perm=[1, 0]))
    dist = (dist1 + dist2) - (2 * dist3)
    return tf.reduce_mean(tf.math.exp(-dist))



@tf.function
def mmd_loss( X, z):    
    z_s_0, z_s_1 = md(X, z)
    loss = kernel(z_s_0, z_s_0) + kernel(z_s_1, z_s_1) - 2 * kernel(z_s_0, z_s_1)
    return loss



def split_sensitive_X( tensor, col, n):
    '''takes Xn (2D feature tensor) and returns 2 tensors(sensitive features and normal features)'''
    dim = tensor.shape[-1]
    pre, sens, post =  tf.split(tensor, (col, n, (dim-(col+n))), axis=1)
    return sens, tf.concat([pre, post], axis=1)