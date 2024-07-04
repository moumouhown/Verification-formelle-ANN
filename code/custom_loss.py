import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable()
def asymMSE(y_true, y_pred):
    lossFactor = 40.0
    d = y_true - y_pred
    maxes = K.argmax(y_true, axis=1)
    maxes_onehot = K.one_hot(maxes, K.int_shape(y_true)[-1])
    others_onehot = maxes_onehot - 1
    d_opt = d * maxes_onehot 
    d_sub = d * others_onehot
    a = lossFactor * (K.int_shape(y_true)[-1] - 1) * (K.square(d_opt) + K.abs(d_opt))
    b = K.square(d_opt)
    c = lossFactor * (K.square(d_sub) + K.abs(d_sub))
    d = K.square(d_sub)
    loss = tf.where(d_sub > 0, c, d) + tf.where(d_opt > 0, a, b)
    return K.mean(loss)
