""" loss related functions """
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


__all__ = ['huber_loss', 'l1_loss', 'l2_loss', 'tv_loss']

def huber_loss(x, delta=1., name='huber_loss'):
    """Huber loss: https://en.wikipedia.org/wiki/Huber_loss."""
    return tf.keras.losses.huber(y_true=tf.zeros_like(x), y_pred=x, delta=delta)  


def l1_loss(x, y, name='l1_loss'):
    """L1 loss: mean(abs(x-y))."""
    loss = tf.reduce_mean(tf.abs(x - y), name=name)
    return loss


def l2_loss(x, y, name='l2_loss'):
    """L2_loss: mean((x-y) ** 2)."""
    loss = tf.reduce_mean(tf.square(x - y), name=name)
    return loss


def tv_loss(x, name='tv_loss'):
    """tv_loss."""
    return tf.reduce_sum(tf.image.total_variation(x))