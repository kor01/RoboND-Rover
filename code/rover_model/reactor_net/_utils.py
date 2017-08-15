import tensorflow as tf


def define_local_variables(shape, dtype):
  ret = tf.Variable(tf.zeros(shape, dtype=dtype),
                    trainable=False,
                    collections=[tf.GraphKeys.LOCAL_VARIABLES])
  return ret
