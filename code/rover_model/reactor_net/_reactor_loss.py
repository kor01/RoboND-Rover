import tensorflow as tf
from collections import namedtuple
from ._reactor_net import SpeedControl


tf.flags.DEFINE_float('learning_rate', 0.001,
                      'initial learning rate')


tf.flags.DEFINE_boolean('use_adam', True, 'use_adam optimizer')

FLAGS = tf.flags.FLAGS


def xent_loss(labels, logits, weights=None):
  xent = tf.nn.sparse_softmax_cross_entropy_with_logits
  loss = xent(labels=labels, logits=logits)
  if weights is not None:
    loss *= weights
  return tf.reduce_mean(loss)


def steer_loss(steer_control, steer_label):
  ret = xent_loss(labels=steer_label,
                  logits=steer_control[1])
  return ret


def speed_loss(speed_control: SpeedControl,
               label: SpeedControl):
  _, throttle = speed_control.throttle

  weights = tf.to_float(label.switch)

  throttle_loss = xent_loss(
    labels=label.throttle, logits=throttle,
    weights=1 - weights)

  _, brake = speed_control.brake
  brake_loss = xent_loss(
    labels=label.brake, logits=brake,
    weights=weights)

  _, switch = speed_control.switch
  
  switch_loss = xent_loss(
    labels=label.switch, logits=switch)

  ret = throttle_loss + brake_loss + switch_loss
  return ret

ReactorTrain = namedtuple(
  'ReactorTrain', ('learning_rate',
                   'global_step', 'train_op', 'loss'))


def reactor_train(losses):

  assert len(losses) > 0, 'at least one loss'
  loss = sum(losses)

  with tf.variable_scope('train'):

    learning_rate = tf.get_variable(
      'LearningRate', initializer=FLAGS.learning_rate,
      dtype=tf.float32, trainable=False)

    global_step = tf.get_variable(
      'GlobalStep', initializer=0,
      dtype=tf.int32, trainable=False)

    if FLAGS.use_adam:
      optimizer = tf.train.AdamOptimizer(
        FLAGS.learning_rate)
      learning_rate = None
    else:
      optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate)

    train_op = optimizer.minimize(loss, global_step)

    # for stage training
    return ReactorTrain(learning_rate=learning_rate,
                        global_step=global_step,
                        train_op=train_op, loss=loss)
