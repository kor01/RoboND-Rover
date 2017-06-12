import tensorflow as tf
from collections import namedtuple
from ._reactor_net import ReactorOutput

tf.flags.DEFINE_float('learning_rate', 0.001,
                      'initial learning rate')


tf.flags.DEFINE_boolean('use_adam', True, 'use_adam optimizer')

FLAGS = tf.flags.FLAGS

ReactorLoss = namedtuple(
  'ReactorLoss', ('total', 'steer', 'speed'))


def xent_loss(labels, logits, weights=None):
  xent = tf.nn.sparse_softmax_cross_entropy_with_logits
  loss = xent(labels=labels, logits=logits)
  if weights is not None:
    loss *= weights
  return tf.reduce_mean(loss)


def reactor_loss(
    output: ReactorOutput, label: ReactorOutput):

  _, throttle = output.throttle

  weights = tf.to_float(label.switch)

  throttle_loss = xent_loss(
    labels=label.throttle, logits=throttle,
    weights=1 - weights)

  _, brake = output.brake
  brake_loss = xent_loss(
    labels=label.brake, logits=brake,
    weights=weights)

  _, steer = output.steer
  steer_loss = xent_loss(labels=label.steer, logits=steer)

  _, switch = output.switch
  switch_loss = xent_loss(labels=label.switch, logits=steer)

  # switch = 0 (throttle mode); switch = 1 (brake mode)

  speed_loss = throttle_loss + brake_loss + switch_loss
  loss = speed_loss + steer_loss

  return ReactorLoss(total=loss, steer=steer_loss,
                     speed=speed_loss)


ReactorTrain = namedtuple(
  'ReactorTrain', ('learning_rate',
                   'global_step', 'train_op'))


def reactor_train(loss, target_scope=None):

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

    if target_scope is not None:
      var_list = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES,
        target_scope.name)
      assert len(var_list) > 0, 'empty var_list'
    else:
      var_list = None

    # for stage training
    train_op = optimizer.minimize(
      loss, global_step, var_list=var_list)
    return ReactorTrain(learning_rate=learning_rate,
                        global_step=global_step,
                        train_op=train_op)
