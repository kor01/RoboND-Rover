import tensorflow as tf
from rover_agent.state_action import THROTTLE_SHAPE
from rover_agent.state_action import BRAKE_SHAPE
from rover_agent.state_action import STEER_SHAPE

from collections import namedtuple

tf.flags.DEFINE_integer('frame_units', 128, 'frame hidden unit size')
tf.flags.DEFINE_integer('feature_units', 64, 'feature hidden unit size')
FLAGS = tf.flags.FLAGS


ReactorOutput = namedtuple(
  'ReactorOutput', ('throttle', 'brake', 'steer', 'switch',
                    'steer_scope', 'speed_scope'))


def _softmax_layer(var_name, shape, inputs):
  unit_shape = inputs.shape.as_list()[-1]
  unit_shape = (unit_shape,)
  projection = tf.get_variable(
    var_name, unit_shape + shape)
  logits = tf.matmul(inputs, projection, name="Logits")
  prob = tf.nn.softmax(logits, name='Prob')
  return prob, logits


def consume_frame(inputs: tf.Tensor, timestep):
  assert inputs.dtype == tf.uint8

  # remove timestep dimension during convolution
  inputs = tf.image.rgb_to_grayscale(inputs)
  inputs = tf.image.resize_images(
    inputs, (80, 160), tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  # inputs are (batch_size, 80, 160, 1) images
  inputs = tf.to_float(inputs) / 255.0

  # convolution step
  with tf.variable_scope('steer') as scope:
    with tf.variable_scope('convolution'):

      conv1 = tf.layers.conv2d(
        inputs, 16, 8, 4, activation=tf.nn.relu, name='Conv1')
      conv2 = tf.layers.conv2d(
        conv1, 32, 3, 2, activation=tf.nn.relu, name='Conv2')
      conv3 = tf.layers.conv2d(
        conv2, 3, 1, activation=tf.nn.relu, name='Conv3')
      flatten = tf.contrib.layers.flatten(conv3)
      fc1 = tf.contrib.layers.fully_connected(
        flatten, FLAGS.frame_units)

    # recurrent step
    fc1 = tf.reshape(fc1, (-1, timestep, FLAGS.frame_units))
    fc1 = tf.unstack(fc1, axis=1)
    with tf.variable_scope('recurrent'):
      cell = tf.nn.rnn_cell.LSTMCell(FLAGS.frame_units)
      # shape (batch_size, timesteps, 256)
      outputs, _ = tf.nn.static_rnn(cell, fc1, dtype=tf.float32)
      outputs = tf.stack(outputs, axis=1)
      outputs = tf.reshape(outputs, (-1, FLAGS.frame_units))
  return outputs, scope


# local kinematics features (speed, yaw, roll)
def consume_features(features: tf.Tensor):
  with tf.variable_scope('steer'):
    with tf.variable_scope('features'):
      hidden1 = tf.layers.dense(
        features, 128, activation=tf.nn.relu,
        name='Hidden1')
      hidden2 = tf.layers.dense(
        hidden1, FLAGS.feature_units,
        activation=tf.nn.relu, name='Hidden2')
    return hidden2


def generate_action(
    frames: tf.Tensor, steer_scope: tf.VariableScope,
    features: tf.Tensor):

  inputs = tf.concat((frames, features), axis=-1)
  with tf.variable_scope('speed') as scope:
    # output for throttle brake switch
    with tf.variable_scope('switch'):
      switch = _softmax_layer(
        'SwitchProjection', (2,), inputs)

    with tf.variable_scope('throttle'):
      hidden = tf.layers.dense(
        inputs, 32, activation=tf.nn.relu, name='Hidden')
      throttle = _softmax_layer(
        'ThrottleProjection', THROTTLE_SHAPE, hidden)

    # output layer for brake
    with tf.variable_scope('brake'):
      hidden = tf.layers.dense(
        inputs, 32, activation=tf.nn.relu, name='Hidden')
      brake = _softmax_layer(
        'BrakeProjection', BRAKE_SHAPE, hidden)

  # output layer for steer
  with tf.variable_scope('steer'):
    hidden = tf.layers.dense(
      inputs, 64, activation=tf.nn.relu, name='Hidden')
    steer = _softmax_layer(
      'SteerProjection', STEER_SHAPE, hidden)

  return ReactorOutput(
    throttle=throttle, brake=brake,
    steer=steer, switch=switch,
    steer_scope=steer_scope, speed_scope=scope)
