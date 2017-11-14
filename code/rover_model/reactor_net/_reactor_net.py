import tensorflow as tf
from rover.state_action import THROTTLE_SHAPE
from rover.state_action import BRAKE_SHAPE
from rover.state_action import STEER_SHAPE

from collections import namedtuple

SpeedControl = namedtuple(
  'SpeedControl', ('throttle', 'brake', 'switch'))


def _softmax_layer(var_name, shape, inputs):
  unit_shape = inputs.shape.as_list()[-1]
  unit_shape = (unit_shape,)
  projection = tf.get_variable(
    var_name, unit_shape + shape)
  logits = tf.matmul(inputs, projection, name="Logits")
  prob = tf.nn.softmax(logits, name='Prob')
  return prob, logits


def process_frame(inputs: tf.Tensor):
  assert inputs.dtype == tf.uint8

  # remove timestep dimension during convolution
  inputs = tf.image.rgb_to_grayscale(inputs)
  inputs = tf.image.resize_images(
    inputs, (80, 160), tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  # inputs are (batch_size, 80, 160, 1) images
  inputs = tf.to_float(inputs) / 255.0
  return inputs


def define_convolution(inputs):
  with tf.variable_scope('convolution'):
    conv1 = tf.layers.conv2d(
      inputs, 16, 8, 4, activation=tf.nn.relu, name='Conv1')
    conv2 = tf.layers.conv2d(
      conv1, 32, 3, 2, activation=tf.nn.relu, name='Conv2')
    conv3 = tf.layers.conv2d(
      conv2, 3, 1, activation=tf.nn.relu, name='Conv3')
    flatten = tf.contrib.layers.flatten(conv3)
    fc1 = tf.contrib.layers.fully_connected(
      flatten, 128)
    return fc1


def steer_consume_frame(inputs: tf.Tensor,
                        timestep, last_state=None):
  # convolution step
  with tf.variable_scope('frame'):
    fc1 = define_convolution(inputs)

  # recurrent step
  with tf.variable_scope('recurrent'):
    fc1 = tf.reshape(fc1, (-1, timestep, 128))
    fc1 = tf.unstack(fc1, axis=1)
    cell = tf.nn.rnn_cell.LSTMCell(128)
    # shape (batch_size, timesteps, 256)
    outputs, state = tf.nn.static_rnn(
      cell, fc1, initial_state=last_state, dtype=tf.float32)
    outputs = tf.stack(outputs, axis=1)
    outputs = tf.reshape(outputs, (-1, 128))
    return outputs, state


def speed_transform_frame(inputs: tf.Tensor):
  with tf.variable_scope('frame'):
    return define_convolution(inputs)


def define_mlp(inputs: tf.Tensor):
  with tf.variable_scope('mlp'):
    hidden1 = tf.layers.dense(
      inputs, 128, activation=tf.nn.relu, name='Hidden1')
    hidden2 = tf.layers.dense(
      hidden1, 128, activation=tf.nn.relu, name='Hidden2')
    return hidden2


def define_embedding_layer(
    name, dim, vocab_size, inputs: tf.Tensor):
  with tf.variable_scope(name):
    embedding = tf.get_variable(
      'Embedding', shape=(vocab_size, dim))
    ret = tf.gather(embedding, inputs)
  return ret


# local kinematics features (speed, yaw, roll)
def steer_transform_features(features=None):
  if features is None:
    return None
  with tf.variable_scope('steer'):
    with tf.variable_scope('feature'):
      return define_mlp(features)


def speed_transform_features(
    true_steer: tf.Tensor, features=None):
  with tf.variable_scope('speed'):
    with tf.variable_scope('feature'):
      if features is not None:
        features = tf.concat((features, true_steer), axis=-1)
      else:
        features = true_steer
      return define_mlp(features)


def generate_steer_control(
    frames: tf.Tensor, mode: tf.Tensor, features=None):

  with tf.variable_scope('steer'):
    with tf.variable_scope('control'):
      if features is not None:
        inputs = tf.concat((frames, features), axis=-1)
      else:
        inputs = frames
      emb_size = inputs.shape.as_list()[-1]
      mode = define_embedding_layer('mode', emb_size, 3, mode)
      inputs += mode
      hidden = tf.layers.dense(
        inputs, 64, activation=tf.nn.relu, name='Hidden')
      steer = _softmax_layer(
        'SteerProjection', STEER_SHAPE, hidden)
    return steer


def generate_speed_control(
    frames: tf.Tensor, features: tf.Tensor, mode: tf.Tensor):

  with tf.variable_scope('speed'):
    with tf.variable_scope('control'):
      inputs = tf.concat((frames, features), axis=-1)
      emb_size = inputs.shape.as_list()[-1]
      mode = define_embedding_layer('mode', emb_size, 3, mode)

      inputs += mode
      hidden = tf.layers.dense(
        inputs, 128, activation=tf.nn.relu, name='Hidden')

      with tf.variable_scope('switch'):
        hidden1 = tf.layers.dense(
          hidden, 32, activation=tf.nn.relu, name='Hidden1')
        switch = _softmax_layer(
          'SwitchProjection', (2,), hidden1)
      with tf.variable_scope('throttle'):
        hidden1 = tf.layers.dense(
          hidden, 32, activation=tf.nn.relu, name='Hidden1')
        throttle = _softmax_layer(
          'ThrottleProjection', THROTTLE_SHAPE, hidden1)

      with tf.variable_scope('brake'):
        hidden1 = tf.layers.dense(
          hidden, 32, activation=tf.nn.relu, name='Hidden1')
        brake = _softmax_layer(
          'BrakeProjection', BRAKE_SHAPE, hidden1)
  return SpeedControl(
    switch=switch, throttle=throttle, brake=brake)


def define_train_graph(
    frames, timestep, true_steer,
    mode, features=None, share_param=False):

  frames = process_frame(frames)
  assert isinstance(frames, tf.Tensor)

  scope = 'motion' if share_param else 'steer'
  with tf.variable_scope(scope):
    steer_frames, _ = steer_consume_frame(frames, timestep)
  steer_features = steer_transform_features(features)
  steer_control = generate_steer_control(
    steer_frames, mode, steer_features)

  scope = 'motion' if share_param else 'speed'
  reuse = None if scope == 'speed' else True
  with tf.variable_scope(scope, reuse=reuse):
    speed_frames = speed_transform_frame(frames)
  speed_features = speed_transform_features(
    true_steer, features)
  speed_control = generate_speed_control(
    speed_frames, speed_features, mode)

  return steer_control, speed_control


def define_steer_inference_graph(
    state_cache, frame, mode,
    features=None, share_param=False):

  if features:
    features = tf.expand_dims(features, 0)

  frame = process_frame(frame)
  frame = tf.expand_dims(frame, 0)

  scope = 'control' if share_param else 'steer'
  with tf.variable_scope(scope):
    frame, state = steer_consume_frame(
      frame, 1, last_state=state_cache)

  updates = [state_cache.c.assign(state.c),
             state_cache.h.assign(state.h)]

  with tf.control_dependencies(updates):
    features = steer_transform_features(features)
    steer_control = generate_steer_control(
      frame, mode, features)

  return steer_control


def define_speed_inference_graph(
    frame, true_steer, mode,
    features=None, share_param=False):
  if features:
    features = tf.expand_dims(features, 0)
  frame = process_frame(frame)
  assert isinstance(frame, tf.Tensor)
  scope = 'control' if share_param else 'speed'
  with tf.variable_scope(scope):
    frame = speed_transform_frame(frame)
  features = speed_transform_features(true_steer, features)
  speed_control = generate_speed_control(frame, features, mode)
  return speed_control
