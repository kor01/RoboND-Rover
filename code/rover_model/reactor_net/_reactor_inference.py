import os
import tensorflow as tf
from collections import namedtuple
from ._utils import define_local_variables
from rover_spec import FRAME_SHAPE
from ._reactor_dataset import NUM_FEATURES
from ._reactor_net import steer_consume_frame
from ._reactor_net import steer_transform_features


ReactorConfig = namedtuple(
  'ReactorConfig', ('feature_units', 'frame_units',
                    'path', 'scope_name'))


class ReactorInference(object):

  def __init__(self, config: ReactorConfig):

    # to avoid GPU memory alloc
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    self._config = config
    self._graph = tf.Graph()
    with self._graph.as_default():
      with tf.name_scope('ReactorInference'):
        # use cpu for inference
        with tf.variable_scope(config.scope_name, reuse=None),\
             tf.device('/cpu:0'):
          self._define_graph()
      self._sess = tf.Session(graph=self._graph)
      self._sess.run(tf.local_variables_initializer())
      saver = tf.train.Saver()
      saver.restore(self._sess, config.path)

  def _define_states(self):
    # recurrent state
    cell = tf.nn.rnn_cell.LSTMCell(
      self._config.frame_units)
    zero_state = cell.zero_state(
      batch_size=1, dtype=tf.float32)
    self._c = define_local_variables(
      zero_state.c.shape, dtype=tf.float32)
    self._h = define_local_variables(
      zero_state.h.shape, dtype=tf.float32)
    self._state = tf.nn.rnn_cell.LSTMStateTuple(
      self._c, self._h)

  def _define_graph(self):
    self._define_states()
    self._frame = tf.placeholder(
      shape=FRAME_SHAPE + (3,), dtype=tf.uint8)
    self._feature = tf.placeholder(
      shape=(NUM_FEATURES,), dtype=tf.float32)

    # to interface the train graph definition functions
    feature = tf.expand_dims(self._feature, 0)
    frame = tf.expand_dims(self._frame, 0)

    output, state = steer_consume_frame(
      frame, 1, last_state=self._state)
    updates = [self._c.assign(state.c), self._h.assign(state.h)]
    with tf.control_dependencies(updates):
      features = steer_transform_features(feature)
      self._actions = generate_action(output, features)
      self._steer = tf.reshape(self._actions.steer[0], (-1,))

  def consume(self, frame, feature):
    """
    predict steers and update internal state
    :param frame: the current frame
    :param feature: cos(pitch), sin(pitch), cos(roll), sin(roll), speed
    :return: steer angle distribution
    """
    feed_dict = {self._frame: frame, self._feature: feature}
    ret = self._sess.run(self._steer, feed_dict)
    return ret
