import time
import threading
import datetime
import tensorflow as tf
import numpy as np
from ._reactor_net import define_train_graph
from ._reactor_net import SpeedControl
from ._reactor_loss import steer_loss
from ._reactor_loss import speed_loss
from ._reactor_loss import reactor_train
from ._reactor_dataset import ReactorDataSet


tf.flags.DEFINE_integer('log_step', 64, 'log step period')
tf.flags.DEFINE_string('stage', 'both', 'steer, speed or both')
tf.flags.DEFINE_boolean('use_features', True, 'use IMU readings in model')
tf.flags.DEFINE_boolean('share_param', False, 'share convolution kernels '
                                              'between steer and speed control')

FLAGS = tf.flags.FLAGS


def str_loss(loss):
  loss = tuple(loss)
  ret = []
  for l in loss:
    ret.append('%.4f' % l)
  ret = ', '.join(ret)
  return ret


class ReactorTrainThread(object):

  def __init__(self, tid, timestep, batch_size,
               dataset: ReactorDataSet,
               model_scope: tf.VariableScope):
    self._name = 'ReactorThread-%d' % tid
    self._tid = tid
    self._batch_size, self._timestep = batch_size, timestep
    self._scope = model_scope
    reuse = (tid != 0)
    with tf.name_scope(self._name):
      with tf.variable_scope(
          model_scope, reuse=reuse):
        self._create_submodel(dataset)
    self._sess, self._thread = None, None
    self._should_stop, self._started = False, False
    self._frame_length = timestep * batch_size
    self._dataset = dataset

  def _create_submodel(self, dataset):
    start = tf.placeholder('int32', shape=tuple())
    end = tf.placeholder('int32', shape=tuple())
    mode = tf.placeholder('int32', shape=tuple())
    self._start, self._end, self._mode = start, end, mode
    frames = dataset.frames[start:end]
    features = dataset.features[start:end]

    throttle = dataset.throttle[start:end]
    brake = dataset.brake[start:end]
    switch = dataset.switch[start:end]
    true_steer = dataset.true_steer[start:end]
    steer_labels = dataset.steer[start:end]
    speed_labels = SpeedControl(
      throttle=throttle, brake=brake, switch=switch)

    features = features if FLAGS.use_features else None
    steer_control, speed_control = define_train_graph(
      frames, self._timestep, true_steer,
      mode, features, FLAGS.share_param)
    self._steer_loss = steer_loss(steer_control, steer_labels)
    self._speed_loss = speed_loss(speed_control, speed_labels)

    if FLAGS.stage == 'speed':
      loss = (self._speed_loss,)
    elif FLAGS.stage == 'steer':
      loss = (self._steer_loss,)
    else:
      assert FLAGS.stage == 'both'
      loss = (self._speed_loss, self._steer_loss)

    train_var = reactor_train(loss)
    self._learning_rate = train_var.learning_rate
    self._global_step = train_var.global_step
    self._train_op = train_var.train_op
    self._fetches = [self._train_op, self._global_step, loss]
    if self._learning_rate is not None:
      self._fetches.append(self._learning_rate)
    else:
      self._fetches.append(tf.constant('N/A'))

  def _run(self):
    while not self._should_stop:
      total_samples = self._dataset.total_samples
      start = np.random.randint(
        0, total_samples - self._frame_length - 1)
      end = start + self._frame_length
      assert end <= total_samples

      time_start = time.time()
      mode = self._dataset.mode
      feed_dict = {self._start: start, self._end: end,
                   self._mode: mode}
      _, step, loss, lr = self._sess.run(
        self._fetches, feed_dict=feed_dict)

      if lr != b'N/A':
        lr = 'E^[%.4f]' % np.log10(lr)
      else:
        lr = 'N/A'

      time_end = time.time()
      if step % FLAGS.log_step == 0:
        timestamp = str(datetime.datetime.now().ctime())
        print('[%s] length=[%d] step=[%d] step-time=[%.4f] '
              'learning_rate=%s loss=[%s]'
              % (timestamp, self._timestep, step,
                 time_end - time_start, lr, str_loss(loss)))

  @property
  def thread(self):
      if self._thread is None:
        self._thread = threading.Thread(target=self._run)
      return self._thread

  def run(self, sess: tf.Session):
    assert self._thread is None, 'start twice'
    self._sess = sess
    self.thread.start()

  def join(self):
    self._should_stop = True
    self.thread.join()

  @property
  def learning_rate(self):
    return self._learning_rate

  @property
  def global_step(self):
    return self._global_step
