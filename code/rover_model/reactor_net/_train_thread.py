import time
import threading
import datetime
import tensorflow as tf
import numpy as np
from ._reactor_net import consume_frame
from ._reactor_net import consume_features
from ._reactor_net import generate_action
from ._reactor_net import ReactorOutput
from ._reactor_loss import reactor_loss
from ._reactor_loss import reactor_train
from ._reactor_dataset import ReactorDataSet


tf.flags.DEFINE_integer('log_step', 64, 'log step period')
tf.flags.DEFINE_string('stage', 'steer', 'steer, speed or total')

FLAGS = tf.flags.FLAGS


class ReactorTrainThread(object):

  def __init__(self, tid, timestep, batch_size,
               dataset: ReactorDataSet,
               model_scope: tf.VariableScope):
    self._name = 'ReactorThread-%d' % tid
    self._tid = tid
    self._batch_size, self._timestep = batch_size, timestep
    reuse = (tid != 0)
    with tf.name_scope(self._name):
      with tf.variable_scope(
          model_scope, reuse=reuse):
        self._create_submodel(dataset)
    self._sess, self._thread = None, None
    self._should_stop, self._started = False, False
    self._frame_length = timestep * batch_size
    self._total_samples = dataset.total_samples

  def _create_submodel(self, dataset):
    start = tf.placeholder('int32', shape=tuple())
    end = tf.placeholder('int32', shape=tuple())
    self._start, self._end = start, end
    frames = dataset.frames[start:end]
    features = dataset.features[start:end]

    throttle = dataset.throttle[start:end]
    steer = dataset.steer[start:end]
    brake = dataset.brake[start:end]
    switch = dataset.switch[start:end]
    labels = ReactorOutput(
      throttle=throttle, steer=steer,
      brake=brake, switch=switch,
      steer_scope=None, speed_scope=None)

    frames, steer_scope = \
      consume_frame(frames, self._timestep)
    features = consume_features(features)
    actions = generate_action(
      frames, steer_scope, features)
    self._loss = reactor_loss(actions, labels)

    if FLAGS.stage == 'total':
      loss, target_scope = self._loss.total, None
    elif FLAGS.stage == 'steer':
      loss, target_scope = self._loss.steer,  actions.steer_scope
    else:
      assert FLAGS.stage == 'speed', \
        'unknown stage %s' % FLAGS.stage
      loss, target_scope = self._loss.speed, actions.speed_scope

    train_var = reactor_train(loss, target_scope)

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
      start = np.random.randint(
        0, self._total_samples - self._frame_length - 1)
      end = start + self._frame_length
      assert end <= self._total_samples

      time_start = time.time()

      feed_dict = {self._start: start, self._end: end}
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
              'learning_rate=%s loss=[%.4f]'
              % (timestamp, self._timestep, step,
                 time_end - time_start, lr, loss))

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
