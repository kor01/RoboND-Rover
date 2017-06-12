import os
import time
import threading
import tensorflow as tf

tf.flags.DEFINE_integer('lr_decay_step', 8192,
                        'learning rate decay step')

tf.flags.DEFINE_string('ckpt_dir', None, 'checkpoint path')
tf.flags.DEFINE_integer('ckpt_step', 1024, 'check point steps')
tf.flags.DEFINE_integer('footage_cycle',
                        8192, 'cycles to read next footage')


FLAGS = tf.flags.FLAGS


class ReactorTrainMonitor(object):

  def __init__(self, sess, global_step, learning_rate, dataset):
    self._sess = sess
    self._global_step = global_step
    self._learning_rate = learning_rate
    self._thread = None
    self._should_stop = False
    self._ckpt_steps = 0
    self._lr_steps = 0
    self._footage_steps = 0
    self._thread = None
    self._dataset = dataset
    if self._learning_rate is not None:
      self._assign = self._learning_rate.assign(
        self._learning_rate / 2)
    else:
      self._assign = None
    self._saver = tf.train.Saver()

  def join(self):
    self._should_stop = True
    self.thread.join()

  @property
  def stopped(self):
    return self._should_stop

  @property
  def thread(self):
    if self._thread is None:
      self._thread = threading.Thread(target=self._run)
    return self._thread

  def _run(self):
    while not self._should_stop:
      step = self._sess.run(self._global_step)
      ckpt_steps = step / FLAGS.ckpt_step
      if ckpt_steps > self._ckpt_steps:
        path = os.path.join(FLAGS.ckpt_dir, 'reactor_model')
        self._saver.save(
          self._sess, path, self._global_step)
        self._ckpt_steps += 1

      lr_steps = step / FLAGS.lr_decay_step
      if lr_steps > self._lr_steps \
          and self._assign is not None:
        self._sess.run(self._assign)
        self._lr_steps += 1

      footage_steps = step / FLAGS.footage_cycle
      if footage_steps > self._footage_steps \
          and self._dataset.num_footage > 1:
        self._dataset.read_next(self._sess)
      time.sleep(1)

  def run(self):
    assert self._thread is None, 'start twice'
    self.thread.start()
