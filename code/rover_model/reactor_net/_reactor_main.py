import atexit
import time
import os
import datetime
import tensorflow as tf
from ._train_thread import ReactorTrainThread
from ._reactor_dataset import ReactorDataSet
from ._train_monitor import ReactorTrainMonitor
from ._reactor_process import process_records


tf.flags.DEFINE_string('scope_name', 'reactor_model',
                       'name of the outer most variable scope')


tf.flags.DEFINE_string('timesteps', "[4, 8, 16, 32]",
                        'timesteps for each training thread')

tf.flags.DEFINE_integer('batch_size', 8, 'batch size of training')

tf.flags.DEFINE_string('data_dir', None, 'dir to output binary footage data')
tf.flags.DEFINE_string('record_dir', None, 'play record directory')
tf.flags.DEFINE_string('footage', None, 'footage list path')
tf.flags.DEFINE_boolean('cpu_only', False, 'use cpu for training (debug)')

FLAGS = tf.flags.FLAGS


def process_main():
  process_records(FLAGS.record_dir, FLAGS.data_dir)


def train_main():
  if FLAGS.cpu_only:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

  scope = tf.VariableScope(
    name=FLAGS.scope_name, reuse=False)
  dataset = ReactorDataSet(FLAGS.footage)

  timesteps = eval(FLAGS.timesteps)
  threads = tuple([ReactorTrainThread(
    i, t, FLAGS.batch_size, dataset, scope)
    for i, t in enumerate(timesteps)])

  assert len(threads) > 0, 'threads size > 0'
  gpu_options = tf.GPUOptions(
    per_process_gpu_memory_fraction=0.333)
  sess = tf.Session(
    config=tf.ConfigProto(gpu_options=gpu_options))
  learning_rate = threads[0].learning_rate
  global_step = threads[0].global_step
  monitor = ReactorTrainMonitor(
    sess, global_step, learning_rate, dataset)

  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())
  coord = tf.train.Coordinator()
  tf.train.start_queue_runners(coord=coord, sess=sess)
  dataset.read_next(sess)

  for t in threads:
    t.run(sess)
  monitor.run()

  def clean_up():
    if not monitor.stopped:
      timestamp = str(datetime.datetime.now().ctime())
      print('[%s] running clean up' % timestamp)
      monitor.join()
      coord.request_stop()
      coord.join()
      for t in threads:
        t.join()

  atexit.register(clean_up)

  while True:
    try:
      time.sleep(30)
    except KeyboardInterrupt:
      clean_up()
      break
