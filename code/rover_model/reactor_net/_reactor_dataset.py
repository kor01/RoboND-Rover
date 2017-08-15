import os
import tensorflow as tf
import threading
import datetime
from rover_spec import FRAME_SHAPE
from ._utils import define_local_variables

NUM_FEATURES = 5

tf.flags.DEFINE_integer('train_buffer_size', 1024 * 16,
                        'max number of training '
                        'frames in one replay footage')
FLAGS = tf.flags.FLAGS

MODES = {'left': 0, 'right': 1, 'slow': 2}


def assign_variables(value, variable, sess):
  sess.run(variable.assign(value))
  return tuple((None, variable))


def create_queue(dirs, name):
  files = [os.path.join(d, name) for d in dirs]
  return tf.train.string_input_producer(
    files, shuffle=False)


def read_modes(dirs):
  files = [os.path.join(d, 'mode.txt') for d in dirs]
  modes = [open(f).read().strip() for f in files]
  modes = tuple([MODES[x] for x in modes])
  return modes


def read_whole_tensor(reader, queue, dtype, shape):
  filename, tensor = reader.read(queue)
  tensor = tf.decode_raw(tensor, dtype)
  tensor = tf.reshape(tensor, shape)
  return filename, tensor


class ReplayFootage(object):
  def __init__(self, path):
    with tf.device('/cpu:0'):
      dirs = []
      with open(path) as ip:
        for line in ip:
          if line.startswith('#'):
            continue
          dirs.append(line[:-1])
      self.dirs = dirs
      frames_queue = create_queue(dirs, 'frames.uint8.bin')
      features_queue = create_queue(dirs, 'features.float32.bin')
      labels_queue = create_queue(dirs, 'labels.int32.bin')
      true_steer_queue = create_queue(dirs, 'true_steer.float32.bin')

      reader = tf.WholeFileReader()
      self.frames = read_whole_tensor(
        reader, frames_queue, 'uint8', (-1,) + FRAME_SHAPE + (3,))

      self.features = read_whole_tensor(
        reader, features_queue, 'float32', (-1, NUM_FEATURES))

      self.labels = read_whole_tensor(
        reader, labels_queue, 'int32', (4, -1))

      # steer sine and cosine as conditional input for speed control network
      self.true_steer = read_whole_tensor(
        reader, true_steer_queue, 'float32', (-1, 2))
      self.modes = read_modes(dirs)


def update_variable_slice(
    variable, update, update_size, batch_major=True):
  update_index = tf.range(0, update_size)
  if batch_major:
    return tf.scatter_update(
      variable, update_index,
      update[:update_size], use_locking=True)
  else:
    update = tf.transpose(update[:, :update_size])
    return tf.scatter_update(
      variable, update_index, update, use_locking=True)


class ReactorDataSet(object):

  # noinspection PyUnresolvedReferences
  def __init__(self, path):

    self._footage = ReplayFootage(path)
    self._update_lock = threading.Lock()
    with tf.device('/cpu:0'):
      buf_size = FLAGS.train_buffer_size

      frame_shape = (buf_size,) + FRAME_SHAPE + (3,)
      self._frames = define_local_variables(frame_shape, 'uint8')

      feature_shape = (buf_size,) + (NUM_FEATURES,)
      self._features = define_local_variables(
        feature_shape, 'float32')

      label_shape = (buf_size,) + (4,)
      self._labels = define_local_variables(
        label_shape, 'int32')

      steer_shape = (buf_size,) + (2,)
      self._true_steer = define_local_variables(
        steer_shape, 'float32')

      self._throttle, self._brake = \
        self._labels[:, 0], self._labels[:, 1]
      self._steer, self._switch = \
        self._labels[:, 2], self._labels[:, 3]

      size = tf.shape(self._footage.frames[1])[0]
      self._size_op = tf.minimum(
        FLAGS.train_buffer_size, size)

      assign_ops = []
      assign_ops.append(update_variable_slice(
        self._frames, self._footage.frames[1], self._size_op))
      assign_ops.append(update_variable_slice(
        self._features, self._footage.features[1], self._size_op))
      assign_ops.append(update_variable_slice(
        self._labels, self._footage.labels[1],
        self._size_op, batch_major=False))
      assign_ops.append(update_variable_slice(
        self._true_steer, self._footage.true_steer[1], self._size_op))
      footage_step = define_local_variables(tuple(), dtype='int32')
      with tf.control_dependencies(assign_ops):
        self._read_op = footage_step.assign_add(1)
      self._total_frames = 0
      self._mode = -1

  @property
  def frames(self):
    return self._frames

  @property
  def features(self):
    return self._features

  @property
  def throttle(self):
      return self._throttle

  @property
  def brake(self):
      return self._brake

  @property
  def steer(self):
      return self._steer

  @property
  def switch(self):
      return self._switch

  @property
  def total_samples(self):
    with self._update_lock:
      return self._total_frames

  @property
  def true_steer(self):
    return self._true_steer

  @property
  def num_footage(self):
    return len(self._footage.dirs)

  @property
  def mode(self):
    with self._update_lock:
      return self._mode

  def read_next(self, sess: tf.Session):
    with self._update_lock:
      fetches = [self._read_op, self._size_op,
                 self._footage.frames[0],
                 self._footage.features[0],
                 self._footage.labels[0],
                 self._footage.true_steer[0]]
      step, size, frame_file, feature_file, label_file, _ \
        = sess.run(fetches)
      step = (step - 1) % len(self._footage.modes)
      mode = self._footage.modes[step]
      timestamp = str(datetime.datetime.now().ctime())
      print('[%s] refresh [%d] [%d] [%s] [%s] [%s]'
            % (timestamp, size, mode, frame_file,
               feature_file, label_file))
      self._total_frames = size
      self._mode = mode
