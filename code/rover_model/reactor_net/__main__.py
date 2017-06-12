import os
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
from ._reactor_main import process_main
from ._reactor_main import train_main


tf.flags.DEFINE_string('task', 'train', 'process or train')

FLAGS = tf.flags.FLAGS


def main(_):
  task = FLAGS.task
  if task == 'train':
    train_main()
  else:
    assert task == 'process', 'only train and process'
    process_main()


if __name__ == '__main__':
  tf.app.run()
