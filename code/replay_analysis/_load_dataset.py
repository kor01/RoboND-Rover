import os
import numpy as np
import matplotlib.image as mpimg
from rover_spec import STR_SCHEMA
from rover_spec import STR_SCHEMA_ROW
from rover_spec import METRIC_DTYPE


class Experience(object):

  def __init__(self, frames: np.ndarray,
               metrics: np.ndarray, timeline: np.ndarray):

    assert len(frames) == len(metrics) == len(timeline), \
      (len(frames), len(metrics), len(timeline), frames.shape)
    assert metrics.dtype == METRIC_DTYPE
    frames.flags.writeable = False
    timeline.flags.writeable = False
    metrics.flags.writeable = False

    self._timeline = timeline
    self._frames = frames
    self._metrics = metrics

  def __len__(self):
    return len(self._timeline)

  def at(self, i):
    return self._frames[i], self._metrics[i], self._timeline[i]

  @property
  def timeline(self):
    return self._timeline

  @property
  def frames(self):
    return self._frames

  @property
  def metrics(self):
    return self._metrics

  def __iter__(self):
    return zip(self._frames, self._metrics, self._timeline)


def load_replay_from_csv(path, experiment_dir=None):
  """
  load experience recorded from train mode
  :param path: path to csv file from train mode
  :param experiment_dir: path to experiment directory
  :return: Experience
  """

  with open(path) as ip:
    lines = list(ip)
  lines = filter(lambda x: STR_SCHEMA not in x, lines)
  rows = [x.split(';') for x in lines]
  images = [x[0] for x in rows]
  csv_metrics = [[float(x) for x in row[1:]] for row in rows]
  metrics = np.zeros((len(csv_metrics)), dtype=METRIC_DTYPE)

  for i, n in enumerate(STR_SCHEMA_ROW):
    metrics[n] = [x[i] for x in csv_metrics]

  frames = None
  for index, img in enumerate(images):
    if experiment_dir is not None:
      img = os.path.basename(img)
      img = os.path.join(experiment_dir, 'IMG', img)

    image = mpimg.imread(img)
    if frames is None:
      frames = np.zeros((len(images),) + image.shape, dtype=image.dtype)
    frames[index] = image

  # TODO: implement timeline recover from csv
  timeline = np.zeros(shape=(len(metrics),), dtype=np.float64)
  return Experience(frames=frames, metrics=metrics, timeline=timeline)


def load_replay(path):
  """
  load experience dump from autonomous mode
  :param path: path to experiment replay directory
  :return: Experience
  """
  metrics = np.load(os.path.join(path, 'metrics.npy'))
  frames = np.load(os.path.join(path, 'frames.npy'))
  timeline = np.load(os.path.join(path, 'timeline.npy'))
  return Experience(metrics=metrics, frames=frames, timeline=timeline)
