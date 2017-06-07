import os
import numpy as np
from rover_resource import METRIC_DTYPE


class RoverReplay(object):

  def __init__(self, save_path):
    self._frame = []
    self._metrics = []
    self._timeline = []
    self._path = save_path

  def consume(self, frame, metrics, current_time):
    self._frame.append(frame)
    self._metrics.append(metrics)
    self._timeline.append(current_time)

  def flush(self):
    timeline = np.array(self._timeline, dtype=np.float64)
    metrics = np.array(self._metrics, dtype=METRIC_DTYPE)
    frames = np.array(self._frame, dtype=np.uint8)
    print('frame shape:', frames.shape)
    print('metric shape:', metrics.shape)
    print('timeline shape:', timeline.shape)
    np.save(os.path.join(self._path, 'timeline.npy'), timeline)
    np.save(os.path.join(self._path, 'frames.npy'), frames)
    np.save(os.path.join(self._path, 'metrics.npy'), metrics)
