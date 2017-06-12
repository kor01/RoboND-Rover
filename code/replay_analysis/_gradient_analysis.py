import numpy as np
from collections import namedtuple
import matplotlib.image as mpimg

GradientMetric = namedtuple('GradientMetric', ('pos', 'neg', 'render', 'ratio'))


class GradientAnalysis(object):

  def __init__(self, ground_truth):
    truth = mpimg.imread(ground_truth)
    self._truth = np.zeros(shape=truth.shape + (3,), dtype=np.uint8)
    self._truth[:, :, 1] = truth * 255

  def analysis(self, gradient, render=False):
    # noinspection PyUnresolvedReferences
    num_pos = (self._truth[gradient[1], gradient[0], 1] > 0).sum()
    num_neg = gradient.shape[1] - num_pos
    ratio = float(num_pos) / gradient.shape[1]
    if render:
        ret = self._truth.copy()
        ret[gradient[1], gradient[0], :] = 255
        render = np.flipud(ret)
    else:
        render = None
    return GradientMetric(pos=num_pos, neg=num_neg,
                          render=render, ratio=ratio)
