import numpy as np
from .interface import ParticleExtractor


class ThresholdExtractor(ParticleExtractor):

  def __init__(self, threshold=(160, 160, 160)):

    self._threshold = np.array(threshold)

  def extract_particles(self, coords, singularity):
    coords = coords > self._threshold
    coords = np.all(coords, axis=-1)
    coords = coords.astype('float32')
    ret = np.array(coords.nonzero())
    return ret
