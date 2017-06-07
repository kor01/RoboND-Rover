import numpy as np


class ManeuverableParticles(object):

  def __init__(self):
    self._particles = None

  @property
  def particles(self):
    assert self._particles is not None
    return self._particles

  def set_particles(self, particles: np.ndarray):
    self._particles = particles

