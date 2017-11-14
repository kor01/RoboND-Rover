"""
define particle extractor and perspective transform interface here
to ease case analysis implementation for different perception configuration
"""
import abc


class ParticleExtractor(metaclass=abc.ABCMeta):

  @abc.abstractmethod
  def extract_particles(self, coords, singularity):
    pass


class PerspectiveTransform(metaclass=abc.ABCMeta):

  @abc.abstractmethod
  def get_singular(self, roll, pitch):
    pass

  @abc.abstractmethod
  def particle_transform(self, roll, pitch, particles):
    pass
