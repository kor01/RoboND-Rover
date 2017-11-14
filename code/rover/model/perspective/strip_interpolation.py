import cv2
import numpy as np
from .interface import ParticleExtractor


def zoom_in(img: np.ndarray, boundary, factor: np.ndarray):
  img = img[boundary[0]: boundary[1], :, :]
  factor = factor.flatten()
  shape = factor * img.shape[:2]
  return cv2.resize(img, (shape[1], shape[0]), fx=0, fy=0)


class StripwiseInterpolation(ParticleExtractor):
  def __init__(self, view_singular, pixel_factor, extractor):
    self._pixel_factor = pixel_factor
    self._extractor = extractor
    self._adjust_parameter(view_singular)

  def _adjust_parameter(self, view_singular):
    # view_singular = view_singular - 10
    self._view_singular = view_singular

    lower_boundary = (0, int(view_singular * 0.8))

    strip_size = view_singular - lower_boundary[1]

    self._mid_factor, self._top_factor \
      = np.array([[3], [2]]), np.array([[6], [2]])

    self._boundary_factor = 2

    self._lower_boundary = lower_boundary

    strip_unit = int(strip_size / 2)
    lower = lower_boundary[1]

    self._mid_boundary = \
      (lower - strip_unit, lower + strip_unit)

    self._top_boundary = \
      (lower, lower + strip_unit * 2)

  def _extract_strip(self, coords, boundary, factor):
    strip = zoom_in(coords, boundary, factor)
    particles = self._extractor(strip)
    particles = particles / factor.astype('float32')
    particles[0, :] += boundary[0]
    particles /= self._pixel_factor
    return particles

  def extract_particles(
      self, coords: np.ndarray, singularity=None):
    if singularity is not None:
      self._adjust_parameter(singularity)

    boundary = self._lower_boundary

    lower_particles = self._extractor(
      coords[boundary[0]: boundary[1], :, :])

    lower_particles = lower_particles / self._pixel_factor

    mid_particles = self._extract_strip(
      coords, self._mid_boundary, self._mid_factor)

    top_particles = self._extract_strip(
      coords, self._top_boundary, self._top_factor)

    return np.concatenate(
      [lower_particles, mid_particles, top_particles], axis=1)
