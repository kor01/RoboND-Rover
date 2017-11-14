import numpy as np
from .interface import PerspectiveTransform


def perspective_denominator(e, pitch):

  pc, ps = np.cos(pitch), np.sin(pitch)
  ex, ey, ez = e

  w = np.array([[pc, 0]], dtype=np.float64)

  b = np.array([-ex * pc + ez * ps], dtype=np.float64)

  return w, b


def perspective_numerator(e, c, pitch):
  ex, ey, ez = e
  cx, cy, cz = c
  pc, ps = np.cos(pitch), np.sin(pitch)

  w = np.array([[ez + cx * pc - cz * ps, 0],
                [cy * pc - ey * pc, cz + ex * pc - ez * ps]])

  b = np.array([[(-cx * ex + cz * ez) * pc + (cz * ex + cx * ez) * ps],
                [-cz * ey - cy * ex * pc + cy * ez * ps]])

  return w, b


def generate_rotation(pitch):
  pc, ps = np.cos(pitch), np.sin(pitch)
  rp = np.array([[pc, 0, ps], [0, 1, 0], [-ps, 0, pc]], dtype=np.float64)
  return rp


class PitchCalibratedPerspective(PerspectiveTransform):

  def __init__(self, camera_pos, view_pos):
    self._e = view_pos
    self._c = camera_pos

  def get_singular(self, pitch, roll):
    ex, ez = self._e[0], self._e[2]
    return ex - ez * np.tan(pitch)

  def particle_transform(self, roll, pitch, particles):

    rot = generate_rotation(pitch)

    e, c = self._e, rot@self._c

    w, b = perspective_numerator(e, c, pitch)

    numerator = w @ particles + b

    w, b = perspective_denominator(e, pitch)

    denominator = w @ particles + b

    return numerator / denominator

