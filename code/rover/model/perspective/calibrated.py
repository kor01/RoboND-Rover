import numpy as np
from .interface import PerspectiveTransform


def perspective_matrix_bias(e, c):
  ex, ey, ez = e
  cx, cy, cz = c
  w = np.array([[-cx - ez, 0], [-cy + ey, -cz - ex]],
               dtype=np.float64)
  b = np.array([[cx * ex - ez * cz], [cy * ex + cz * ey]],
               dtype=np.float64)
  return w, b


class CalibratedPerspective(PerspectiveTransform):
  """
  calibrated perspective transform

  parameter obtained by numeric optimization
  and strong geometric assumption

  """

  def __init__(self, camera_pos, view_pos):
    self._w, self._b = perspective_matrix_bias(
      camera_pos, view_pos)
    self._ex = view_pos[0]

  def get_singular(self, roll, pitch):
    return self._ex

  def particle_transform(self, roll, pitch, particles):
    ret = self._w @ particles + self._b
    denominator = self._ex - particles[0:1, :]
    ret /= denominator
    return ret
