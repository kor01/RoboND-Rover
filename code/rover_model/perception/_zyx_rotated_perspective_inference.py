import numpy as np


def perspective_numerator(e, c, pitch, roll):
  ex, ey, ez = e
  cx, cy, cz = c
  ps, pc = np.sin(pitch), np.cos(pitch)
  rs, rc = np.sin(roll), np.cos(roll)

  w = np.array([[ez * rc * (ez + cx * pc - cz * ps),
                 -ez * (ez + cx * pc - cz * ps) * rs],
                [ez * (pc * (-ey + cy * rc) + (cz - ez * ps) * rs),
                 ez * (rc * (cz - ez * ps) + pc * (ex - cy * rs))]])

  b = np.array([[ez * (pc * (cz * ez - cx * ex * rc + cx * ey * rs) +
                       ps * (cx * ez + cz * ex * rc - cz * ey * rs))],
                [ez * (-(cz * ey + cy * ex * pc) * rc +
                       cy * ez * ps + (-cz * ex + cy * ey * pc) * rs)]])

  return w, b

def perspective_denominator(pitch, roll, e):

  ps, pc = np.sin(pitch), np.cos(pitch)
  rs, rc = np.sin(roll), np.cos(roll)
  ex, ey, ez = e

  w = np.array([ez * pc * rc,  -ez * pc * rs])
  b = np.array([ez * (ez * ps + pc * (-ex * rc + ey * rs))])
  return w, b


def generate_rotation(roll, pitch):
  theta = np.array([pitch, roll])
  (ps, rs), (pc, rc) = np.sin(theta), np.cos(theta)
  rr = np.array([[1, 0, 0], [0, rc, -rs], [0, rs, rc]], dtype=np.float64)
  rp = [[pc, 0, ps], [0, 1, 0], [-ps, 0, pc]]
  return np.matmul(rp, rr)


def singular_line(ty, e, pitch, roll):
  ex, ey, ez = e
  ps, pc = np.sin(pitch), np.cos(pitch)
  rs, rc = np.sin(roll), np.cos(roll)
  return ex - ez * ps / pc  - (ey - ty) * rs / (rc *  pc)


class ZYXRotatedPerspectiveInference(object):
  def __init__(self, camera_pos, view_pos):
    self._c = camera_pos
    self._e = view_pos


  def get_singular(self, pitch, roll, horizon_length):

    min_x = min(singular_line(0, self._e, pitch, roll),
                singular_line(horizon_length, self._e, pitch, roll))
    return min_x


  def particle_transform(
      self, roll, pitch, particles) -> np.ndarray:

    r = generate_rotation(roll, pitch)
    c, e = r @ self._c, self._e

    w, b = perspective_numerator(e, c, pitch, roll)

    numerator = w @ particles + b

    w, b = perspective_denominator(pitch, roll, e)

    denominator = w @ particles + b
    return (numerator / denominator).transpose()
