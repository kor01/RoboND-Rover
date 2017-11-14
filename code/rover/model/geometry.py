import numpy as np


def color_thresh(img, rgb_thresh=(0, 0, 0), dtype='uint8'):
  thresh = np.array(rgb_thresh)
  shape = [1] * img.ndim
  shape[-1] = -1
  thresh.reshape(shape)
  binary_image = img > thresh
  binary_image = np.all(binary_image, axis=-1)
  binary_image = binary_image.astype(dtype)
  return binary_image


def rotation_matrix_2d(yaw: float):
  yaw = (yaw / 180.0) * np.pi
  ret = np.array([[np.cos(yaw), -np.sin(yaw)],
                  [np.sin(yaw), np.cos(yaw)]])
  return ret


def translation(x, y, points):
  points = points.astype('float32')
  points[0] += x
  points[1] += y
  return points


def clip_to_integer(x, max_size):
  ret = np.around(x).astype('uint32')
  ret = np.clip(ret, 0, max_size - 1)
  return ret


def circle_distance(a, b, rad_unit=False):
  unit = 2 * np.pi if rad_unit else 360
  assert -1e-1 <= a <= unit, a
  assert -1e-1 <= b <= unit, b
  d = np.absolute(b - a)
  cd = np.absolute(d - unit)
  return np.minimum(d, cd)


def convert_camera_coords(image_or_coords: np.ndarray) -> np.ndarray:
  """
  convention: left down corner as image origin,
  vertical up direction as x axis
  horizontal right direction as y axis
  z axis points into the picture
  :param image_or_coords: the image or coords tensor
  :return: the camera coordinates tensor or image
  """
  return np.flip(image_or_coords, axis=0)


def to_polar_coords(x, y):
  r = np.sqrt(x**2 + y**2)
  theta = np.arctan2(y, x)
  theta[theta == np.nan] = 0
  pred = theta >= -np.pi
  pred = np.logical_and(theta <= np.pi, pred)
  assert pred.all()
  return r, theta


def degree_to_rad(degree):
  return (degree / 180) * np.pi


def rad_to_degree(rad):
  return (rad / np.pi) * 180


def _quant_get_param(arr, precision):
  arr_min = arr.min()
  arr_span = arr.max() - arr_min
  return arr_min, int(arr_span * precision + 1)


def quant_unique(coords, precision):

  x_min, x_span = _quant_get_param(coords[0, :], precision)
  y_min, y_span = _quant_get_param(coords[1, :], precision)
  quants = np.zeros((x_span, y_span), dtype=np.uint8)

  x_quant = np.around((coords[0, :] - x_min) * precision)
  y_quant = np.around((coords[1, :] - y_min) * precision)

  x_quant = np.clip(x_quant, 0, x_span - 1).astype('uint32')
  y_quant = np.clip(y_quant, 0, y_span - 1).astype('uint32')
  quants[x_quant, y_quant] += 1
  x_quant, y_quant = quants.nonzero()

  x_coords = x_quant.astype('float32') / precision + x_min
  y_coords = y_quant.astype('float32') / precision + y_min

  return np.array([x_coords, y_coords])


def drop_negative(coords, axis):
  predicate = coords[axis, :] >= 0
  coords = coords[:, predicate]
  return coords


def drop_range(coords, axis, low, high):
  predicate = coords[axis, :] <= high
  predicate = np.logical_and(predicate, coords[axis, :] >= low)
  coords = coords[:, predicate]
  return coords


def drop_linear(coords, kx, ky, b, drop):
  param = np.array([kx, ky, b])
  param = param / param[0]
  factor = np.sqrt(1 + (param[1]) ** 2)
  drop = - drop * factor
  combo = param[:2] @ coords + param[2]
  combo = combo / param[0]
  predicate = combo < drop
  ret = coords[:, predicate]
  return ret
