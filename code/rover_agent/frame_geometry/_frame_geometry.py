import cv2
import numpy as np

from rover_resource import FRAME_SHAPE
from rover_resource import FRAME_ORIGIN
from rover_resource import DST_SIZE
from rover_resource import  WORLD_SIZE


def color_thresh(img, rgb_thresh=(0, 0, 0), dtype='uint8'):
  thresh = np.array(rgb_thresh)
  shape = [1] * img.ndim
  shape[-1] = -1
  thresh.reshape(shape)
  binary_image = img > thresh
  binary_image = np.all(binary_image, axis=-1)
  binary_image = binary_image.astype(dtype)
  return binary_image

def rover_coords(binary_img):
  # Extract xpos and ypos pixel positions from binary_img and
  # Convert xpos and ypos to rover-centric coordinates
  ypos, xpos = binary_img.nonzero()
  xpos = xpos.astype('float32')
  ypos = ypos.astype('float32')
  nxpos = FRAME_ORIGIN[1] - ypos
  nypos = FRAME_ORIGIN[0] - xpos
  return np.array([nxpos, nypos])

def inverse_rotation_matrix(yaw: float):
  yaw = (yaw / 180.0) * np.pi
  ret = np.array([[np.cos(yaw), -np.sin(yaw)],
                  [np.sin(yaw), np.cos(yaw)]])
  return ret


def translation_and_scale(x, y, points):
  points = points.astype('float32')
  points /= 2 * DST_SIZE
  points[0] += x
  points[1] += y
  return points

def translation(x, y, points):
  points = points.astype('float32')
  points[0] += x
  points[1] += y
  return points

def clip_coordinates(x):
  ret = np.around(x).astype('uint32')
  ret = np.clip(ret, 0, WORLD_SIZE - 1)
  return ret

def circle_distance(a, b, rad_unit=False):
  unit = 2 * np.pi if rad_unit else 360
  assert  -1e-1 <= a <= unit, a
  assert  -1e-1 <= b <= unit, b
  d = np.absolute(b - a)
  cd = np.absolute(d - unit)
  return np.minimum(d, cd)

def perspective_transform(imgs: np.ndarray, projection: np.ndarray):
  expand = False
  if imgs.ndim == 3:
    imgs = imgs[np.newaxis,:]
    expand = True

  ret = np.zeros_like(imgs)
  size = (FRAME_SHAPE[1], FRAME_SHAPE[0])
  for index, img in enumerate(imgs):
    ret[index] = cv2.warpPerspective(img, projection, size)

  if expand:
    ret = ret[0]
  return ret

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

