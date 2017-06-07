import numpy as np

from rover_model.perception import CalibratedPerspectiveInference
from rover_model.perception import StripwiseInterpolation
from rover_model.perception import PitchCalibratedPerspectiveInference
from rover_model.perception import ZYXRotatedPerspectiveInference
from rover_model.perception import ZXYRotatedPerspectiveInference


from rover_resource import VIEW_POINT_POSITION
from rover_resource import PIXEL_SCALING
from rover_resource import CAMERA_POSITION

from ._frame_geometry import color_thresh
from rover_resource import DEFAULT_THRESHOLD


def create_calibrated_perspective():
  perspect = CalibratedPerspectiveInference(CAMERA_POSITION, VIEW_POINT_POSITION)
  return perspect

def create_interpolation(threshold=None):

  threshold = threshold or DEFAULT_THRESHOLD

  singular_pixel = int(VIEW_POINT_POSITION[0] * PIXEL_SCALING)
  interpo = StripwiseInterpolation(
    singular_pixel, PIXEL_SCALING,
    lambda x: np.array(color_thresh(x, threshold, 'float32').nonzero()))
  return interpo


def create_pitch_perspective():
  return PitchCalibratedPerspectiveInference(
    CAMERA_POSITION, VIEW_POINT_POSITION)


def create_zyx_perspective():
  return ZYXRotatedPerspectiveInference(
    CAMERA_POSITION, VIEW_POINT_POSITION)


def create_zxy_perspective():
  return ZXYRotatedPerspectiveInference(
    CAMERA_POSITION, VIEW_POINT_POSITION)

