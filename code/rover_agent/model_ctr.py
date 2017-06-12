import numpy as np

from rover_model.perception import CalibratedPerspectiveInference
from rover_model.perception import StripwiseInterpolation
from rover_model.perception import PitchCalibratedPerspectiveInference
from rover_model.perception import ZYXRotatedPerspectiveInference
from rover_model.perception import ZXYRotatedPerspectiveInference
from rover_model.perception import CV2Perspective
from rover_model.perception import ThresholdExtractor

from rover_model.geometry import color_thresh
from rover_model.geometry import clip_to_integer

from rover_spec import VIEW_POINT_POSITION
from rover_spec import PIXEL_SCALING
from rover_spec import CAMERA_POSITION
from rover_spec import STD_PERSPECTIVE_SOURCE
from rover_spec import STD_PERSPECTIVE_TARGET
from rover_spec import DEFAULT_THRESHOLD
from rover_spec import FRAME_SHAPE
from rover_spec import DST_SIZE
from rover_spec import WORLD_SIZE


def create_calibrated_perspective():
  perspect = CalibratedPerspectiveInference(CAMERA_POSITION, VIEW_POINT_POSITION)
  return perspect


def clip_fit_map(particles):
  return clip_to_integer(particles, WORLD_SIZE)


def singular_to_frame_pos(singular, drop):
  return int(singular * PIXEL_SCALING) - drop


def pixel_scale_to_frame_scale(quant):
  return quant / float(PIXEL_SCALING)


def create_interpolation(threshold=None):

  threshold = threshold if threshold is not None else DEFAULT_THRESHOLD

  singular_pixel = int(VIEW_POINT_POSITION[0] * PIXEL_SCALING)
  interpo = StripwiseInterpolation(
    singular_pixel, PIXEL_SCALING,
    lambda x: np.array(color_thresh(x, threshold, 'float32').nonzero()))
  return interpo


def create_pitch_perspective():
  return PitchCalibratedPerspectiveInference(
    CAMERA_POSITION, VIEW_POINT_POSITION)


def create_zyx_perspective():
  horizon_length = float(FRAME_SHAPE[1]) / PIXEL_SCALING
  return ZYXRotatedPerspectiveInference(
    CAMERA_POSITION, VIEW_POINT_POSITION, horizon_length)


def create_zxy_perspective():
  horizon_length = float(FRAME_SHAPE[1]) / PIXEL_SCALING
  return ZXYRotatedPerspectiveInference(
    CAMERA_POSITION, VIEW_POINT_POSITION, horizon_length)


def create_cv2_perspective():
  return CV2Perspective(
    STD_PERSPECTIVE_SOURCE, STD_PERSPECTIVE_TARGET,
    FRAME_SHAPE, 2 * DST_SIZE)


def create_threshold_extractor():
  return ThresholdExtractor(DEFAULT_THRESHOLD)


def get_horizon_length():
  return PIXEL_SCALING / float(FRAME_SHAPE[1])


def unique_particles(particles):
  mp = np.zeros((WORLD_SIZE, WORLD_SIZE), dtype=np.uint8)
  mp[particles[0], particles[1]] = 1
  ret = np.array(mp.nonzero())
  return ret
