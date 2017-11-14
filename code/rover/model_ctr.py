import numpy as np

from rover import spec
from rover.model import perspective
from rover.model.geometry import color_thresh
from rover.model.geometry import clip_to_integer

from rover_model.geometry import color_thresh
from rover_model.geometry import clip_to_integer


def create_calibrated_perspective():
  perspect = perspective.CalibratedPerspective(
    spec.CAMERA_POSITION, spec.VIEW_POINT_POSITION)
  return perspect


def clip_fit_map(particles):
  return clip_to_integer(particles, spec.WORLD_SIZE)


def singular_to_frame_pos(singular, drop):
  return int(singular * spec.PIXEL_SCALING) - drop


def pixel_scale_to_frame_scale(quant):
  return quant / float(spec.PIXEL_SCALING)


def create_interpolation(threshold=None):

  threshold = threshold if threshold is not None else spec.DEFAULT_THRESHOLD

  singular_pixel = int(spec.VIEW_POINT_POSITION[0] * spec.PIXEL_SCALING)
  interpo = perspective.StripwiseInterpolation(
    singular_pixel, spec.PIXEL_SCALING,
    lambda x: np.array(color_thresh(x, threshold, 'float32').nonzero()))
  return interpo


def create_pitch_perspective():
  return perspective.PitchCalibratedPerspective(
    spec.CAMERA_POSITION, spec.VIEW_POINT_POSITION)


def create_zyx_perspective():
  horizon_length = float(spec.FRAME_SHAPE[1]) / spec.PIXEL_SCALING
  return perspective.ZYXRotatedPerspective(
    spec.CAMERA_POSITION, spec.VIEW_POINT_POSITION, horizon_length)


def create_zxy_perspective():
  horizon_length = float(spec.FRAME_SHAPE[1]) / spec.PIXEL_SCALING
  return perspective.ZXYRotatedPerspective(
    spec.CAMERA_POSITION, spec.VIEW_POINT_POSITION, horizon_length)


def create_cv2_perspective():
  return perspective.CV2Perspective(
    spec.STD_PERSPECTIVE_SOURCE,
    spec.STD_PERSPECTIVE_TARGET,
    spec.FRAME_SHAPE, 2 * spec.DST_SIZE)


def create_threshold_extractor():
  return perspective.ThresholdExtractor(spec.DEFAULT_THRESHOLD)


def get_horizon_length():
  return spec.PIXEL_SCALING / float(spec.FRAME_SHAPE[1])


def unique_particles(particles):
  mp = np.zeros((spec.WORLD_SIZE, spec.WORLD_SIZE), dtype=np.uint8)
  mp[particles[0], particles[1]] = 1
  ret = np.array(mp.nonzero())
  return ret
