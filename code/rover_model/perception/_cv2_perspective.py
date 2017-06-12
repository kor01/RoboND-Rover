import cv2
import numpy as np
from ._interface import PerspectiveTransform
from ._interface import ParticleExtractor


class CV2Perspective(PerspectiveTransform):

  def get_singular(self, roll, pitch):
    """
    CV2Perspective use a build in interpolation scheme
    needs no singularity information
    """
    return None

  def __init__(self, source, target, frame_shape, scale):

    """
    :param source: the coordinates of reference points on image
    :param target: the coordinates of reference points on ground
    """
    self._projection = cv2.getPerspectiveTransform(source, target)
    # by cv2 coordinate frame convention
    self._shape = (frame_shape[1], frame_shape[0])
    self._frame_shape = frame_shape
    self._scale = scale

  def particle_transform(self, roll, pitch, particles):
    frame_shape = self._frame_shape
    image = np.zeros(shape=frame_shape, dtype=np.uint8)
    image[particles[0], particles[1]] = 1
    image = np.flip(image, axis=0)
    ret = cv2.warpPerspective(image, self._projection, self._shape)
    xs, ys = ret.nonzero()
    xs = (frame_shape[0] - xs).astype('float64')
    ys = (frame_shape[1] / 2 - ys).astype('float64')
    ret = np.array([xs, ys])
    ret = ret / self._scale
    return ret


class ThresholdExtractor(ParticleExtractor):

  def __init__(self, threshold=(160, 160, 160)):

    self._threshold = np.array(threshold)

  def extract_particles(self, coords, singularity):
    coords = coords > self._threshold
    coords = np.all(coords, axis=-1)
    coords = coords.astype('float32')
    ret = np.array(coords.nonzero())
    return ret
