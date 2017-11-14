import numpy as np

from rover.interface import MapUpdate
from rover.state import RoverState
from rover.interface import Perception
from rover.perception.ctr import register_perception

from rover.model_ctr import create_interpolation
from rover.model_ctr import create_pitch_perspective
from rover.model_ctr import clip_fit_map
from rover.model_ctr import singular_to_frame_pos
from rover.model_ctr import unique_particles

from rover.model.geometry import rotation_matrix_2d
from rover.model.geometry import translation
from rover.model.geometry import convert_camera_coords
from rover.model.geometry import degree_to_rad
from rover.model.geometry import quant_unique


@register_perception
class Pitch(Perception):
  """
  baseline perception module
  using a constant perspective without perspective calibration
  without rock detection; simply update map
  """

  def __init__(self, global_step, config, debug):
    self._extractor = None
    self._singular_drop = None
    self._thresh = None
    self._roll_thresh = None
    self._selective = None
    self._perspect = create_pitch_perspective()
    super().__init__(global_step, config, debug)

  def init(self, opt):
    self._thresh = np.array(
      eval(opt['Threshold']), dtype=np.uint32)
    self._selective = opt.getboolean('Selective')
    if self._selective:
      self._roll_thresh = opt.getfloat('RollThreshold')
    self._singular_drop = opt.getint('PERCEPTION', 'SingularDrop')
    self._extractor = create_interpolation(threshold=self._thresh)

  def init_ctx(self, state: RoverState):
    pass

  def update(self, state: RoverState):

    if self._selective and state.roll >= self._roll_thresh:
      update = MapUpdate(
        navi=([], []), obstacle=([], []), rock=([], []))
      return update, None

    coords = convert_camera_coords(state.img)
    pitch = degree_to_rad(state.pitch)
    singular = self._perspect.get_singular(pitch=pitch, roll=None)
    singular = singular_to_frame_pos(singular, self._singular_drop)
    particles = self._extractor.extract_particles(
      coords, singularity=singular)
    s_coords = self._perspect.particle_transform(
      particles=particles, pitch=pitch, roll=None)
    s_coords = quant_unique(s_coords, 8)
    rotation = rotation_matrix_2d(state.yaw)
    particles = rotation @ s_coords
    particles = translation(*state.pos, particles)
    particles = clip_fit_map(particles)
    particles = unique_particles(particles)

    update = MapUpdate(
      navi=particles, obstacle=([], []), rock=([], []))
    return update, None
