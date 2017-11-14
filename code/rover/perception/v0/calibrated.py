from rover_model.geometry import rotation_matrix_2d
from rover_model.geometry import convert_camera_coords
from rover_model.geometry import translation
from rover_model.geometry import quant_unique

from rover.model_ctr import create_calibrated_perspective
from rover.model_ctr import create_interpolation
from rover.model_ctr import clip_fit_map

from rover.state import RoverState
from rover.interface import Perception
from rover.interface import MapUpdate
from rover.perception.ctr import register_perception


@register_perception
class Calibrated(Perception):
  """
  baseline perception module
  using a constant perspective without perspective calibration
  without rock detection; simply update map
  """

  def __init__(self, global_step, config, debug):

    self._extractor = create_interpolation()
    self._perspect = create_calibrated_perspective()
    self._selective = False
    self._roll_bar = None
    self._pitch_bar = None
    self._singular_drop = None
    self._singular = None
    super().__init__(global_step, config, debug)

  def init(self, opt):
    self._singular_drop = opt.getint('SingularDrop')
    self._selective = opt.getboolean('Selective')
    if self._selective:
      self._roll_bar = opt.getfloat('RollBar')
      self._pitch_bar = opt.getfloat('PitchBar')
    singular = self._perspect.get_singular(roll=None, pitch=None)
    self._singular = singular

  def init_ctx(self, state: RoverState):
    pass

  def update(self, state: RoverState):

    if self._selective:
      if state.roll >= self._roll_bar \
        or state.pitch >= self._pitch_bar:
        update = MapUpdate(
          navi=([], []), obstacle=([], []), rock=([], []))
        return update, None

    coords = convert_camera_coords(state.img)
    particles = self._extractor.extract_particles(
      coords, singularity=self._singular)

    s_coords = self._perspect.particle_transform(
      roll=None, pitch=None, particles=particles)

    s_coords = quant_unique(s_coords, 8)
    rotation = rotation_matrix_2d(state.yaw)
    particles = rotation @ s_coords
    particles = translation(*state.pos, particles)
    particles = clip_fit_map(particles)

    update = MapUpdate(
      navi=particles, obstacle=([], []), rock=([], []))
    return update, None
