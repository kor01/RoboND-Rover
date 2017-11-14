from rover.interface import MapUpdate
from rover.model_ctr import clip_fit_map

from rover.model.geometry import rotation_matrix_2d
from rover.model.geometry import translation
from rover.model.geometry import convert_camera_coords

from rover.state import RoverState
from rover.interface import Perception
from rover.perception.ctr import register_perception

from rover.model_ctr import create_threshold_extractor
from rover.model_ctr import create_cv2_perspective


@register_perception
class CV2(Perception):
  """
  baseline perception module
  using a constant perspective without perspective calibration
  without rock detection; simply update map
  """

  def __init__(self, global_step, config, debug):
    self._selective = None
    self._roll_bar = None
    self._pitch_bar = None
    self._singular_drop = None
    self._extractor = create_threshold_extractor()
    self._perspective = create_cv2_perspective()
    super().__init__(global_step, config, debug)

  def init(self, opt):
    self._selective = opt.getboolean('Selective')
    self._roll_bar = opt.getfloat('PERCEPTION', 'RollBar')
    self._pitch_bar = opt.getfloat('PERCEPTION', 'PitchBar')

  def update(self, state: RoverState):

    if self._selective:
      if state.roll >= self._roll_bar \
          or state.pitch >= self._pitch_bar:
        update = MapUpdate(
          navi=([], []), obstacle=([], []), rock=([], []))
        return update, None

    coords = convert_camera_coords(state.img)
    particles = self._extractor.extract_particles(
      coords, singularity=None)
    s_coords = self._perspective.particle_transform(
      roll=None, pitch=None, particles=particles)
    rotation = rotation_matrix_2d(state.yaw)
    particles = rotation @ s_coords
    particles = translation(*state.pos, particles)
    particles = clip_fit_map(particles)

    update = MapUpdate(
      navi=particles, obstacle=([], []), rock=([], []))

    return update, None

  def init_ctx(self, state: RoverState):
    pass
