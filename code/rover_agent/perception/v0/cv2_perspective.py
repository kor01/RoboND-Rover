from rover_config import RoverConfig

from rover_agent.model_ctr import clip_fit_map

from rover_model.geometry import rotation_matrix_2d
from rover_model.geometry import translation
from rover_model.geometry import circle_distance
from rover_model.geometry import convert_camera_coords


from rover_agent.state_action import RoverState
from rover_agent.perception import Perception
from rover_agent.perception import register_perception

from rover_agent.model_ctr import create_threshold_extractor
from rover_agent.model_ctr import create_cv2_perspective
from rover_agent.model_ctr import unique_particles

from .maneuverable_particles import ManeuverableParticles


@register_perception
class CV2Perspective(Perception):
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

  def initialize(self, config: RoverConfig):
    config = config.internal()
    self._selective = config.getboolean('PERCEPTION', 'Selective')
    self._roll_bar = config.getfloat('PERCEPTION', 'RollBar')
    self._pitch_bar = config.getfloat('PERCEPTION', 'PitchBar')

  def create_state(self):
    return ManeuverableParticles()

  def update_state(self, state: RoverState):
    coords = convert_camera_coords(state.raw.img)
    particles = self._extractor.extract_particles(
      coords, singularity=None)
    s_coords = self._perspective.particle_transform(
      roll=None, pitch=None, particles=particles)
    self.write_debug_info('topdown', s_coords)
    rotation = rotation_matrix_2d(state.raw.yaw)
    particles = rotation @ s_coords
    particles = translation(*state.raw.pos, particles)
    particles = clip_fit_map(particles)

    # unique particles and save debug information
    self.write_debug_info(
      'map_gradients', unique_particles(particles))

    if not self._selective:
      state.raw.update_navigable_map(*particles)
    else:
      if circle_distance(state.raw.pitch, 0) <= self._pitch_bar:
        if circle_distance(state.raw.roll, 0) <= self._roll_bar:
          state.raw.update_navigable_map(*particles)
