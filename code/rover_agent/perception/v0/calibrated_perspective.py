from rover_model.geometry import rotation_matrix_2d
from rover_model.geometry import convert_camera_coords
from rover_model.geometry import circle_distance
from rover_model.geometry import translation
from rover_model.geometry import quant_unique

from rover_agent.model_ctr import create_calibrated_perspective
from rover_agent.model_ctr import create_interpolation
from rover_agent.model_ctr import clip_fit_map
from rover_agent.model_ctr import singular_to_frame_pos

from rover_agent.state_action import RoverState
from rover_agent.perception import Perception
from rover_agent.perception import register_perception
from rover_agent.model_ctr import unique_particles

from .maneuverable_particles import ManeuverableParticles


@register_perception
class CalibratedPerspective(Perception):
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

  def initialize(self, config):
    config = config.internal()
    self._singular_drop = config.getint('PERCEPTION', 'SingularDrop')
    self._selective = config.getboolean('PERCEPTION', 'Selective')
    if self._selective:
      self._roll_bar = config.getfloat('PERCEPTION', 'RollBar')
      self._pitch_bar = config.getfloat('PERCEPTION', 'PitchBar')
    singular = self._perspect.get_singular(roll=None, pitch=None)
    singular = singular_to_frame_pos(singular, self._singular_drop)
    self._singular = singular

  def create_state(self):

    return ManeuverableParticles()
  
  def update_state(self, state: RoverState):

    coords = convert_camera_coords(state.raw.img)
    particles = self._extractor.extract_particles(
      coords, singularity=self._singular)
    
    s_coords = self._perspect.particle_transform(
      roll=None, pitch=None, particles=particles)

    s_coords = quant_unique(s_coords, 8)
    self.write_debug_info('topdown', s_coords)

    rotation = rotation_matrix_2d(state.raw.yaw)
    # rotate back
    particles = rotation @ s_coords
    particles = translation(*state.raw.pos, particles)
    particles = clip_fit_map(particles)

    self.write_debug_info(
      'map_gradients', unique_particles(particles))
    if not self._selective:
      state.raw.update_navigable_map(*particles)
    else:
      if circle_distance(state.raw.pitch, 0) <= self._pitch_bar:
        if circle_distance(state.raw.roll, 0) <= self._roll_bar:
          self.write_debug_info('type', 'calibrated')
          state.raw.update_navigable_map(*particles)
        else:
          self.write_debug_info('type', 'drop')