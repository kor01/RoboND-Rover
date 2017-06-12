import numpy as np

from rover_agent.state_action import RoverState
from rover_agent.perception import Perception
from rover_agent.perception import register_perception

from rover_agent.model_ctr import create_interpolation
from rover_agent.model_ctr import create_pitch_perspective
from rover_agent.model_ctr import clip_fit_map
from rover_agent.model_ctr import singular_to_frame_pos
from rover_agent.model_ctr import unique_particles

from rover_model.geometry import rotation_matrix_2d
from rover_model.geometry import translation
from rover_model.geometry import convert_camera_coords
from rover_model.geometry import degree_to_rad
from rover_model.geometry import quant_unique

from .maneuverable_particles import ManeuverableParticles


@register_perception
class PitchCalibratedPerspective(Perception):
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

  def initialize(self, config):
    config = config.internal()
    self._thresh = np.array(
      eval(config['PERCEPTION']['Threshold']), dtype=np.uint32)
    self._selective = config.getboolean('PERCEPTION', 'Selective')
    if self._selective:
      self._roll_thresh = config.getfloat('PERCEPTION', 'RollThreshold')
    self._singular_drop = config.getint('PERCEPTION', 'SingularDrop')
    self._extractor = create_interpolation(threshold=self._thresh)

  def create_state(self):
    return ManeuverableParticles()

  def update_state(self, state: RoverState):

    coords = convert_camera_coords(state.raw.img)
    pitch = degree_to_rad(state.raw.pitch)
    singular = self._perspect.get_singular(pitch=pitch, roll=None)
    singular = singular_to_frame_pos(singular, self._singular_drop)
    particles = self._extractor.extract_particles(
      coords, singularity=singular)

    s_coords = self._perspect.particle_transform(
      particles=particles, pitch=pitch, roll=None)
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
      return

    if state.raw.roll < self._roll_thresh:
      self.write_debug_info('type', 'pitch')
      state.raw.update_navigable_map(*particles)
    else:
      self.write_debug_info('type', 'drop')
