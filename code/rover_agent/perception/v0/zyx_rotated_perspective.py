import numpy as np

from rover_config import RoverConfig

from rover_agent.state_action import RoverState
from rover_agent.perception import Perception
from rover_agent.perception import register_perception

from rover_model.geometry import rotation_matrix_2d
from rover_model.geometry import translation
from rover_model.geometry import convert_camera_coords
from rover_model.geometry import degree_to_rad
from rover_model.geometry import quant_unique
from rover_model.geometry import drop_range
from rover_model.geometry import drop_linear

from rover_agent.model_ctr import clip_fit_map
from rover_agent.model_ctr import get_horizon_length
from rover_agent.model_ctr import create_zyx_perspective
from rover_agent.model_ctr import create_pitch_perspective
from rover_agent.model_ctr import create_interpolation
from rover_agent.model_ctr import unique_particles
from rover_agent.model_ctr import singular_to_frame_pos
from rover_agent.model_ctr import pixel_scale_to_frame_scale

from .maneuverable_particles import ManeuverableParticles


@register_perception
class ZYXRotatedPerspective(Perception):
  """
  fully implemented ZYX calibrated perspective transform
  """

  def __init__(self, global_step, config, debug):
    self._extractor = None
    self._pitch_singular_drop = None
    self._zyx_singular_drop = None
    self._roll_thresh = None
    self._thresh = None
    self._horizon_length = get_horizon_length()
    self._perspect = create_zyx_perspective()
    self._pitch_perspective = create_pitch_perspective()
    super().__init__(global_step, config, debug)

  def initialize(self, config: RoverConfig):
    self._thresh = np.array(
      eval(config['PERCEPTION']['Threshold']), dtype=np.uint32)
    self._extractor = create_interpolation(threshold=self._thresh)
    self._zyx_singular_drop = \
      config.internal().getint('PERCEPTION', 'ZYXSingularDrop')

  def create_state(self):
    return ManeuverableParticles()

  def update_state(self, state: RoverState):
    coords = convert_camera_coords(state.raw.img)
    pitch = degree_to_rad(state.raw.pitch)
    roll = -degree_to_rad(state.raw.roll)

    singular = self._perspect.get_singular(
      pitch=pitch, roll=roll)
    singular = singular_to_frame_pos(singular, 0)
    # drop near singular measure for high fidelity
    particles = self._extractor.extract_particles(
      coords, singularity=singular)
    kx, ky, b = self._perspect.denominator(roll, pitch)
    drop = pixel_scale_to_frame_scale(
      self._zyx_singular_drop)
    particles = drop_linear(particles, kx, ky, b, drop)
    s_coords = self._perspect.particle_transform(
      particles=particles, pitch=pitch, roll=roll)

    # the out of range particle estimates
    # are resulted by singularity of projection
    s_coords = drop_range(s_coords, axis=0, low=0, high=40)
    s_coords = drop_range(s_coords, axis=1, low=-160, high=160)
    s_coords = quant_unique(s_coords, 8)
    self.write_debug_info('topdown', s_coords)
    rotation = rotation_matrix_2d(state.raw.yaw)
    # rotate back
    particles = rotation @ s_coords
    particles = translation(*state.raw.pos, particles)
    particles = clip_fit_map(particles)

    self.write_debug_info(
      'map_gradients', unique_particles(particles))

    state.raw.update_navigable_map(*particles)
