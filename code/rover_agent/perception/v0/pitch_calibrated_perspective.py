import numpy as np

from rover_agent.frame_geometry import inverse_rotation_matrix
from rover_agent.frame_geometry import translation
from rover_agent.frame_geometry import clip_coordinates
from rover_agent.frame_geometry import convert_camera_coords

from rover_agent.state_action import RoverState
from rover_agent.perception import Perception
from rover_agent.perception import register_perception
from rover_resource import DEFAULT_THRESHOLD
from rover_agent.frame_geometry import create_interpolation
from rover_agent.frame_geometry import create_pitch_perspective
from rover_agent.frame_geometry import PIXEL_SCALING


class ManeuverableParticles(object):

  def __init__(self):
    self._particles = None

  @property
  def particles(self):
    assert self._particles is not None
    return self._particles

  def set_particles(self, particles: np.ndarray):
    self._particles = particles


VIEW_LIMIT = 20


@register_perception
class PitchCalibratedPerspective(Perception):
  """
  baseline perception module
  using a constant perspective without perspective calibration
  without rock detection; simply update map
  """

  def initialize(self, config):

    self._interpo = create_interpolation()

    self._perspect = \
      create_pitch_perspective()
    self._thresh = DEFAULT_THRESHOLD

  def create_state(self):
    return ManeuverableParticles()

  def update_state(self, state: RoverState):

    coords = convert_camera_coords(state.raw.img)

    pitch = (state.raw.pitch / 180) * np.pi

    # drop near singular measure for high fidelity
    singular = int(self._perspect.get_singular(pitch) * PIXEL_SCALING) - 3

    particles = self._interpo.extract_particles(
      coords, singular=singular)
    s_coords = self._perspect.particle_transform(
      particles=particles, pitch=pitch)

    #predicates = s_coords[:, 0] < VIEW_LIMIT
    #s_coords = s_coords[predicates, :]

    # clip out the remote particles where the precision
    #  is lost due to perspective singularity
    s_coords = s_coords.transpose((1, 0))
    # experimental reflection by x to see if there are bugs
    #s_coords[0, :] *= -1

    rotation = inverse_rotation_matrix(state.raw.yaw)
    # rotate back
    particles = rotation @ s_coords
    particles = translation(*state.raw.pos, particles)
    particles = clip_coordinates(particles)

    state.raw.clear_map()
    if state.raw.roll < 1:
      self.write_debug_info('write_map', 1)
      state.raw.update_navigable_map(*particles)
    else:
      state.raw.update_navigable_map(*particles)
      self.write_debug_info('write_map', 0)
