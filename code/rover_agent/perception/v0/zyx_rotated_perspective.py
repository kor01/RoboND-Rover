import numpy as np

from rover_config import RoverConfig

from rover_agent.frame_geometry import inverse_rotation_matrix
from rover_agent.frame_geometry import translation
from rover_agent.frame_geometry import clip_coordinates
from rover_agent.frame_geometry import convert_camera_coords

from rover_agent.state_action import RoverState
from rover_agent.perception import Perception

from rover_agent.perception import register_perception
from rover_agent.frame_geometry import create_interpolation
from rover_agent.frame_geometry import create_zyx_perspective
from rover_agent.frame_geometry import create_pitch_perspective
from rover_agent.frame_geometry import circle_distance

from rover_resource import PIXEL_SCALING

from .maneuverable_particles import ManeuverableParticles


@register_perception
class ZYXRotatedPerspective(Perception):
  """
  baseline perception module
  using a constant perspective without perspective calibration
  without rock detection; simply update map
  """

  def initialize(self, config: RoverConfig):

    self._thresh = np.array(
      eval(config['PERCEPTION']['Threshold']), dtype=np.uint32)

    self._interpo = create_interpolation(threshold=self._thresh)
    self._perspect = create_zyx_perspective()
    self._pitch_perspective = create_pitch_perspective()
    self._roll_thresh = config.internal().getfloat(
      'PERCEPTION', 'RollThreshold')

    self._pitch_singular_drop = \
      config.internal().getint('PERCEPTION', 'PitchSingularDrop')

    self._zyx_singular_drop = \
      config.internal().getint('PERCEPTION', 'ZYXSingularDrop')

    self._horizon_length = 320.0 / PIXEL_SCALING

  def create_state(self):
    return ManeuverableParticles()

  def update_state(self, state: RoverState):

    coords = convert_camera_coords(state.raw.img)

    pitch = (state.raw.pitch / 180) * np.pi

    roll = -(state.raw.roll / 180) * np.pi



    dist = circle_distance(state.raw.roll, 0)

    if dist > self._roll_thresh:
      self.write_debug_info('type', 'zyx')
      singular = self._perspect.get_singular(
        pitch=pitch, roll=roll,
        horizon_length=self._horizon_length)

      # drop near singular measure for high fidelity
      singular = int(singular * PIXEL_SCALING) \
                 - self._zyx_singular_drop

      particles = self._interpo.extract_particles(
        coords, singular=singular)
      s_coords = self._perspect.particle_transform(
        particles=particles, pitch=pitch, roll=roll)

    else:

      self.write_debug_info('type', 'pitch')
      singular = self._pitch_perspective.get_singular(pitch)

      # drop near singular measure for high fidelity
      singular = int(singular * PIXEL_SCALING) \
                 - self._pitch_singular_drop

      particles = self._interpo.extract_particles(
        coords, singular=singular)

      s_coords = self._pitch_perspective.particle_transform(
        particles=particles, pitch=pitch)

    #  is lost due to perspective singularity
    s_coords = s_coords.transpose((1, 0))

    rotation = inverse_rotation_matrix(state.raw.yaw)
    # rotate back
    particles = rotation @ s_coords
    particles = translation(*state.raw.pos, particles)
    particles = clip_coordinates(particles)

    # unique particles

    mp = np.zeros((200, 200), dtype=np.uint8)
    mp[particles[0], particles[1]] = 1
    map_gradients = np.array(mp.nonzero())

    self.write_debug_info('map_gradients', map_gradients)

    #state.raw.clear_map()
    state.raw.update_navigable_map(*particles)
