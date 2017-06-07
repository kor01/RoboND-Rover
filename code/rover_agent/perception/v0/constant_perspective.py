import cv2
import numpy as np

from rover_config import RoverConfig

from rover_agent.frame_geometry import color_thresh
from rover_agent.frame_geometry import inverse_rotation_matrix
from rover_agent.frame_geometry import perspective_transform
from rover_agent.frame_geometry import rover_coords
from rover_agent.frame_geometry import translation
from rover_agent.frame_geometry import clip_coordinates
from rover_agent.frame_geometry import circle_distance

from rover_agent.state_action import RoverState
from rover_agent.perception import Perception
from rover_agent.perception import register_perception

from rover_resource import STD_PERSPECTIVE_SOURCE
from rover_resource import STD_PERSPECTIVE_TARGET
from rover_resource import DEFAULT_THRESHOLD
from rover_resource import DST_SIZE

from .maneuverable_particles import ManeuverableParticles


@register_perception
class ConstantPerspective(Perception):
  """
  baseline perception module
  using a constant perspective without perspective calibration
  without rock detection; simply update map
  """

  def initialize(self, config: RoverConfig):

    config = config.internal()

    self._selective = config.getboolean('PERCEPTION', 'Selective')
    self._roll_bar = config.getfloat('PERCEPTION', 'RollBar')
    self._pitch_bar = config.getfloat('PERCEPTION', 'PitchBar')

    self._proj = cv2.getPerspectiveTransform(
      STD_PERSPECTIVE_SOURCE, STD_PERSPECTIVE_TARGET)
    self._proj.flags.writeable = False
    self._thresh = DEFAULT_THRESHOLD

  def create_state(self):
    return ManeuverableParticles()

  def update_state(self, state: RoverState):

    # perspective transform
    top_down = perspective_transform(state.raw.img, self._proj)
    top_down = color_thresh(top_down, self._thresh)

    state.raw.update_navigable_vision(top_down)

    # get all detected road particles
    particles = rover_coords(top_down)
    state.perception.set_particles(particles)

    rotation = inverse_rotation_matrix(state.raw.yaw)

    particles = rotation @ particles
    particles /= 2 * DST_SIZE
    particles = translation(*state.raw.pos, particles)
    particles = clip_coordinates(particles)

    # add debug information for bad case analysis
    mp = np.zeros((200, 200), dtype=np.uint8)
    mp[particles[0], particles[1]] = 1
    map_gradients = np.array(mp.nonzero())
    self.write_debug_info('map_gradients', map_gradients)

    if not self._selective:
      state.raw.update_navigable_map(*particles)
    else:
      # update to world map
      if circle_distance(state.raw.pitch, 0) <= self._pitch_bar:
        if circle_distance(state.raw.roll, 0) <= self._roll_bar:
          state.raw.update_navigable_map(*particles)
