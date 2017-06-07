from rover_agent.frame_geometry import inverse_rotation_matrix
from rover_agent.frame_geometry import translation
from rover_agent.frame_geometry import clip_coordinates
from rover_agent.frame_geometry import convert_camera_coords
from rover_agent.frame_geometry import create_calibrated_perspective
from rover_agent.frame_geometry import create_interpolation
from rover_agent.frame_geometry import circle_distance

from rover_agent.state_action import RoverState
from rover_agent.perception import Perception
from rover_agent.perception import register_perception

from .maneuverable_particles import ManeuverableParticles


@register_perception
class CalibratedPerspective(Perception):
  """
  baseline perception module
  using a constant perspective without perspective calibration
  without rock detection; simply update map
  """

  def initialize(self, config):
    self._interpo = create_interpolation()
    self._perspect = create_calibrated_perspective()

    self._selective = config.getboolean('PERCEPTION', 'Selective')
    if self._selective:
      self._roll_bar = config.getfloat('PERCEPTION', 'RollBar')
      self._pitch_bar = config.getfloat('PERCEPTION', 'PitchBar')

  def create_state(self):
    return ManeuverableParticles()

  def update_state(self, state: RoverState):

    coords = convert_camera_coords(state.raw.img)
    particles = self._interpo.extract_particles(coords)
    s_coords = self._perspect.particle_transform(particles)
    s_coords = s_coords.transpose((1, 0))

    rotation = inverse_rotation_matrix(state.raw.yaw)
    # rotate back
    particles = rotation @ s_coords
    particles = translation(*state.raw.pos, particles)
    particles = clip_coordinates(particles)

    if not self._selective:
      state.raw.update_navigable_map(*particles)
    else:
      if circle_distance(state.raw.pitch, 0) <= self._pitch_bar:
        if circle_distance(state.raw.roll, 0) <= self._roll_bar:
          state.raw.update_navigable_map(*particles)
