import numpy as np

from rover.state_action import RoverState
from rover.state_action import ActionDistribution
from rover.frame_geometry import to_polar_coords

from rover.navi_policy import ProactivePolicy
from rover.navi_policy.proactive import register_proactor
from rover.state_action import continuous_steer_to_discrete


@register_proactor
class MaxAnglePolicy(ProactivePolicy):

  def initialize(self, config):
    pass

  def create_state(self):
    return None

  def evaluate(self, state: RoverState,
               action: ActionDistribution) -> ActionDistribution:

    particles = state.perception.particles
    r, theta = to_polar_coords(*particles)

    if len(theta) == 0:
      theta_mean = -np.pi
    else:
      theta_mean = np.mean(theta)
    assert -np.pi <= theta_mean <= np.pi, (theta_mean, theta)

    theta_mean = continuous_steer_to_discrete(theta_mean)

    self.write_debug_info('max_angle', theta_mean)

    # almost deterministic steer and throttle policy
    steer = action.steer
    steer[theta_mean] *= 10000
    steer /= steer.sum()

    throttle = action.throttle
    throttle[19] *= 10000
    throttle /= throttle.sum()

    self.write_debug_info('max_angle_index', np.argmax(steer))

    ret = ActionDistribution(
      throttle=throttle, brake=action.brake,
      steer=steer, switch=np.array([0.0001, 0.9999], dtype=np.float32))
    return ret

