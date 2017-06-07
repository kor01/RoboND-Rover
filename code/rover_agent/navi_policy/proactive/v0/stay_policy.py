import numpy as np

from rover_agent.state_action import RoverState
from rover_agent.state_action import ActionDistribution

from rover_agent.navi_policy import ProactivePolicy
from rover_agent.navi_policy.proactive import register_proactor

from rover_agent.state_action import ZERO_BRAKE
from rover_agent.state_action import ZERO_THROTTLE
from rover_agent.state_action import ZERO_STEER


@register_proactor
class StayPolicy(ProactivePolicy):

  def initialize(self, config):
    pass

  def create_state(self):
    return None

  def evaluate(self, state: RoverState,
               action: ActionDistribution) -> ActionDistribution:

    switch = np.ones_like(action.switch)
    switch / switch.sum()

    throttle = np.zeros_like(action.throttle)
    throttle[ZERO_THROTTLE] = 1

    brake = np.zeros_like(action.brake)
    brake[ZERO_BRAKE] = 1

    steer = np.zeros_like(action.steer)
    steer[ZERO_STEER] = 1

    ret = ActionDistribution(
      throttle=throttle,
      brake=brake,
      steer=steer,
      switch=switch)

    return ret

