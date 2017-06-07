import numpy as np
from rover_agent.navi_policy import ReactivePolicy
from rover_agent.state_action import RoverState
from rover_agent.state_action import ActionDistribution
from rover_agent.navi_policy.reactive import register_reactor

from rover_agent import state_action as sa

@register_reactor
class UniformPolicy(ReactivePolicy):
  """
  baseline policy: no reactive policy implemented
  """

  def initialize(self, config):
    """
    UniformPolicy needs not initialization
    """
    pass

  def create_state(self):
    return None

  def evaluate(self, state: RoverState) -> ActionDistribution:
    throttle = np.ones(shape=sa.THROTTLE_SHAPE, dtype=np.float32)
    brake = np.ones(shape=sa.BRAKE_SHAPE, dtype=np.float32)
    steer = np.ones(shape=sa.STEER_SHAPE, dtype=np.float32)
    switch = np.ones(shape=(2,), dtype=np.float32)

    return ActionDistribution(
      throttle=throttle / throttle.sum(),
      brake=brake / brake.sum(),
      steer=steer / steer.sum(),
      switch=switch / switch.sum())
