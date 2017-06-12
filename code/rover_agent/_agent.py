from rover_config import RoverConfig

from rover_agent.perception import create_perception
from rover_agent.navi_policy.proactive import create_proactor
from rover_agent.navi_policy.reactive import create_reactor

from rover_agent.state_action import RoverState
from rover_agent.state_action import RawState
from rover_agent.state_action import ActionDistribution

from ._global_step import GlobalStep


class RoverAgent(object):

  def __init__(self, config: RoverConfig, debug=True):
    """
    :param config: a rover configuration object
    :param debug: open debug mode
    """

    self._step = GlobalStep()

    self._perception = create_perception(
      config.perception, config, self._step, debug)

    self._proactor = create_proactor(
      config.proactor, config, self._step, debug)
    self._reactor = create_reactor(
      config.reactor, config, self._step, debug)
    self._sample = (config.inference_mode != 'MAP')

    raw_state = RawState()
    self._state = RoverState(
      raw_state, self._perception.create_state(),
      self._reactor.create_state(),
      self._proactor.create_state())

  def set_env(self, ground_truth, sample_location):
    self._state.raw.set_env(ground_truth, sample_location)

  def consume(self, metrics, frame, distribution=False):
    """
    :param metrics: the metrics data, dtype defined in rover_spec
    :param frame: the frame image
    :param distribution: return distribution of actions
    :return: action to take
    """
    self._state.raw.update(metrics, frame)
    self._perception.update_state(self._state)
    reaction = self._reactor.evaluate(self._state)
    action = self._proactor.evaluate(self._state, reaction)
    assert isinstance(action, ActionDistribution)

    # increment global step for debug
    self._step.increment()
    
    if distribution:
      return action

    # sample from action distribution
    if self._sample:
      return action.sample()
    # perform MAP inference
    else:
      ret = action.maximum()
      return ret

  @property
  def world_map(self):
    return self._state.raw.world_map_raw()

  @property
  def local_vision(self):
    return self._state.raw.vision_raw()

  @property
  def perception(self):
    return self._perception

  @property
  def proactor(self):
    return self._proactor

  @property
  def reactor(self):
    return self._reactor
