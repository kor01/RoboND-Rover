import abc

from rover.interface import RoverModule
from rover.state_action import RoverState
from rover.state_action import ActionDistribution


class ReactivePolicy(RoverModule):
  """
  reactive policy:
   optimize local navigation safety, smoothness and efficiency 
  """

  @abc.abstractmethod
  def create_state(self):
    pass

  @abc.abstractmethod
  def evaluate(self, state: RoverState) -> ActionDistribution:
    pass


class ProactivePolicy(RoverModule):
  """
  proactive policy:
    optimize long term reward: rock pickup, 
      map coverage and fidelity, cruise time etc.
  """

  @abc.abstractmethod
  def create_state(self):
    pass

  @abc.abstractmethod
  def evaluate(self, state: RoverState,
               action: ActionDistribution) -> ActionDistribution:
    pass
