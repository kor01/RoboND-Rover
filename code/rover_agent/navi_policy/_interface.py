import abc

from rover_agent.agent_module import AgentModule
from rover_agent.state_action import RoverState
from rover_agent.state_action import ActionDistribution


class ReactivePolicy(AgentModule):
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


class ProactivePolicy(AgentModule):
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
