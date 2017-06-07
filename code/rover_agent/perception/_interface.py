import abc
from rover_agent.agent_module import AgentModule
from rover_agent.state_action import RoverState


class Perception(AgentModule):

  @abc.abstractmethod
  def create_state(self):
    pass

  @abc.abstractmethod
  def update_state(self, state: RoverState):
    pass

