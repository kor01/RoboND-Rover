import abc
from rover.interface import RoverModule
from rover.interface import TrajectoryPlan


class Planner(RoverModule):

  @abc.abstractmethod
  def plan(self, state, last_target,
           last_plan, except_state=None) -> TrajectoryPlan:
    pass

