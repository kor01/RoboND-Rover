import numpy as np
from rover.interface import Planner
from rover.state import RoverState
from rover.planner.ctr import register_planner


def identify_search_point(world_map):




@register_planner
class MaxAnglePlanner(Planner):

  def __init__(self, state: RoverState, opt, debug):
    super().__init__(state, opt, debug)

  def init_ctx(self, state: RoverState):
    pass

  def init(self, opt):
    pass

  def need_plan(self, state, target):
    return True

  def plan(self, state, last_target,
           last_plan, except_state=None):

    return super().plan(state, last_target,
                        last_plan, except_state)
