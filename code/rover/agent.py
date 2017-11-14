from rover.config import RoverConfig

from rover.perception.ctr import create_perception
from rover.planner.ctr import create_planner
from rover.controller.ctr import create_controller

from rover.state import RoverState

from rover.interface import MapUpdate
from rover.interface import ExceptState


class RoverAgent(object):

  def __init__(self, config: RoverConfig, debug=True):
    """
    :param config: a rover configuration object
    :param debug: open debug mode
    """
    self.step = 0

    self.state = RoverState()
    self.target, self.plan = None, None

    self.perception = create_perception(
      config.perception, config.perception_opt, self.state, debug)

    self.planner = create_planner(
      config.planner, config.planner_opt, self.state, debug)

    self.controller = create_controller(
      config.controller, config.controller_opt, self.state, debug)

    self.finished = False

  def consume(self, metrics, frame):
    """
    :param metrics: the metrics data, dtype defined in rover_spec
    :param frame: the frame image
    :return: action to take
    """

    self.state.update(metrics, frame)

    update, except_state = \
      self.perception.update(self.state)

    assert isinstance(update, MapUpdate)

    self.state.update_navigable_map(
      y=update.navi.y, x=update.navi.x)
    self.state.update_obstacle_map(
      x=update.obstacle.x, y=update.obstacle.y)
    self.state.update_rock_sample_map(
      x=update.rock.x, y=update.rock.y)

    if except_state is not None:
      assert isinstance(except_state, ExceptState)
      self.target, self.plan = self.planner.plan(
        self.state, self.target, self.plan, except_state)

    elif self.planner.need_plan(self.target, self.state):
        self.target, self.plan = self.planner.plan(
          self.state, self.target, self.plan)

    if self.target is None:
      self.finished = True

    action = self.controller.act(self.state, self.plan)
    self.step += 1

    return action

  @property
  def world_map(self):
    return self.state.world_map_raw()

  @property
  def local_vision(self):
    return self.state.vision_raw()

