import abc
import importlib
from typing import Tuple
from collections import defaultdict
from collections import namedtuple
from rover.state import RoverState


class RoverModule(metaclass=abc.ABCMeta):
  """
   all policy and perception module will subclass this class
   to keep track of global step, write debug information in debug mode
   and perform initialization
  """

  def __init__(self, state: RoverState, opt, debug):
    self._debug_mode = debug
    self._records = defaultdict(lambda: ([], []))
    self.init(opt)
    self.init_ctx(state)

  def debug(self, key, value, state):
    if self._debug_mode:
      info = self._records[key]
      info[0].append(state.step)
      info[1].append(value)

  @abc.abstractmethod
  def init(self, opt):
    pass

  @abc.abstractmethod
  def init_ctx(self, state: RoverState):
    pass


class ModuleRegistry(object):

  def __init__(self, base_module: str):
    self._base_module_name = base_module
    self._base_module = tuple(base_module.split('.'))
    self._ctrs = {}

  def create_instance(self, cname, opt, state, debug):

    if cname in self._ctrs:
      ctr = self._ctrs[cname]
      ret = ctr(state, opt, debug)
      return ret

    name = cname.split('.')
    mod_name = self._base_module + tuple(name[:-1])
    mod_name = '.'.join(mod_name)
    importlib.import_module(mod_name)
    assert cname in self._ctrs
    ctr = self._ctrs[cname]
    ret = ctr(state, opt, debug)
    return ret

  def register_cls(self, cls):
    mod = cls.__module__
    assert mod.startswith(self._base_module_name + '.')
    key = '.'.join((mod, cls.__name__))
    key = key[len(self._base_module_name) + 1:]
    self._ctrs[key] = cls


Action = namedtuple(
  'Action', ('throttle', 'brake', 'steer', 'pick_up'))

Coordinates = namedtuple('Coordinates', ('x', 'y'))


class MapUpdate(object):

  def __init__(
      self, navi: Coordinates,
      obstacle: Coordinates, rock: Coordinates):

    self.navi = navi
    self.obstacle = obstacle
    self.rock = rock


class ExceptState(object):

  def __init__(self):
    pass


class TrajectoryPlan(object):

  def __init__(self, xs, ys):
    self.xs, self.ys = xs, ys
    self.progress = 0

  def set_progress(self, p):
    self.progress = p


class Planner(RoverModule):

  @abc.abstractmethod
  def plan(self, state, last_target,
           last_plan, except_state=None) -> TrajectoryPlan:
    pass

  @abc.abstractmethod
  def need_plan(self, state, target):
    pass


class Perception(RoverModule):

  @abc.abstractmethod
  def update(self, state) -> Tuple[MapUpdate, ExceptState]:
    pass


class Controller(RoverModule):

  @abc.abstractmethod
  def act(self, state: RoverState,
          plan: TrajectoryPlan) -> Action:
    pass
