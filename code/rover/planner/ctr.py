from rover.interface import ModuleRegistry
from rover.interface import Planner


_PLANNER_CTRS = ModuleRegistry('rover.planner')


def register_planner(cls):
  assert issubclass(cls, Planner)
  _PLANNER_CTRS.register_cls(cls)
  return cls


def create_planner(name, opt, state, debug):
  return _PLANNER_CTRS.create_instance(
    name, opt, state, debug)
