from rover.interface import ModuleRegistry
from rover.interface import Controller


_CONTROLLER_CTRS = ModuleRegistry('rover.controller')


def register_controller(cls):
  assert issubclass(cls, Controller)
  _CONTROLLER_CTRS.register_cls(cls)


def create_controller(name, opt, state, debug):
  return _CONTROLLER_CTRS.create_instance(
    name, opt, state, debug)

