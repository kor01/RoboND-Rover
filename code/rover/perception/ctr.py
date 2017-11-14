from rover.interface import ModuleRegistry
from rover.interface import Perception


_PERCEPTION_CTRS = ModuleRegistry(
  'rover.perception')


def register_perception(cls):
  assert issubclass(cls, Perception)
  _PERCEPTION_CTRS.register_cls(cls)
  return cls


def create_perception(name, config, state, debug):
  return _PERCEPTION_CTRS.create_instance(
    name, config, state, debug)
