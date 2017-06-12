from rover_agent.agent_module import ModuleRegistry
from ._interface import Perception


_PERCEPTION_CTRS = ModuleRegistry(
  'rover_agent.perception')


def register_perception(cls):
  assert issubclass(cls, Perception)
  _PERCEPTION_CTRS.register_cls(cls)
  return cls


def create_perception(name, config, step, debug):
  return _PERCEPTION_CTRS.create_instance(name, config, step, debug)
