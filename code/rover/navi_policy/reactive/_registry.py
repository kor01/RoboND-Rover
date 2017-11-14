from rover.navi_policy import ReactivePolicy
from rover.interface import ModuleRegistry

_REACTIVE_POLICY_CTRS = ModuleRegistry(
  'rover.navi_policy.reactive')

def register_reactor(cls):
  assert issubclass(cls, ReactivePolicy)
  _REACTIVE_POLICY_CTRS.register_cls(cls)
  return cls


def create_reactor(name, config, step, debug):
  return _REACTIVE_POLICY_CTRS.create_instance(
    name, config, step, debug)
