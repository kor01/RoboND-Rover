from rover.navi_policy import ProactivePolicy
from rover.interface import ModuleRegistry

_PROACTIVE_POLICY_CTRS = ModuleRegistry(
  'rover.navi_policy.proactive')

def register_proactor(cls):
  assert issubclass(cls, ProactivePolicy)
  _PROACTIVE_POLICY_CTRS.register_cls(cls)
  return cls


def create_proactor(name, config, step, debug):
  return _PROACTIVE_POLICY_CTRS.create_instance(
    name, config, step, debug)
