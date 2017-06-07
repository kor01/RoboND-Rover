from rover_agent.navi_policy import ReactivePolicy
from rover_agent.agent_module import ModuleRegistry

_REACTIVE_POLICY_CTRS = ModuleRegistry(
  'rover_agent.navi_policy.reactive')

def register_reactor(cls):
  assert issubclass(cls, ReactivePolicy)
  _REACTIVE_POLICY_CTRS.register_cls(cls)
  return cls


def create_reactor(spec, step, debug):
  return _REACTIVE_POLICY_CTRS.create_instance(spec, step, debug)
