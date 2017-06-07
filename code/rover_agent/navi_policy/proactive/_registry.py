from rover_agent.navi_policy import ProactivePolicy
from rover_agent.agent_module import ModuleRegistry

_PROACTIVE_POLICY_CTRS = ModuleRegistry(
  'rover_agent.navi_policy.proactive')

def register_proactor(cls):
  assert issubclass(cls, ProactivePolicy)
  _PROACTIVE_POLICY_CTRS.register_cls(cls)
  return cls


def create_proactor(spec, step, debug):
  return _PROACTIVE_POLICY_CTRS.create_instance(spec, step, debug)
