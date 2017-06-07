import abc
import importlib
from collections import namedtuple
from ._global_step import GlobalStep


class AgentModule(metaclass=abc.ABCMeta):
  """
   all policy and perception module will subclass this class
   to keep track of global step, write debug information in debug mode
   and perform initialization
  """

  def __init__(self, global_step: GlobalStep,
               config, debug=False):
    self._debug = debug
    self._debug_info = []
    self._step = global_step
    self.initialize(config)

  def write_debug_info(self, key, value):
    if self._debug:
      self._debug_info.append((self._step.step, key, value))

  @property
  def debug_info(self):
    return self._debug_info

  @abc.abstractmethod
  def initialize(self, config):
    pass

  def clear(self):
    self._debug_info = []


RoverModuleSpec = namedtuple('RoverModuleSpec', ('name', 'config'))


class ModuleRegistry(object):

  def __init__(self, base_module: str):
    self._base_module_name = base_module
    self._base_module = tuple(base_module.split('.'))
    self._ctrs = {}

  def create_instance(self, name, config, step, debug):

    if name in self._ctrs:
      ctr = self._ctrs[name]
      ret = ctr(step, config, debug)
      return ret

    name = name.split('.')
    mod_name = self._base_module + tuple(name[:-1])
    mod_name = '.'.join(mod_name)
    importlib.import_module(mod_name)
    assert name in self._ctrs
    ctr = self._ctrs[name]
    ret = ctr(step, config, debug)
    return ret

  def register_cls(self, cls):
    mod = cls.__module__
    assert mod.startswith(self._base_module_name + '.')
    key = '.'.join((mod, cls.__name__))
    key = key[len(self._base_module_name) + 1:]
    self._ctrs[key] = cls
