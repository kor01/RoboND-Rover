import os
import configparser


class RoverConfig(object):

  def __init__(self, config_path=None, config=None):

    self.path = None

    if config_path is not None:
      config = configparser.ConfigParser()
      config.read(config_path)

    self.config = config

    if 'Path' in config['AGENT']:
      self.path = config['AGENT']['Path']
    else:
      self.path = os.path.dirname(os.path.abspath(config_path))

    self.perception = config['AGENT']['Perception']
    self.planner = config['AGENT']['Planner']
    self.controller = config['AGENT']['Controller']

    if 'GAME' not in config:
      self.with_map = False
      self.with_sample_location = True
    else:
      self.with_map = config.getboolean('GAME', 'WithMap')
      self.with_sample_location = config.getboolean(
        'GAME', 'WithSampleLocation')

  def override_config(self, configs):
    for k, kvs in configs.items():
      for sk, sv in kvs.items():
        self.config[k][sk] = sv

  @property
  def perception_opt(self):
    return self.config['PERCEPTION']

  @property
  def planner_opt(self):
    return self.config['PLANNER']

  @property
  def controller_opt(self):
    return self.config['CONTROLLER']
