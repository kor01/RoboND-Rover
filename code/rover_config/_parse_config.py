import configparser


class RoverConfig(object):

  def __init__(self, config_path=None, config=None):

    self._path = None

    if config_path is not None:
      config = configparser.ConfigParser()
      config.read(config_path)

    self._config = config

    self._path = config['AGENT']['Path']
    self._perception = config['AGENT']['Perception']
    self._reactor = config['AGENT']['Reactor']
    self._proactor = config['AGENT']['Proactor']
    self._inference_mode = config['AGENT']['Mode']

    self._with_map = config.getboolean('GAME', 'WithMap')
    self._with_sample_location = config.getboolean(
      'GAME', 'WithSampleLocation')

    self._replay_format = config.get('REPLAY', 'FORMAT')

    if 'RELATIVE_PATH' in config['REPLAY']:
      self._replay_relative = config['REPLAY']['RELATIVE_PATH']
    else:
      self._replay_relative = None

  @property
  def path(self):
    return self._path

  @property
  def perception(self):
      return self._perception

  @property
  def reactor(self):
      return self._reactor

  @property
  def proactor(self):
      return self._proactor

  @property
  def inference_mode(self):
      return self._inference_mode

  @property
  def replay_format(self):
    return self._replay_format

  @property
  def replay_relative(self):
    return self._replay_relative

  @property
  def with_map(self):
      return self._with_map

  @property
  def with_sample_location(self):
      return self._with_sample_location

  def set_config(self, section, key, value):
    if section not in self._config:
      self._config.add_section(section)
    self._config[section][key] = value

  def internal(self):
    return self._config

  def __getitem__(self, item):
    return self._config.__getitem__(item)
