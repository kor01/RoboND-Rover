import os
import matplotlib.image as mpimg
from collections import defaultdict
from collections import namedtuple
from rover_agent import RoverAgent
from rover_agent import AgentModule
from rover_config import RoverConfig
from env_physics import EnvPhysics
from rover_resource import GROUND_TRUTH_MAP

from ._load_dataset import load_replay_from_csv
from ._load_dataset import load_replay


FrameExperience = namedtuple(
  'FrameExperience', ('frame', 'metric', 'debug',
                      'action', 'fidelity',
                      'coverage', 'map', 'vision'))


def filter_frames(frames, key, value):
  ret = []
  for frame in frames:
    assert key in frame.debug
    if frame.debug[key] == value:
      ret.append(frame)
  return ret


def get_debug_values(frames, key):
  ret = []
  for frame in frames:
    assert key in frame.debug
    ret.append(frame.debug[key])
  return ret


def _create_index(mod: AgentModule):
  ret = defaultdict(dict)
  info = mod.debug_info
  for t, k, v in info:
    ret[t][k] = v
  return dict(ret)


class DebugInfo(object):

  def __init__(self, agent: RoverAgent):

    self._perception = \
      _create_index(agent.perception)
    self._reactor = \
      _create_index(agent.reactor)
    self._proactor = \
      _create_index(agent.proactor)

  def at(self, i):
    ret = {}
    if i in self._perception:
      ret.update(self._perception[i])

    if i in self._reactor:
      ret.update(self._reactor[i])

    if i in self._proactor:
      ret.update(self._proactor[i])
    return ret


class ExperienceRelay(object):
  """
  analysis replay from previous experiment
  """

  def __init__(self, path=None, config=None, debug=True, experience=None):
    """
    :param path: path to a experiment directory  
    """

    if path is not None:
      config = RoverConfig(config_path=path)
    assert config is not None, 'config not set'

    self._config, self._debug = config, debug

    if experience is None:
      self._load_experience()
    else:
      self._experience = experience

    self._env = EnvPhysics(mpimg.imread(GROUND_TRUTH_MAP))
    self._env.load(os.path.join(path, 'environment'))
    self._setup_agent()

  def _load_experience(self):
    path = self._config.path
    if self._config.replay_format == 'csv':
      csv = os.path.join(path, 'replay/robot_log.csv')
      if self._config.replay_relative is not None:
        frame_dir = os.path.dirname(os.path.abspath(__file__))
        frame_dir = os.path.abspath(frame_dir + '/../../records')
        frame_dir = os.path.join(frame_dir, self._config.replay_relative)
      else:
        frame_dir = None
      self._experience = load_replay_from_csv(csv, frame_dir)
    else:
      self._experience = load_replay(os.path.join(path, 'replay/'))

  def _setup_agent(self):
    config = self._config
    self._agent = RoverAgent(
      config=self._config, debug=self._debug)

    with_map, with_sample_location = \
      config.with_map, config.with_sample_location

    true_map = self._env.ground_truth if with_map else None
    sample_pos = self._env.samples_pos if with_sample_location else None
    self._agent.set_env(true_map, sample_pos)

    self._actions = None
    self._views = None
    self._debug_info = None

  def replay(self, distribution=False):
    self._actions = []
    self._views = []
    for frame, metrics, tm in self._experience:

      self._env.consume(None, tm)
      actions = self._agent.consume(
        metrics, frame, distribution=distribution)
      self._actions.append(actions)
      ret = self._env.create_output_images(
        self._agent.world_map,
        self._agent.local_vision, serialize=False)

      self._views.append(ret)
    self._debug_info = DebugInfo(self.agent)

  @property
  def agent(self):
    return self._agent

  @property
  def actions(self):
    return self._actions

  @property
  def views(self):
    return self._views

  @property
  def experience(self):
    return self._experience

  @property
  def debug_info(self):
    return self._debug_info

  def at(self, i):
    exp = self.experience.at(i)
    debug = self._debug_info.at(i)
    action = self._actions[i]
    fidelity, coverage, mp, vision = self._views[i]
    ret = FrameExperience(
      frame=exp[0], metric=exp[1],
      debug=debug, action=action,
      fidelity=fidelity, coverage=coverage,
      map=mp, vision=vision)
    return ret

  @property
  def frames(self):
    ret = []
    for i in range(len(self)):
      ret.append(self.at(i))
    return ret

  def __len__(self):
    return len(self._experience)

  def reset_config(self, config):
    self._config = RoverConfig(config=config)
    self._setup_agent()

  @property
  def experience(self):
    return self._experience
