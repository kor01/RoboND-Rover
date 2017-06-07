import numpy as np
from collections import namedtuple


Action = namedtuple('Action', ('throttle', 'brake', 'steer', 'pick_up'))

BRAKE_MIN = 0.0
BRAKE_MAX = 1.0
BRAKE_STEP = 0.1

STEER_MIN = -15.0
STEER_MAX = 15.0
STEER_STEP = 0.1

THROTTLE_MIN = -1.0
THROTTLE_MAX = 1.0
THROTTLE_STEP = 0.1

# noinspection PyTypeChecker
THROTTLE_SHAPE = (np.around((THROTTLE_MAX - THROTTLE_MIN)
                            / THROTTLE_STEP).astype(np.uint32),)
# noinspection PyTypeChecker
STEER_SHAPE = (np.around((STEER_MAX - STEER_MIN)
                         / STEER_STEP).astype(np.uint32),)
# noinspection PyTypeChecker
BRAKE_SHAPE = (np.around((BRAKE_MAX - BRAKE_MIN)
                         / BRAKE_STEP).astype(np.uint32),)

# noinspection PyTypeChecker
ZERO_BRAKE = np.around((-BRAKE_MIN)
                       / BRAKE_STEP).astype(np.uint32)
# noinspection PyTypeChecker
ZERO_THROTTLE = np.around((-THROTTLE_MIN)
                          / THROTTLE_STEP).astype(np.uint32)
# noinspection PyTypeChecker
ZERO_STEER = np.around((-STEER_MIN)
                       / STEER_STEP).astype(np.uint32)


def continuous_steer_to_discrete(steer, rad_unit=True):

  steer = (steer * 180) / np.pi if rad_unit else steer

  if steer < STEER_MIN:
    steer = STEER_MIN

  if steer > STEER_MAX:
    steer = STEER_MAX

  # noinspection PyTypeChecker
  steer = np.around((steer - STEER_MIN) / STEER_STEP).astype('uint32')
  assert steer <= STEER_SHAPE[0]

  if steer == STEER_SHAPE[0]:
    steer = STEER_SHAPE[0] - 1

  return steer


def discrete_action_to_continuous(action: Action) -> Action:
  if action.throttle is not None:
    throttle = action.throttle * THROTTLE_STEP + THROTTLE_MIN
  else:
    throttle = 0

  if action.brake is not None:
    brake = action.brake * BRAKE_STEP + BRAKE_MIN
  else:
    brake = 0

  steer = action.steer * STEER_STEP + STEER_MIN
  return Action(throttle=throttle, brake=brake, steer=steer, pick_up=action.pick_up)


class ActionDistribution(object):
  """
  stochastic action discretized to ease policy inference
  """

  def __init__(self, switch: np.ndarray,
               throttle: np.ndarray, brake: np.ndarray,
               steer: np.ndarray):

    assert throttle.shape == THROTTLE_SHAPE
    assert brake.shape == BRAKE_SHAPE
    assert steer.shape == STEER_SHAPE
    assert switch.shape == (2,)

    self._throttle = throttle
    self._brake = brake
    self._steer = steer
    self._switch = switch

  @property
  def throttle(self):
    return self._throttle.copy()

  @property
  def brake(self):
      return self._brake.copy()

  @property
  def steer(self):
      return self._steer.copy()

  @property
  def switch(self):
    return self._switch.copy()

  def sample(self) -> Action:
    # sample switch
    angle = np.random.choice(
      len(self._steer), self._steer)

    switch = np.random.choice(2, self._switch)

    # use throttle
    if switch:
      throttle = np.random.choice(
        len(self._throttle), self._throttle)
      return Action(throttle=throttle, brake=None, steer=angle)

    else:
      brake = np.random.choice(len(self._brake), self._brake)

      return discrete_action_to_continuous(
        Action(throttle=None, brake=brake, steer=angle))

  def maximum(self) -> Action:

    angle = np.argmax(self._steer)

    use_throttle = (np.argmax(self._switch) == 1)

    if use_throttle:
      throttle = np.argmax(self._throttle)
      return discrete_action_to_continuous(
        Action(throttle=throttle, brake=None,
               steer=angle, pick_up=False))
    else:
      brake = np.argmax(self._brake)
      return discrete_action_to_continuous(
        Action(throttle=None, brake=brake,
               steer=angle, pick_up=False))
