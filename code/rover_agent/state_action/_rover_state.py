from ._raw_state import RawState

class RoverState(object):

  def __init__(self, raw: RawState, perception, reactor, proactor):
    """
    :param raw: RawState from telemetry 
    :param perception: perception state
    :param reactor: reactor policy state
    :param proactor: proactor policy state
    """

    self._raw = raw
    self._perception = perception
    self._reactor = reactor
    self._proactor = proactor

  @property
  def raw(self):
      return self._raw

  @property
  def perception(self):
      return self._perception

  @property
  def reactor(self):
      return self._reactor

  @property
  def proactor(self):
      return self._proactor
