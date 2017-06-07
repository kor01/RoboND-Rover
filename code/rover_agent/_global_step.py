

class GlobalStep(object):

  def __init__(self):
    self._step = 0

  def increment(self):
    self._step += 1

  @property
  def step(self):
    return self._step
