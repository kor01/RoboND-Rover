import numpy as np
import matplotlib.pyplot as plt
from rover_model.geometry import circle_distance


class PerspectiveRender(object):

  def __init__(self, scale, resolution):

    self._scale = scale
    self._resolution = resolution

  def render(self, particles, inplace=False):
    particles = particles.transpose()
    particles = particles * self._scale
    particles[:, 1] *= -1
    particles[:, 1] += self._resolution / 2
    particles = np.around(particles).astype('uint32')
    particles[:, 0] = self._resolution - particles[:, 0]
    particles = particles.clip(0, self._resolution - 1)
    render = np.zeros([self._resolution, self._resolution], dtype=np.uint8)
    render[particles[:, 0], particles[:, 1]] += 1
    if inplace:
      plt.imshow(render)
      plt.show()
    return render


# noinspection PyTypeChecker
def highlight_position(mp: np.ndarray, x, y):
    mp = np.asarray(mp).copy()
    x = int(np.around(x))
    y = int(np.around(y))
    y_max = mp.shape[0]
    x_max = mp.shape[1]
    xs = [max(x-1, 0), x, min(x+1, y_max - 1)]
    ys = [max(y-1, 0), y, min(y+1, x_max - 1)]
    for x in xs:
        for y in ys:
            mp[y_max - y, x, 0] = 128
    return mp


def pitch_roll_deviation(frames):

  pitches = np.array([circle_distance(x.metric['Pitch'], 0)
                      for x in frames])
  rolls = np.array([circle_distance(x.metric['Roll'], 0)
                    for x in frames])
  return pitches.mean(), rolls.mean()


def analysis_case(frame, analysis, render):

  print('roll and pitch deviation:',
        circle_distance(frame.metric['Roll'], 0),
        circle_distance(frame.metric['Pitch'], 0))

  print('yaw =', frame.metric['Yaw'], ', pitch = ',
        frame.metric['Pitch'], ', roll = ', frame.metric['Roll'])

  print('rover position:', frame.metric['X_Position'], frame.metric['Y_Position'])
  print('map gradient: fidelity =', analysis.ratio,
        'pos =', analysis.pos, 'neg =', analysis.neg)
  mp = highlight_position(frame.map, frame.metric['X_Position'], frame.metric['Y_Position'])

  top_view = render.render(frame.debug['topdown'])

  plt.figure(figsize=(80, 160))
  plt.subplot(141)
  plt.imshow(analysis.render)
  plt.subplot(142)
  plt.imshow(frame.frame)
  plt.subplot(143)
  plt.imshow(mp)
  plt.subplot(144)
  plt.imshow(top_view)
  plt.show()

