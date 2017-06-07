import numpy as np
from rover_resource import METRIC_DTYPE


class RawState(object):

  def __init__(self):

    self.img = None
    self.pos = None
    self.yaw = None
    self.pitch = None
    self.roll = None
    self.velocity = None
    self.steer = 0
    self.throttle = 0
    self.brake = 0
    self._vision_image = np.zeros((160, 320, 3), dtype=np.float)
    self._worldmap = np.zeros((200, 200, 3), dtype=np.float)
    self.samples_pos = None
    self.samples_found = 0
    self.near_sample = False
    self.picking_up = False

  def set_env(self, ground_truth, sample_location):
    self.samples_pos = sample_location
    self._ground_truth = ground_truth

  @property
  def ground_truth(self):
    self._ground_truth.flags.writeable = False
    return self._ground_truth

  def update(self, metrics: np.ndarray, frame: np.ndarray):
    assert metrics.dtype == METRIC_DTYPE
    self.velocity = metrics['Speed']
    self.pos = np.array([metrics['X_Position'], metrics['Y_Position']])
    self.yaw = metrics['Yaw']
    self.pitch = metrics['Pitch']
    self.roll = metrics['Roll']
    self.throttle = metrics['Throttle']
    self.steer = metrics['SteerAngle']
    self.brake = metrics['Brake']
    self.near_sample = metrics['NearSample']
    self.picking_up = metrics['PickingUp']
    self.img = frame

  def clear_map(self):
    self._worldmap[:] = 0

  def update_navigable_map(self, x, y):
    self._worldmap[y, x, 2] += 1

  def update_rock_sample_map(self, x, y):
    self._worldmap[y, x, 1] += 1

  def update_obstacle_map(self, x, y):
    self._worldmap[y, x, 0] += 1

  def update_navigable_vision(self, vision):
    self._vision_image[:, :, 2] = vision * 255

  def update_rock_sample_vision(self, vision):
    self._vision_image[:, :, 1] = vision

  def update_obstacle_vision(self, vision):
    self._vision_image[:, :, 0] = vision

  def world_map_raw(self):
    return self._worldmap

  def vision_raw(self):
    return self._vision_image
