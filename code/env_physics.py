import os
import time
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image


class EnvPhysics(object):

  def __init__(self, ground_truth):

    self._started = False
    self._start_time = None
    self._current_time = None
    self._ground_truth = np.dstack(
      (ground_truth * 0, ground_truth * 255,
       ground_truth * 0)).astype(np.float)

    self._samples_pos = None
    self._samples_found = None

  @property
  def started(self):
    return self._started

  def ckpt(self, path):
    """
    save a checkpoint for sample positions and start time for replay analysis
    :param path: path to save 
    :return: None
    """
    np.save(os.path.join(path, 'samples_pos.npy'), self.samples_pos)
    np.save(os.path.join(path, 'start_time.npy'),
            np.array([self._start_time], dtype=np.float64))

  def load(self, path):
    self._samples_pos = np.load(os.path.join(path, 'samples_pos.npy'))
    self._start_time = np.load(os.path.join(path, 'start_time.npy'))[0]
    self._samples_found = np.zeros((len(self._samples_pos[0]))).astype(np.int)
    self._started = True

  def _start(self, msg):

    self._start_time = time.time()
    samples_xpos = np.array(
      [np.float(pos.strip()) for pos in msg["samples_x"].split(',')],
      dtype=np.uint32)
    samples_ypos = np.array(
      [np.float(pos.strip()) for pos in msg["samples_y"].split(',')],
      dtype=np.uint32)

    self._samples_pos = (samples_xpos, samples_ypos)
    self._samples_found = np.zeros((len(self._samples_pos[0]))).astype(np.int)
    self._started = True

  def consume(self, msg, current_time=None):
    if not self._started:
      self._start(msg)
    self._current_time = current_time if current_time else time.time()

  @property
  def total_time(self):
    return self._current_time - self._start_time

  @property
  def current_time(self):
    return self._current_time

  @property
  def ground_truth(self):
    return self._ground_truth

  @property
  def samples_found(self):
    return self._samples_found

  @property
  def samples_pos(self):
    return self._samples_pos

  def create_output_images(
      self, worldmap, vision_image, serialize=True):

    # Create a scaled map for plotting and clean up obs/nav pixels a bit
    if np.max(worldmap[:, :, 2]) > 0:
      nav_pix = worldmap[:, :, 2] > 0
      navigable = worldmap[:, :, 2] * (255 / np.mean(worldmap[nav_pix, 2]))
    else:
      navigable = worldmap[:, :, 2]
    if np.max(worldmap[:, :, 0]) > 0:
      obs_pix = worldmap[:, :, 0] > 0
      obstacle = worldmap[:, :, 0] * (255 / np.mean(worldmap[obs_pix, 0]))
    else:
      obstacle = worldmap[:, :, 0]

    likely_nav = navigable >= obstacle
    obstacle[likely_nav] = 0
    plotmap = np.zeros_like(worldmap)
    plotmap[:, :, 0] = obstacle
    plotmap[:, :, 2] = navigable
    plotmap = plotmap.clip(0, 255)
    # Overlay obstacle and navigable terrain map with ground truth map
    map_add = cv2.addWeighted(plotmap, 1, self.ground_truth, 0.5, 0)

    # Check whether any rock detections are present in worldmap
    rock_world_pos = worldmap[:, :, 1].nonzero()
    # If there are, we'll step through the known sample positions
    # to confirm whether detections are real
    if rock_world_pos[0].any():
      rock_size = 2
      for idx in range(len(self.samples_pos[0]) - 1):
        test_rock_x = self.samples_pos[0][idx]
        test_rock_y = self.samples_pos[1][idx]
        rock_sample_dists = np.sqrt((test_rock_x - rock_world_pos[1]) ** 2 + \
                                    (test_rock_y - rock_world_pos[0]) ** 2)
        # If rocks were detected within 3 meters of known sample positions
        # consider it a success and plot the location of the known
        # sample on the map
        if np.min(rock_sample_dists) < 3:
          self.samples_found[idx] = 1
          map_add[test_rock_y - rock_size:test_rock_y + rock_size,
          test_rock_x - rock_size:test_rock_x + rock_size, :] = 255

    # Calculate some statistics on the map results
    # First get the total number of pixels in the navigable terrain map
    tot_nav_pix = np.float(len((plotmap[:, :, 2].nonzero()[0])))
    # Next figure out how many of those correspond to ground truth pixels
    # noinspection PyUnresolvedReferences
    good_nav_pix = np.float(len(((plotmap[:, :, 2] > 0)
                                 & (self.ground_truth[:, :, 1] > 0)).nonzero()[0]))
    # Next find how many do not correspond to ground truth pixels
    # noinspection PyUnresolvedReferences
    bad_nav_pix = np.float(len(((plotmap[:, :, 2] > 0)
                                & (self.ground_truth[:, :, 1] == 0)).nonzero()[0]))
    # Grab the total number of map pixels
    tot_map_pix = np.float(len((self.ground_truth[:, :, 1].nonzero()[0])))
    # Calculate the percentage of ground truth map that has been successfully found
    perc_mapped = round(100 * good_nav_pix / tot_map_pix, 1)
    # Calculate the number of good map pixel detections divided by total pixels
    # found to be navigable terrain
    if tot_nav_pix > 0:
      fidelity = round(100 * good_nav_pix / tot_nav_pix, 1)
    else:
      fidelity = 0
    # Flip the map for plotting so that the y-axis points upward in the display
    map_add = np.flipud(map_add).astype(np.float32)
    # Add some text about map and rock sample detection results
    cv2.putText(map_add, "Time: " + str(np.round(self.total_time, 1)) + ' s', (0, 10),
                cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(map_add, "Mapped: " + str(perc_mapped) + '%', (0, 25),
                cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(map_add, "Fidelity: " + str(fidelity) + '%', (0, 40),
                cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(map_add, "Rocks Found: " + str(np.sum(self.samples_found)), (0, 55),
                cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)

    # Convert map and vision image to base64 strings for sending to server
    pil_img = Image.fromarray(map_add.astype(np.uint8))
    # for replay analysis
    if not serialize:
      return fidelity, perc_mapped, pil_img, vision_image

    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    encoded_string1 = base64.b64encode(buff.getvalue()).decode("utf-8")

    pil_img = Image.fromarray(vision_image.astype(np.uint8))
    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    encoded_string2 = base64.b64encode(buff.getvalue()).decode("utf-8")

    return encoded_string1, encoded_string2
