import os
import numpy as np
from replay_analysis import load_replay_from_csv
from rover_model.geometry import degree_to_rad
from rover import state_action as sa


def discretize_action(low, step, dim, data):

  data = data - low
  data = data / step
  data = np.clip(np.around(data), 0, dim - 1)
  data = data.astype('int32')
  return data


def save_tensor(arr: np.ndarray, path):
  assert arr.flags.c_contiguous
  with open(path, 'wb') as op:
    op.write(arr.data)


def process_records(input_path, output_path):

  img_dir = os.path.dirname(input_path)
  experience = load_replay_from_csv(input_path, img_dir)

  frames_bin = os.path.join(
    output_path, 'frames.uint8.bin')
  save_tensor(experience.frames, frames_bin)

  pitch = degree_to_rad(experience.metrics['Pitch'])
  roll = degree_to_rad(experience.metrics['Roll'])

  pc, rc = np.cos((pitch, roll))
  ps, rs = np.sin((pitch, roll))

  speed = experience.metrics['Speed']
  features = np.array(
    [pc, ps, rc, rs, speed], dtype=np.float32)
  features_copy = features.transpose().copy()
  feature_path = os.path.join(output_path, 'features.float32.bin')
  save_tensor(features_copy, feature_path)

  steer = discretize_action(
    sa.STEER_MIN, sa.STEER_STEP, sa.STEER_DIM,
    experience.metrics['SteerAngle'])

  throttle = discretize_action(
    sa.THROTTLE_MIN, sa.THROTTLE_STEP,
    sa.THROTTLE_DIM, experience.metrics['Throttle'])

  brake = discretize_action(
    sa.BRAKE_MIN, sa.BRAKE_STEP,
    sa.BRAKE_DIM, experience.metrics['Brake'])

  switch = np.greater(brake, 0).astype('int32')
  labels = np.array([throttle, brake, steer, switch])
  label_path = os.path.join(output_path, 'labels.int32.bin')
  save_tensor(labels, label_path)

  true_steer = degree_to_rad(experience.metrics['SteerAngle'])
  true_steer = np.array(
    [np.cos(true_steer), np.sin(true_steer)], dtype=np.float32)
  true_steer = true_steer.transpose().copy()
  true_steer_path = os.path.join(output_path, 'true_steer.float32.bin')
  save_tensor(true_steer, true_steer_path)
