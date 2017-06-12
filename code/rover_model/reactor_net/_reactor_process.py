import os
import numpy as np
from replay_analysis import load_replay_from_csv
from rover_model.geometry import degree_to_rad
from rover_agent import state_action as sa


def discretize_action(low, step, dim, data):

  data = data - low
  data = data / step
  data = np.clip(np.around(data), 0, dim - 1)
  data = data.astype('int32')
  return data


def process_records(input_path, output_path):

  experience = load_replay_from_csv(input_path)

  frame_path = os.path.join(output_path, 'frames.npy')
  np.save(frame_path, experience.frames)
  frames_bin = os.path.join(
    output_path, 'frames.uint8.bin')
  with open(frames_bin, 'wb') as op:
    op.write(experience.frames.data)

  pitch = degree_to_rad(experience.metrics['Pitch'])
  roll = degree_to_rad(experience.metrics['Roll'])

  pc, rc = np.cos((pitch, roll))
  ps, rs = np.sin((pitch, roll))

  speed = experience.metrics['Speed']
  features = np.array(
    [pc, ps, rc, rs, speed], dtype=np.float32)
  features = features.transpose()
  feature_path = os.path.join(output_path, 'features.npy')
  np.save(feature_path, features)

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

  label_path = os.path.join(output_path, 'labels.npy')
  np.save(label_path, labels)

