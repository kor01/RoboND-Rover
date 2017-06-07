import sys
import os
import socketio
import base64
import atexit

import matplotlib.image as mpimg
import numpy as np

import eventlet.wsgi
from flask import Flask
from io import BytesIO
from PIL import Image

from env_physics import EnvPhysics
from rover_agent import RoverAgent

from rover_resource import GROUND_TRUTH_MAP
from rover_resource import METRIC_DTYPE

from rover_replay import RoverReplay
from rover_config import RoverConfig

ROVER_AGENT = None
ROVER_REPLAY = None

# under truth map is not visible to agent
WITH_SAMPLE_LOCATION = True
# rock sample location is visible to agent
WITH_MAP = False

MODULE_PATH = os.path.dirname(__file__)

ENV_PHYSIC = EnvPhysics(mpimg.imread(GROUND_TRUTH_MAP))

FLASK_APP = Flask(__name__)

SIO = socketio.Server()

IMAGE_FOLDER = None


def parse_message(msg):
  metrics = np.zeros(tuple(), dtype=METRIC_DTYPE)
  metrics['Speed'] = np.float(msg["speed"])
  pos = np.fromstring(msg["position"], dtype=float, sep=',')
  metrics['X_Position'] = pos[0]
  metrics['Y_Position'] = pos[1]
  metrics['Yaw'] = np.float(msg["yaw"])
  metrics['Pitch'] = np.float(msg["pitch"])
  metrics['Roll'] = np.float(msg["roll"])
  metrics['Throttle'] = np.float(msg["throttle"])
  metrics['SteerAngle'] = np.float(msg["steering_angle"])
  metrics['NearSample'] = np.int(msg["near_sample"])
  metrics['PickingUp'] = np.int(msg['picking_up'])
  frame = np.asarray(Image.open(
    BytesIO(base64.b64decode(msg["image"]))))
  return metrics, frame


@SIO.on('telemetry')
def telemetry(sid, msg):
  global ROVER_AGENT, ROVER_REPLAY
  assert isinstance(ROVER_AGENT, RoverAgent)
  if msg:
    metrics, frame = parse_message(msg)

    if not ENV_PHYSIC.started:
      # control visibility of the game
      ENV_PHYSIC.consume(msg)
      ground_truth = None if WITH_SAMPLE_LOCATION \
        else ENV_PHYSIC.ground_truth
      sample_location = ENV_PHYSIC.samples_pos \
        if WITH_MAP else None
      ROVER_AGENT.set_env(ground_truth, sample_location)
      ENV_PHYSIC.ckpt(os.path.join(EXPERIMENT_DIR, 'environment'))
    else:
      ENV_PHYSIC.consume(msg)

    action = ROVER_AGENT.consume(metrics, frame)
    print(action)

    # save replay
    if isinstance(ROVER_REPLAY, RoverReplay):
      ROVER_REPLAY.consume(frame, metrics, ENV_PHYSIC.current_time)

    # flush actions and debug information to environment
    commands = (action.throttle, action.brake, action.steer)
    pictures = ENV_PHYSIC.create_output_images(
      ROVER_AGENT.world_map, ROVER_AGENT.local_vision)
    send_control(commands, *pictures)

    if action.pick_up:
      send_pickup()

  else:
    print('switch to manual control')
    SIO.emit('manual', data={}, skip_sid=True)
    if isinstance(ROVER_REPLAY, RoverReplay):
      ROVER_REPLAY.flush()


@SIO.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control((0, 0, 0), '', '')
    sample_data = {}
    SIO.emit(
        "get_samples",
        sample_data,
        skip_sid=True)


def send_control(commands, image_string1, image_string2):
    # Define commands to be sent to the rover
    print("sending control", len(image_string1), len(image_string2))
    data={
        'throttle': commands[0].__str__(),
        'brake': commands[1].__str__(),
        'steering_angle': commands[2].__str__(),
        'inset_image1': image_string1,
        'inset_image2': image_string2,
        }
    # Send commands via socketIO server
    SIO.emit("data", data, skip_sid=True)


# Define a function to send the "pickup" command 
def send_pickup():
    print("Picking up")
    pickup = {}
    SIO.emit("pickup", pickup, skip_sid=True)


def exit_handle():
  global ENV_PHYSIC
  ROVER_REPLAY.flush()

atexit.register(exit_handle)

if __name__ == '__main__':

  assert len(sys.argv) >= 3, \
    'usage: python drive_rover.py experiment_dir save_replay=[True or False]'

  EXPERIMENT_DIR = sys.argv[1]

  if sys.argv[2].lower() == 'true':
    ROVER_REPLAY = RoverReplay(
      os.path.join(EXPERIMENT_DIR, 'replay'))

  config = os.path.join(EXPERIMENT_DIR, 'agent_spec.cfg')
  config = RoverConfig(config_path=config)

  WITH_SAMPLE_LOCATION, WITH_MAP = config.with_sample_location, config.with_map

  ROVER_AGENT = RoverAgent(config, debug=False)

  FLASK_APP = socketio.Middleware(SIO, FLASK_APP)

  eventlet.wsgi.server(eventlet.listen(('', 4567)), FLASK_APP)
