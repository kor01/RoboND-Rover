import os
import numpy as np


FRAME_SHAPE = (160, 320)
FRAME_ORIGIN = (FRAME_SHAPE[0], FRAME_SHAPE[1]/2)

DST_SIZE = 5
BOTTOM_OFFSET = 6
WORLD_SIZE = 200

# perspective source and target when the roll and pitch = 0

STD_PERSPECTIVE_SOURCE = \
  np.float32([[14.32, 140.71], [120.78, 95.5],
              [199.49, 96.84], [302.7, 140.71]])

STD_PERSPECTIVE_TARGET = \
  np.float32([[FRAME_SHAPE[1]/2 - DST_SIZE,
               FRAME_SHAPE[0] - BOTTOM_OFFSET],
              [FRAME_SHAPE[1]/2 - DST_SIZE,
               FRAME_SHAPE[0] - BOTTOM_OFFSET - 2 * DST_SIZE],
              [FRAME_SHAPE[1]/2 + DST_SIZE,
               FRAME_SHAPE[0] - BOTTOM_OFFSET - 2 * DST_SIZE],
              [FRAME_SHAPE[1]/2 + DST_SIZE,
               FRAME_SHAPE[0] - BOTTOM_OFFSET]])

STD_THRESH = np.uint32([160, 160, 160])


DIR_NAME = os.path.dirname(__file__)

PROJECT_DIR = os.path.abspath(DIR_NAME + '../../../')

GROUND_TRUTH_MAP = os.path.join(
  DIR_NAME, 'calibration_images/map_bw.png')


DEFAULT_THRESHOLD = np.float32([160, 160, 160])


STR_SCHEMA = 'SteerAngle;Throttle;Brake;Speed;X_Position;Y_Position;Pitch;Yaw;Roll'
STR_SCHEMA_ROW = tuple(STR_SCHEMA.split(';'))

METRIC_DTYPE = [(x, np.float32) for x in STR_SCHEMA.split(';')]
METRIC_DTYPE += [('NearSample', np.uint32), ('PickingUp', np.uint32)]
METRIC_DTYPE = np.dtype(METRIC_DTYPE)


# evaluated from calibration optimization
CAMERA_POSITION = np.array(
  [0.27883144,  0.07262362,  0.17211595], dtype=np.float64)
CAMERA_POSITION.flags.writeable = False

# evaluated from calibration optimization
VIEW_POINT_POSITION = np.array(
  [0.04027264,  0.08037242, -0.05411853], dtype=np.float64)
VIEW_POINT_POSITION.flags.writeable = False

# guess value
PIXEL_SCALING = 2000
