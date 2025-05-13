import numpy as np
import math

WIDTH, HEIGHT = 800, 600

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
ORANGE = (255, 100, 0)
ORANGE_LIGHT = (255, 219, 187)
BLUE = (0, 0, 255)
BLUE_LIGHT = (173, 216, 230)

GREEN = (0, 255, 0)
DOTTED_LINE_COLOR = (100, 100, 255)



MAP_RESOLUTION = 10   # Size of each grid cell in pixels



robot_radius = 20
robot_x, robot_y = 100, 100  # Initial robot position
robot_angle = 0  # Initial angle - facing right
robot_speed = 70
N_ANGLE_OFFSETS = 8

MAX_SENSOR_RANGE = 100
MAX_HISTORY_SIZE = 50000

# Kalman Filter parameters
INITIAL_STATE = np.array([robot_x, robot_y, math.radians(robot_angle)])
INITIAL_COVARIANCE = np.diag([10.0, 10.0, 0.1])
MEASUREMENT_NOISE = np.diag([0.01, 0.01])
MOTION_NOISE = np.array([0.1, 0.05, 0.02, 0.01])

# Feature detection
FEATURE_RADIUS = 5
FEATURE_COUNT = 10  # Number of features to generate