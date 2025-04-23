WIDTH, HEIGHT = 800, 600

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
ORANGE = (255, 100, 0)
ORANGE_LIGHT = (255, 219, 187)
BLUE = (0, 0, 255)
BLUE_LIGHT = (173, 216, 230)

robot_radius = 20
robot_x, robot_y = 100, 100  # Initial robot position
robot_angle = 0  # Initial angle - facing right
robot_speed = 2
N_ANGLE_OFFSETS = 8

# standard deviation - motion model
SR_X, SR_Y, SR_THETA = 0.2, 0.2, 0.005
# standard deviation - sensor model
SQ_X, SQ_Y, SQ_THETA = 1, 1, 1

MAX_SENSOR_RANGE = 200
MAX_HISTORY_SIZE = 500