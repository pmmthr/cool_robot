import math
import numpy as np
import pygame
import os, sys

# ensure project root is on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from constants import *
from kalman_filter import KalmanFilter
from maze import init_landmarks, init_maze, Door
from utils import move_along_wall
from mapping import Mapping

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Occupancy Mapping with Moving Door")
clock = pygame.time.Clock()

# sensor angles
angle_offsets = np.linspace(0, 360, N_ANGLE_OFFSETS, endpoint=False) + 90

# World setup
maze_walls = init_maze()  # static walls
# instantiate the door at that location
door = Door(pygame.Rect(500, 450, 100, 10))
# remove any static walls overlapping the door so it is the only barrier
maze_walls = [w for w in maze_walls if not w.colliderect(door.rect)]
landmarks = init_landmarks()
mapping = Mapping(WIDTH, HEIGHT, 8, 0.25)

# Robot true state
robot_x, robot_y = INITIAL_STATE[0], INITIAL_STATE[1]
robot_angle = math.degrees(INITIAL_STATE[2])
robot_trace = [(robot_x, robot_y)]
est_trace = [(robot_x, robot_y)]

# EKF for localization
kf = KalmanFilter(
    INITIAL_STATE.copy(),
    INITIAL_COVARIANCE.copy(),
    MOTION_NOISE,
    MEASUREMENT_NOISE,
    MAX_SENSOR_RANGE,
    landmarks)

# Sensor reading function
def get_sensor_values(x, y, walls):
    distances = []
    for angle in angle_offsets:
        θ = math.radians(angle)
        for d in range(1, MAX_SENSOR_RANGE + 1):
            sx = x + d * math.cos(θ)
            sy = y + d * math.sin(θ)
            if any(w.collidepoint(sx, sy) for w in walls):
                distances.append(d)
                break
        else:
            distances.append(MAX_SENSOR_RANGE)
    return distances

# Main loop
running = True
while running:
    dt = clock.tick(60) / 1000.0
    screen.fill(WHITE)

        # Build current collision list including dynamic door barrier
    # If the door is closed, barrier is horizontal rect; if open, barrier is vertical swing
    if door.is_open:
        ox, oy, ow, oh = door.rect
        barrier = pygame.Rect(ox, oy - ow, oh, ow)
    else:
        barrier = door.rect
    current_walls = maze_walls + [barrier]

    # Draw static walls
    for wall in maze_walls:
        pygame.draw.rect(screen, BLACK, wall)
    # Draw door (closed: horizontal, open: swung up)
    if door.is_open:
        ox, oy, ow, oh = door.rect
        open_rect = pygame.Rect(ox, oy - ow, oh, ow)
        pygame.draw.rect(screen, (139, 69, 19), open_rect)
    else:
        pygame.draw.rect(screen, (139, 69, 19), door.rect)

    # Draw landmarks
    for _, lm in landmarks:
        pygame.draw.circle(screen, ORANGE, lm.center, lm.radius)

    mapping.draw(screen)

    # Handle events
    v = 0.0; w = 0.0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_o:
            door.toggle()


    # Keyboard motion
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:    v = robot_speed
    if keys[pygame.K_DOWN]:  v = -robot_speed
    if keys[pygame.K_LEFT]:  w = -math.radians(90)
    if keys[pygame.K_RIGHT]: w = math.radians(90)
    mapping.update_with_scan((robot_x, robot_y),math.radians(robot_angle),MAX_SENSOR_RANGE,current_walls,landmarks,30)

    # Move robot with collision checking
    new_x = robot_x + v * dt * math.cos(math.radians(robot_angle))
    new_y = robot_y + v * dt * math.sin(math.radians(robot_angle))
    if not any(wall.collidepoint(new_x, new_y) for wall in current_walls):
        robot_x, robot_y = new_x, new_y
    else:
        robot_x, robot_y = move_along_wall(
            robot_x, robot_y, robot_angle,
            robot_speed, current_walls, dt
        )
    robot_angle = (robot_angle + math.degrees(w) * dt) % 360


    # EKF predict
    kf.predict(np.array([v, w, dt]))

    # EKF update using up to 3 closest landmarks
    detected = kf.detect_features(landmarks, (robot_x, robot_y), math.radians(robot_angle))
    detected.sort(key=lambda f: f[3])
    for fid, feat, bearing, dist in detected[:3]:
        kf.update(np.array([dist, bearing]), fid)

    # Draw true robot trace
    robot_trace.append((robot_x, robot_y))
    if len(robot_trace) > MAX_HISTORY_SIZE:
        robot_trace.pop(0)
    for px, py in robot_trace:
        pygame.draw.circle(screen, ORANGE_LIGHT, (int(px), int(py)), 2)

    # Draw EKF estimated trace
    ex, ey, _ = kf.state
    est_trace.append((ex, ey))
    if len(est_trace) > MAX_HISTORY_SIZE:
        est_trace.pop(0)
    for ex, ey in est_trace:
        pygame.draw.circle(screen, BLUE_LIGHT, (int(ex), int(ey)), 2)

    # Draw robot & heading
    pygame.draw.circle(screen, ORANGE, (int(robot_x), int(robot_y)), robot_radius)
    fx = robot_x + robot_radius * math.cos(math.radians(robot_angle))
    fy = robot_y + robot_radius * math.sin(math.radians(robot_angle))
    pygame.draw.line(screen, BLACK, (int(robot_x), int(robot_y)), (int(fx), int(fy)), 2)

    # Draw EKF covariance ellipse
    kf.draw_covariance_ellipse(screen)

    pygame.display.flip()

pygame.quit()
