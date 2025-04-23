#!/usr/bin/env python3
import math
import numpy as np
import pygame
import random

from constants import *
from maze import init_maze
from utils import move_along_wall
from kalman_filter import KalmanFilter
import matplotlib.pyplot as plt


pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Robot in Maze with EKF Localization")
clock = pygame.time.Clock()

# Precompute sensor/display angles
angle_offsets = np.linspace(0, 360, N_ANGLE_OFFSETS, endpoint=False) + 90
sensors_text_offsets = [
    (math.cos(math.radians(a)), math.sin(math.radians(a)))
    for a in angle_offsets
]

entrance = pygame.Rect(50, 250, 10, 100)
finish   = pygame.Rect(730, 250, 10, 100)
maze_walls = init_maze()

# Fixed landmarks
features = [
    (i, [random.randint(100, WIDTH-100), random.randint(100, HEIGHT-100)])
    for i in range(FEATURE_COUNT)
]

# EKF instance
kf = KalmanFilter(
    INITIAL_STATE.copy(),
    INITIAL_COVARIANCE.copy(),
    MOTION_NOISE,
    MEASUREMENT_NOISE,
    MAX_SENSOR_RANGE,
    features
)

# how many landmarks to fuse per update
N_LANDMARKS = 3

def draw_dotted_line(scr, p0, p1, col, sp=10):
    dx, dy = p1[0]-p0[0], p1[1]-p0[1]
    d = math.hypot(dx, dy)
    if d < 1e-6: return
    ux, uy = dx/d, dy/d
    for i in range(0, int(d), sp*2):
        a = (int(p0[0]+i*ux), int(p0[1]+i*uy))
        b = (int(p0[0]+min(i+sp,d)*ux), int(p0[1]+min(i+sp,d)*uy))
        pygame.draw.line(scr, col, a, b, 1)

# “True” state
robot_x, robot_y = INITIAL_STATE[0], INITIAL_STATE[1]
robot_theta     = INITIAL_STATE[2]  # radians

# traces for plotting
robot_trace     = [(robot_x, robot_y)]
estimated_trace = [(robot_x, robot_y)]

# speeds
LIN_SPEED = robot_speed         # px/sec
ANG_SPEED = math.radians(90)    # rad/sec

running = True

errors_1 = []
errors_2 = []
errors_3 = []

while running:
    ex, ey, _ = kf.state
# Compute localization error
    true_pos = np.array([robot_x, robot_y])
    est_pos = np.array([ex, ey])
    error = np.linalg.norm(true_pos - est_pos)

    # 1) handle quit & compute dt, v, w
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            running = False

    dt_ms = clock.tick(60)     # limit to 60 FPS
    dt    = dt_ms / 1000.0     # convert to seconds
    keys  = pygame.key.get_pressed()

    # 🔧 Define motion commands based on key presses
    v_cmd = LIN_SPEED if keys[pygame.K_UP] else -LIN_SPEED if keys[pygame.K_DOWN] else 0.0
    w_cmd = ANG_SPEED if keys[pygame.K_RIGHT] else -ANG_SPEED if keys[pygame.K_LEFT] else 0.0

    # 2) Move the “true” robot
    dx, dy = v_cmd * math.cos(robot_theta) * dt, v_cmd * math.sin(robot_theta) * dt
    nx, ny = robot_x + dx, robot_y + dy
    if not any(wall.collidepoint(nx, ny) for wall in maze_walls):
        robot_x, robot_y = nx, ny
    else:
        robot_x, robot_y = move_along_wall(robot_x, robot_y, robot_theta, v_cmd * dt, maze_walls)

    robot_theta = math.atan2(
        math.sin(robot_theta + w_cmd * dt),
        math.cos(robot_theta + w_cmd * dt)
    )

    # 3) Add noise to control inputs for odometry
    α1, α2, α3, α4 = MOTION_NOISE
    σ_v = math.sqrt(α1 * v_cmd**2 + α2 * w_cmd**2)
    σ_w = math.sqrt(α3 * v_cmd**2 + α4 * w_cmd**2)
    v_odom = v_cmd + np.random.randn() * σ_v
    w_odom = w_cmd + np.random.randn() * σ_w

    # 4) Predict step of EKF
    robot_trace.append((robot_x, robot_y))
    if len(robot_trace) > MAX_HISTORY_SIZE:
        robot_trace.pop(0)
    kf.predict(np.array([v_odom, w_odom, dt]))
    ex, ey, _ = kf.state
    estimated_trace.append((ex, ey))
    if len(estimated_trace) > MAX_HISTORY_SIZE:
        estimated_trace.pop(0)




    # 5) Update step of EKF with nearby features
    detected = kf.detect_features((robot_x, robot_y), robot_theta)
    detected.sort(key=lambda f: f[3])  # sort by distance
    num_detected = len(detected)
    top_features = detected[:N_LANDMARKS]


    true_pos = np.array([robot_x, robot_y])
    est_pos = np.array([ex, ey])
    error = np.linalg.norm(true_pos - est_pos)

    if num_detected == 1:
        errors_1.append(error)
    elif num_detected == 2:
        errors_2.append(error)
    elif num_detected >= 3:
        errors_3.append(error)


    for fid, feat, bearing, dist in top_features:
        σ_r  = math.sqrt(MEASUREMENT_NOISE[0, 0])
        σ_φ  = math.sqrt(MEASUREMENT_NOISE[1, 1])
        dist_meas    = dist + np.random.randn() * σ_r
        bearing_meas = bearing + np.random.randn() * σ_φ


        kf.update(np.array([dist_meas, bearing_meas]), fid)


    # 6) Draw everything
    screen.fill(WHITE)
    for wall in maze_walls:
        pygame.draw.rect(screen, BLACK, wall)
    pygame.draw.rect(screen, BLUE, entrance)
    pygame.draw.rect(screen, RED, finish)

    for _, pos in features:
        pygame.draw.circle(screen, BLACK, (int(pos[0]), int(pos[1])), FEATURE_RADIUS)
    for fid, feat, bearing, dist in top_features:
        pygame.draw.line(
            screen, GREEN,
            (int(robot_x), int(robot_y)),
            (int(feat[0]), int(feat[1])),
            1
        )

    if len(robot_trace) > 1:
        pygame.draw.lines(screen, BLACK, False, robot_trace, 2)

    if len(estimated_trace) > 1:
        for i in range(len(estimated_trace) - 1):
            draw_dotted_line(screen, estimated_trace[i], estimated_trace[i + 1], DOTTED_LINE_COLOR)

    # Robot visualization
    pygame.draw.circle(screen, ORANGE, (int(robot_x), int(robot_y)), robot_radius)
    hx = robot_x + robot_radius * math.cos(robot_theta)
    hy = robot_y + robot_radius * math.sin(robot_theta)
    pygame.draw.line(screen, BLACK, (int(robot_x), int(robot_y)), (int(hx), int(hy)), 3)

    # EKF visualization
    pygame.draw.circle(screen, DOTTED_LINE_COLOR, (int(ex), int(ey)), robot_radius, 1)
    kf.draw_covariance_ellipse(screen)

    if finish.collidepoint(robot_x, robot_y):
        print("You reached the finish!")
        running = False

    pygame.display.flip()
    




def print_and_plot_errors():
    labels = ['1 Landmark', '2 Landmarks', '3 Landmarks']
    data = [
        errors_1,
        errors_2,
        errors_3
    ]

    means = [sum(e)/len(e) if e else 0 for e in data]
    for label, mean, samples in zip(labels, means, data):
        print(f"{label}: {mean:.2f} px (based on {len(samples)} samples)")

    # Plot
    plt.figure()
    plt.bar(labels, means)
    plt.ylabel("Mean Localization Error (px)")
    plt.title("Error vs. Number of Visible Landmarks")
    plt.grid(True)
    plt.show()

print_and_plot_errors()