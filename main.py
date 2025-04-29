import math
import random
import numpy as np
import pygame

from constants import *
from kalman_filter import KalmanFilter
from maze import init_maze
from utils import move_along_wall
from mapping import Mapping

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Robot in Maze with EKF Localization")
clock = pygame.time.Clock()

# Sensors setup
angle_offsets = np.linspace(0, 360, N_ANGLE_OFFSETS, endpoint=False) + 90
sensors_text_offsets = [(np.cos(np.radians(a)), np.sin(np.radians(a))) for a in angle_offsets]

# Maze and entrance/finish
maze_walls = init_maze()
entrance = pygame.Rect(50, 250, 10, 100)
finish = pygame.Rect(730, 250, 10, 100)



# Robot state
robot_x, robot_y = INITIAL_STATE[0], INITIAL_STATE[1]
robot_angle = math.degrees(INITIAL_STATE[2])  # in degrees

# Traces
robot_trace = [(robot_x, robot_y)]
estimated_trace = [(robot_x, robot_y)]

# Dotted line helper
def draw_dotted_line(screen, start, end, color, step=10):
    """
    Draw a dashed line from start to end.
    Aborts early if the distance is NaN/Inf or too small.
    """
    x1, y1 = start
    x2, y2 = end
    dx, dy = x2 - x1, y2 - y1
    distance = math.hypot(dx, dy)

    # Bail out if distance is not a finite number or is essentially zero
    if not math.isfinite(distance) or distance < 1e-6:
        return

    ux, uy = dx / distance, dy / distance
    dash = step
    gap = step

    pos = 0.0
    while pos < distance:
        start_pos = pos
        end_pos = min(pos + dash, distance)

        p1 = (int(x1 + ux * start_pos), int(y1 + uy * start_pos))
        p2 = (int(x1 + ux * end_pos),   int(y1 + uy * end_pos))
        pygame.draw.line(screen, color, p1, p2, 1)

        pos += dash + gap
# Get sensor readings
def get_sensor_values(x, y, walls):
    distances = []
    for angle_offset in angle_offsets:
        angle = math.radians(angle_offset)
        for dist in range(1, MAX_SENSOR_RANGE + 1):
            sensor_x = x + dist * math.cos(angle)
            sensor_y = y + dist * math.sin(angle)
            if any(wall.collidepoint(sensor_x, sensor_y) for wall in walls):
                distances.append(dist)
                break
        else:
            distances.append(MAX_SENSOR_RANGE)
    return distances

def generate_features(count, maze_walls):
    features = []
    for _ in range(count):
        while True:
            x = random.randint(100, WIDTH - 100)
            y = random.randint(100, HEIGHT - 100)
            if not any(wall.collidepoint(x, y) for wall in maze_walls):
                features.append([x, y])
                break
    return features


features_list = generate_features(FEATURE_COUNT, maze_walls)
features = [(i, pos) for i, pos in enumerate(features_list)]
mapping = Mapping(WIDTH, HEIGHT, MAP_RESOLUTION)
kf = KalmanFilter(INITIAL_STATE.copy(), INITIAL_COVARIANCE.copy(), MOTION_NOISE, MEASUREMENT_NOISE, MAX_SENSOR_RANGE, features)

running = True
while running:
    dt = clock.tick(100) / 1000.0  # Delta time

    v=0.0
    w=0.0
    # --- Event handling ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        new_x = robot_x + robot_speed *dt* math.cos(math.radians(robot_angle))
        new_y = robot_y + robot_speed *dt* math.sin(math.radians(robot_angle))
        v = robot_speed

        
        if not any(wall.collidepoint(new_x, new_y) for wall in maze_walls):
            robot_x, robot_y = new_x, new_y
        else:
            robot_x, robot_y = move_along_wall(robot_x, robot_y, robot_angle, robot_speed, maze_walls,dt)
            
    if keys[pygame.K_DOWN]:
        new_x = robot_x - robot_speed *dt* math.cos(math.radians(robot_angle))
        new_y = robot_y - robot_speed *dt* math.sin(math.radians(robot_angle))
        v = - robot_speed

        if not any(wall.collidepoint(new_x, new_y) for wall in maze_walls):
            robot_x, robot_y = new_x, new_y
        else:
            robot_x, robot_y = move_along_wall(robot_x, robot_y, robot_angle, robot_speed, maze_walls,dt)
            
    if keys[pygame.K_LEFT]:
        robot_angle = (robot_angle - 5) % 360
        w = -math.radians(90)

    if keys[pygame.K_RIGHT]:
        robot_angle = (robot_angle + 5) % 360
        w = math.radians(90)



    # --- Kalman Filter Predict Step ---
    kf.predict(np.array([v, w, dt]))

    # --- Detect Features ---
    detected_features = kf.detect_features((robot_x, robot_y), math.radians(robot_angle))
    detected_features.sort(key=lambda f: f[3])  # Sort by distance (f[3] is distance)

    # --- Kalman Filter Update Step with up to 3 closest features ---
    for fid, feat, bearing, dist in detected_features[:3]:
        dr = math.sqrt(MEASUREMENT_NOISE[0, 0])
        dw = math.sqrt(MEASUREMENT_NOISE[1, 1])
        dist_meas = dist + np.random.randn() * dr
        bearing_meas = bearing + np.random.randn() * dw
        kf.update(np.array([dist_meas, bearing_meas]), fid)

    # --- Update Mapping ---
    mapping.update_with_scan((robot_x, robot_y), math.radians(robot_angle), MAX_SENSOR_RANGE, maze_walls, n_beams=N_ANGLE_OFFSETS)

    # --- Update Traces ---
    if robot_trace[-1] != (robot_x, robot_y):
        robot_trace.append((robot_x, robot_y))
    if len(robot_trace) > MAX_HISTORY_SIZE:
        robot_trace.pop(0)

    ex, ey, _ = kf.state
    if estimated_trace[-1] != (ex, ey):
        estimated_trace.append((ex, ey))
    if len(estimated_trace) > MAX_HISTORY_SIZE:
        estimated_trace.pop(0)

    # --- Drawing ---

    screen.fill(WHITE)
    mapping.draw(screen)

    #for wall in maze_walls:
       # pygame.draw.rect(screen, BLACK, wall)

    pygame.draw.rect(screen, BLUE, entrance)
    pygame.draw.rect(screen, RED, finish)

    # --- Draw features ---
    for fid, pos in features:
        pygame.draw.circle(screen, BLACK, (int(pos[0]), int(pos[1])), FEATURE_RADIUS)


    # --- Draw sensor lines to detected features ---
    for fid, feat, bearing, dist in detected_features:
        pygame.draw.line(screen, GREEN, (int(robot_x), int(robot_y)), (int(feat[0]), int(feat[1])), 1)

    # --- Draw true robot trace ---
    if len(robot_trace) > 1:
        pygame.draw.lines(screen, BLACK, False, robot_trace, 2)

    # --- Draw estimated robot trace (dotted) ---
    if len(estimated_trace) > 1:
        for i in range(len(estimated_trace) - 1):
            draw_dotted_line(screen, estimated_trace[i], estimated_trace[i + 1], DOTTED_LINE_COLOR)

    # --- Draw true robot ---
    pygame.draw.circle(screen, ORANGE, (int(robot_x), int(robot_y)), robot_radius)
    hx = robot_x + robot_radius * math.cos(math.radians(robot_angle))
    hy = robot_y + robot_radius * math.sin(math.radians(robot_angle))
    pygame.draw.line(screen, BLACK, (int(robot_x), int(robot_y)), (int(hx), int(hy)), 2)

    # --- Draw EKF estimated position and uncertainty ---
    if math.isfinite(ex) and math.isfinite(ey):
        pygame.draw.circle(screen, DOTTED_LINE_COLOR,
                           (int(ex), int(ey)),
                           robot_radius, 1)

        # only draw the covariance ellipse if it’s a finite matrix

        kf.draw_covariance_ellipse(screen)

    # --- Check if robot reached finish ---
    if finish.collidepoint(robot_x, robot_y):
        print("You reached the finish!")
        running = False

    pygame.display.flip()


pygame.quit()
