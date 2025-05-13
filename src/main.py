import math
import random
import numpy as np
import pygame

from constants import *
from kalman_filter import KalmanFilter
from maze import init_landmarks, init_maze
from utils import move_along_wall
from mapping import Mapping

# Initialize Pygame

pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Robot in Maze Simulation")

angle_offsets = np.arange(0, 360 + 1, 360//N_ANGLE_OFFSETS) + 90
sensors_text_offsets = [(np.cos(angle * np.pi / 180), np.sin(angle * np.pi / 180)) for angle in angle_offsets]

entrance = pygame.Rect(50, 250, 10, 100)
finish = pygame.Rect(730, 250, 10, 100)


# Robot state
robot_x, robot_y = INITIAL_STATE[0], INITIAL_STATE[1]
robot_angle = math.degrees(INITIAL_STATE[2])  # in degrees

robot_trace = [(robot_x, robot_y)]

est_x, est_y= robot_x, robot_y
est_trace = [(est_x, est_y)]



entrance = pygame.Rect(50, 250, 10, 100)



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

running = True
clock = pygame.time.Clock()

finish = pygame.Rect(730, 250, 10, 100)


maze_walls = init_maze()
landmarks = init_landmarks()
#mapping = Mapping(WIDTH, HEIGHT, MAP_RESOLUTION)

trace_file = open("trace.txt", "w")

kf = KalmanFilter(INITIAL_STATE.copy(), INITIAL_COVARIANCE.copy(), MOTION_NOISE, MEASUREMENT_NOISE, MAX_SENSOR_RANGE, landmarks)

running = True


while running:
    dt = clock.tick(100) / 1000.0  # Delta time
    screen.fill(WHITE)

    # Maze drawing
    for wall in maze_walls:
        pygame.draw.rect(screen, BLACK, wall)

    # Landmarks drawing
    for id, landmark in landmarks:
        pygame.draw.circle(screen, ORANGE, landmark.center, landmark.radius)
        
    # Mapping on top of maze drawing
    #mapping.draw(screen)
    pygame.draw.rect(screen, RED, finish)


    # Sensor values
    sensor_values = get_sensor_values(robot_x, robot_y, maze_walls)
    font = pygame.font.SysFont(None, 18)
    for i, value in enumerate(sensor_values):
        text = font.render(str(value), True, BLACK)
        text_offset_x, text_offset_y = sensors_text_offsets[i]
        screen.blit(text, (robot_x + (1.5*robot_radius)*text_offset_x - 5, robot_y + (1.5*robot_radius)*text_offset_y - 5)) # yes weird hardcoding but for alignment

    

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



    # Kalman Filter Predict Step 
    kf.predict(np.array([v, w, dt]))

    # --- Detect Features ---
    detected_features = kf.detect_features(landmarks,(robot_x, robot_y), math.radians(robot_angle))
    detected_features.sort(key=lambda f: f[3])  # Sort by distance (f[3] is distance)

    for fid, feat, bearing, dist in detected_features[:3]:
        dist_meas = dist 
        bearing_meas = bearing
        kf.update(np.array([dist_meas, bearing_meas]), fid)

    # --- Update Mapping ---
    #mapping.update_with_scan((robot_x, robot_y), math.radians(robot_angle), MAX_SENSOR_RANGE, maze_walls, landmarks, n_beams=N_ANGLE_OFFSETS)

    est_x, est_y, _ = kf.state

    # --- Update Traces ---
    if robot_trace[-1] != (robot_x, robot_y):
        robot_trace.append((robot_x, robot_y))
    if est_trace[-1] != (est_x, est_y):
        est_trace.append((est_x, est_y))
    if len(robot_trace) > MAX_HISTORY_SIZE:  # Limit the trace length
        robot_trace.pop(0)
    if len(est_trace) > MAX_HISTORY_SIZE:  # Limit the trace length
        est_trace.pop(0)


 # DRAWING 

    # Robot drawing
    pygame.draw.circle(screen, ORANGE, (int(robot_x), int(robot_y)), robot_radius)
    line_x = robot_x + robot_radius * math.cos(math.radians(robot_angle))
    line_y = robot_y + robot_radius * math.sin(math.radians(robot_angle))
    pygame.draw.line(screen, BLACK, (int(robot_x), int(robot_y)), (int(line_x), int(line_y)), 2)

    
    # Draw true robot trace 
    for trace_x, trace_y in robot_trace:
        pygame.draw.circle(screen, ORANGE_LIGHT, (int(trace_x), int(trace_y)), 2)
    # Draw estimated trace
    for est_trace_x, est_trace_y in est_trace:
        pygame.draw.circle(screen, BLUE_LIGHT, (int(est_trace_x), int(est_trace_y)), 2)
        
        
    # Sensor lines
    for fid, feat, bearing, dist in detected_features:
        pygame.draw.line(screen, GREEN, (int(robot_x), int(robot_y)), (int(feat[0]), int(feat[1])), 1)
        
        
    # Draw EKF estimated position and uncertainty 
    if math.isfinite(est_x) and math.isfinite(est_y):
        pygame.draw.circle(screen, DOTTED_LINE_COLOR,(int(est_x), int(est_y)),robot_radius, 1)

        kf.draw_covariance_ellipse(screen)

    # --- Check if robot reached finish ---
    if finish.collidepoint(robot_x, robot_y):
        print("You reached the finish!")
        running = False
        
        
    trace_file.write(f"{robot_x},{robot_y},{robot_angle}\n")

    pygame.display.flip()


pygame.quit()
