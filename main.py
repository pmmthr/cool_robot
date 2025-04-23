import math
import numpy as np
import pygame

from constants import *
from sensors import get_sensor_values, get_observation
from maze import init_maze, init_landmarks
from utils import move_along_wall
from forward import forward

pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Robot in Maze Simulation")

angle_offsets = np.arange(0, 360 + 1, 360//N_ANGLE_OFFSETS) + 90
sensors_text_offsets = [(np.cos(angle * np.pi / 180), np.sin(angle * np.pi / 180)) for angle in angle_offsets]

entrance = pygame.Rect(50, 250, 10, 100)
finish = pygame.Rect(730, 250, 10, 100)

robot_trace = [(robot_x, robot_y)]
est_x, est_y = 300, 300
est_trace = [(est_x, est_y)]  

running = True
clock = pygame.time.Clock()

maze_walls = init_maze()
landmarks = init_landmarks()

while running:
    screen.fill(WHITE)

    # Maze drawing
    for wall in maze_walls:
        pygame.draw.rect(screen, BLACK, wall)
    for landmark in landmarks:
        pygame.draw.circle(screen, ORANGE, landmark.center, landmark.radius)

    pygame.draw.rect(screen, BLUE, entrance)
    pygame.draw.rect(screen, RED, finish)

    # Draw robot trace
    for trace_x, trace_y in robot_trace:
        pygame.draw.circle(screen, ORANGE_LIGHT, (int(trace_x), int(trace_y)), 2)
    # Draw estimated trace
    for est_trace_x, est_trace_y in est_trace:
        pygame.draw.circle(screen, BLUE_LIGHT, (int(est_trace_x), int(est_trace_y)), 2)

    # Robot drawing
    pygame.draw.circle(screen, ORANGE, (int(robot_x), int(robot_y)), robot_radius)
    line_x = robot_x + robot_radius * math.cos(math.radians(robot_angle))
    line_y = robot_y + robot_radius * math.sin(math.radians(robot_angle))
    pygame.draw.line(screen, BLACK, (int(robot_x), int(robot_y)), (int(line_x), int(line_y)), 3)

    # Sensor values
    sensor_values = get_sensor_values(robot_x, robot_y, maze_walls)
    font = pygame.font.SysFont(None, 18)
    for i, value in enumerate(sensor_values):
        text = font.render(str(value), True, BLACK)
        text_offset_x, text_offset_y = sensors_text_offsets[i]
        screen.blit(text, (robot_x + (1.5*robot_radius)*text_offset_x - 5, robot_y + (1.5*robot_radius)*text_offset_y - 5)) # yes weird hardcoding but for alignment

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    keys = pygame.key.get_pressed()
    keys_mask = [
        keys[pygame.K_UP], 
        keys[pygame.K_DOWN],
        keys[pygame.K_LEFT],
        keys[pygame.K_RIGHT],
    ]

    # Robot movement
    robot_x, robot_y, robot_angle = forward(robot_x, robot_y, robot_angle, keys_mask, robot_speed, maze_walls)
    

    if robot_trace[-1] != (robot_x, robot_y):
        robot_trace.append((robot_x, robot_y))
    if est_trace[-1] != (est_x, est_y):
        est_trace.append((est_x, est_y))
    if len(robot_trace) > MAX_HISTORY_SIZE:  # Limit the trace length
        robot_trace.pop(0)
    if len(est_trace) > MAX_HISTORY_SIZE:  # Limit the trace length
        est_trace.pop(0)

    # Checking for finish
    if finish.collidepoint(robot_x, robot_y):
        print("You reached the finish!")
        running = False

    # Updating display
    pygame.display.flip()
    clock.tick(30)

pygame.quit()
