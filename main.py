import math
import numpy as np
import pygame

from constants import *
from maze import init_maze
from utils import move_along_wall


pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Robot in Maze Simulation")

angle_offsets = np.arange(0, 360 + 1, 360//N_ANGLE_OFFSETS) + 90
sensors_text_offsets = [(np.cos(angle * np.pi / 180), np.sin(angle * np.pi / 180)) for angle in angle_offsets]


entrance = pygame.Rect(50, 250, 10, 100)
finish = pygame.Rect(730, 250, 10, 100)


def get_sensor_values(robot_x, robot_y, robot_angle, maze_walls):
    
    distances = []
    for angle_offset in angle_offsets:
        angle = math.radians(angle_offset)
        for dist in range(1, MAX_SENSOR_RANGE + 1):
            sensor_x = robot_x + dist * math.cos(angle)
            sensor_y = robot_y + dist * math.sin(angle)
            if any(wall.collidepoint(sensor_x, sensor_y) for wall in maze_walls):
                distances.append(dist)
                break
        else:
            distances.append(MAX_SENSOR_RANGE)  
    return distances

robot_trace = [(robot_x, robot_y)]  

running = True
clock = pygame.time.Clock()

maze_walls = init_maze()

while running:
    screen.fill(WHITE)

    # Maze drawing
    for wall in maze_walls:
        pygame.draw.rect(screen, BLACK, wall)
    pygame.draw.rect(screen, BLUE, entrance)
    pygame.draw.rect(screen, RED, finish)

    # Draw robot trace
    for trace_x, trace_y in robot_trace:
        pygame.draw.circle(screen, ORANGE_LIGHT, (int(trace_x), int(trace_y)), 2)

    # Robot drawing
    pygame.draw.circle(screen, ORANGE, (int(robot_x), int(robot_y)), robot_radius)
    line_x = robot_x + robot_radius * math.cos(math.radians(robot_angle))
    line_y = robot_y + robot_radius * math.sin(math.radians(robot_angle))
    pygame.draw.line(screen, BLACK, (int(robot_x), int(robot_y)), (int(line_x), int(line_y)), 3)

    # Sensor values
    sensor_values = get_sensor_values(robot_x, robot_y, robot_angle, maze_walls)
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
    if keys[pygame.K_UP]:
        new_x = robot_x + robot_speed * math.cos(math.radians(robot_angle))
        new_y = robot_y + robot_speed * math.sin(math.radians(robot_angle))
        if not any(wall.collidepoint(new_x, new_y) for wall in maze_walls):
            robot_x, robot_y = new_x, new_y
        else:
            robot_x, robot_y = move_along_wall(robot_x, robot_y, robot_angle, robot_speed, maze_walls)
    if keys[pygame.K_DOWN]:
        new_x = robot_x - robot_speed * math.cos(math.radians(robot_angle))
        new_y = robot_y - robot_speed * math.sin(math.radians(robot_angle))
        if not any(wall.collidepoint(new_x, new_y) for wall in maze_walls):
            robot_x, robot_y = new_x, new_y
        else:
            robot_x, robot_y = move_along_wall(robot_x, robot_y, robot_angle, robot_speed, maze_walls)
    if keys[pygame.K_LEFT]:
        robot_angle = (robot_angle - 5) % 360
    if keys[pygame.K_RIGHT]:
        robot_angle = (robot_angle + 5) % 360

    if robot_trace[-1] != (robot_x, robot_y):
        robot_trace.append((robot_x, robot_y))
    if len(robot_trace) > 500:  # Limit the trace length
        robot_trace.pop(0)

    # Checking for finish
    if finish.collidepoint(robot_x, robot_y):
        print("You reached the finish!")
        running = False

    # Updating display
    pygame.display.flip()
    clock.tick(30)

pygame.quit()
