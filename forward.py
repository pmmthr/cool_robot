import math
from utils import move_along_wall

def forward(robot_x, robot_y, robot_angle, keys_mask, robot_speed, maze_walls):
    if keys_mask[0]:  # UP
        new_x = robot_x + robot_speed * math.cos(math.radians(robot_angle))
        new_y = robot_y + robot_speed * math.sin(math.radians(robot_angle))
        if not any(wall.collidepoint(new_x, new_y) for wall in maze_walls):
            robot_x, robot_y = new_x, new_y
        else:
            robot_x, robot_y = move_along_wall(robot_x, robot_y, robot_angle, robot_speed, maze_walls)
    if keys_mask[1]:  # DOWN
        new_x = robot_x - robot_speed * math.cos(math.radians(robot_angle))
        new_y = robot_y - robot_speed * math.sin(math.radians(robot_angle))
        if not any(wall.collidepoint(new_x, new_y) for wall in maze_walls):
            robot_x, robot_y = new_x, new_y
        else:
            robot_x, robot_y = move_along_wall(robot_x, robot_y, robot_angle, robot_speed, maze_walls)
    if keys_mask[2]: # LEFT
        robot_angle = (robot_angle - 5) % 360
    if keys_mask[3]: # RIGHT
        robot_angle = (robot_angle + 5) % 360
    
    return robot_x, robot_y, robot_angle