import math

def move_along_wall(robot_x, robot_y, robot_angle, robot_speed, maze_walls):
    
    dx = robot_speed * math.cos(math.radians(robot_angle))
    dy = robot_speed * math.sin(math.radians(robot_angle))

    collided_x = any(wall.collidepoint(robot_x + dx, robot_y) for wall in maze_walls)
    collided_y = any(wall.collidepoint(robot_x, robot_y + dy) for wall in maze_walls)

    if not collided_x:
        robot_x += dx
    if not collided_y:
        robot_y += dy
        
    return robot_x, robot_y