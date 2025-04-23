def forward(robot_x, robot_y, robot_angle, robot_speed, maze_walls):
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

	return robot_x, robot_y, robot_angle