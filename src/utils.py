import math
import heapq

def move_along_wall(robot_x, robot_y, robot_angle, robot_speed, maze_walls,dt):
    
    dx = robot_speed *dt* math.cos(math.radians(robot_angle))
    dy = robot_speed *dt* math.sin(math.radians(robot_angle))

    collided_x = any(wall.collidepoint(robot_x + dx, robot_y) for wall in maze_walls)
    collided_y = any(wall.collidepoint(robot_x, robot_y + dy) for wall in maze_walls)

    if not collided_x:
        robot_x += dx
    if not collided_y:
        robot_y += dy
        
    return robot_x, robot_y

def a_star(start, goal, maze_walls, grid_size):
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1], True

        neighbors = [
            (current[0] + dx, current[1] + dy)
            for dx, dy in [(-grid_size, 0), (grid_size, 0), (0, -grid_size), (0, grid_size)]
        ]

        for neighbor in neighbors:
            if any(wall.collidepoint(neighbor) for wall in maze_walls):
                continue

            tentative_g_score = g_score[current] + 1
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return path, False