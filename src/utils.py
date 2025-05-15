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
    """
    Grid-based A* with 4-connected neighbors.
    start, goal: (row, col) in grid coords
    maze_walls: list of pygame.Rect in world coords
    grid_size: size of one grid cell in pixels
    Returns (path, found)
    """
    def heuristic(a, b):
        # Manhattan distance
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    closed = set()

    while open_set:
        _, current = heapq.heappop(open_set)
        if current in closed:
            continue
        if current == goal:
            # reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1], True

        closed.add(current)
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            neighbor = (current[0] + dr, current[1] + dc)
            if neighbor in closed:
                continue
            # check collision
            wx = (neighbor[1] + 0.5) * grid_size
            wy = (neighbor[0] + 0.5) * grid_size
            if any(wall.collidepoint(wx, wy) for wall in maze_walls):
                continue

            tentative_g = g_score[current] + 1
            if tentative_g < g_score.get(neighbor, math.inf):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return [], False