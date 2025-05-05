import math

def get_dist(x1, y1, x2, y2):
	return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def circle_intersection(c1, r1, c2, r2):
    (x1, y1), (x2, y2) = c1, c2
    dx, dy = x2 - x1, y2 - y1
    d = math.hypot(dx, dy)
    
    if d == 0 or d > r1 + r2 or d < abs(r1 - r2):
        return None
    
    a = (r1**2 - r2**2 + d**2) / (2 * d)
    x = x1 + dx * (a / d)
    y = y1 + dy * (a / d)
    return (x, y)

def triangulate(c1, r1, c2, r2, c3, r3):
    (x1, y1), (x2, y2), (x3, y3) = c1, c2, c3
    A = 2*(x2 - x1)
    B = 2*(y2 - y1)
    C = r1*r1 - r2*r2 - x1*x1 + x2*x2 - y1*y1 + y2*y2

    D = 2*(x3 - x1)
    E = 2*(y3 - y1)
    F = r1*r1 - r3*r3 - x1*x1 + x3*x3 - y1*y1 + y3*y3

    det = A*E - B*D
    if abs(det) < 1e-9:         
        return (0, 0)

    x = (C*E - B*F) / det
    y = (A*F - C*D) / det
    return (x, y)


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




def build_ground_truth(width, height, resolution, walls):
    rows = int(math.ceil(height / resolution))
    cols = int(math.ceil(width  / resolution))
    gt = np.zeros((rows, cols), dtype=int)
    for r in range(rows):
        for c in range(cols):
            x = (c + 0.5) * resolution
            y = (r + 0.5) * resolution
            for w in walls:
                if w.collidepoint(x, y):
                    gt[r, c] = 1
                    break
    return gt
