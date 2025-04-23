import pygame

def init_maze():
    # TODO: Here possibly add generation of random maze or maze from the picture
    
    maze_walls = [
        pygame.Rect(50, 50, 700, 10),  # Top wall
        pygame.Rect(50, 50, 10, 500),  # Left wall
        pygame.Rect(50, 540, 700, 10),  # Bottom wall
        pygame.Rect(740, 50, 10, 500),  # Right wall
        pygame.Rect(300, 50, 10, 400),  # Vertical wall
        pygame.Rect(300, 450, 200, 10),  # Horizontal wall
    ]
    
    return maze_walls