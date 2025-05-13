from collections import namedtuple
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

def init_landmarks():
    Landmark = namedtuple('Landmark', ['center', 'radius'])
    # points = [
    #     (520, 259), 
    #     (111, 521), 
    #     (325, 482), 
    #     (322, 210), 
    #     (444, 229), 
    #     (600, 300), 
    #     (110, 190),
    # ]
    
    points = [
        # Top-Left region
        (100, 100), (200, 100), (100, 300), (200, 300), (150, 200),

        # Bottom-Left region 
        (100, 480), (200, 480), (250, 530), (400, 530), (490, 400),

        # Bottom-Right region
        (520, 480), (650, 500), (700, 520),
    ]
    return [
        (i, Landmark(center=pt, radius=10))
        for i, pt in enumerate(points)
    ]
    
    
class Door:
    def __init__(self, rect):
        self.rect = rect
        self.is_open = False

    def toggle(self):
        self.is_open = not self.is_open

    def collision_rects(self):
        # if closed, it collides like a wall; if open, no collision
        return [] if self.is_open else [self.rect]

    def draw(self, screen, color=(139,69,19)):
        if not self.is_open:
            pygame.draw.rect(screen, color, self.rect)