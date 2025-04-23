import pygame
from collections import namedtuple

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
    points = [
        (520, 259), 
        (111, 521), 
        (325, 482), 
        (322, 210), 
        (444, 229), 
        (402, 510), 
        (520, 500), 
        (168, 480), 
        (485, 195), 
        (326, 100),
        (134, 349), 
        (577, 211), 
        (299, 388), 
        (420, 583), 
        (65, 290), 
        (550, 115), 
        (368, 421), 
        (241, 564), 
        (512, 300), 
        (110, 190),
    ]

    landmarks = [Landmark(center=p, radius=10) for p in points]    
    return landmarks