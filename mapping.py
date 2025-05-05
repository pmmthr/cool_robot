import math
import numpy as np
import pygame

from constants import *

class Mapping:

    def __init__(self, width, height, resolution,step_factor=0.25):
        # World dimensions (pixels) and cell resolution
        self.width = width
        self.height = height
        self.resolution = resolution
        self.cols = int(math.ceil(width / resolution))
        self.rows = int(math.ceil(height / resolution))
        self.step_factor = step_factor

        # Prior p₀=0.5 and its log-odds l₀ (Eq. 9.7)
        self.p0 = 0.5
        self.l0 = math.log(self.p0 / (1 - self.p0))

        self._last_pose  = None
        self._last_theta = None
        
        # Slide 3 in introduction to assignement 3 slides.
        self.p_occ = 0.9    
        self.p_free = 0.3   
        

        # Obstacle thickness alpha (Table 9.2)
        self.alpha = resolution

        # Initialize log-odds grid to prior l₀ 
        self.log_odds = np.full((self.rows, self.cols), self.l0, dtype=float)

    def world_to_map(self, x, y):

        return int(y // self.resolution), int(x // self.resolution)

    def map_to_world(self, row, col):

        return ((col + 0.5) * self.resolution,
                (row + 0.5) * self.resolution)

    def inverse_sensor_model(self, row, col, pose, z, z_max):
        """
        Three cases from Table 9.2:
          1) r > min(z_max, z + α/2)   -> last p0
          2) z < z_max and |r−z|<α/2    -> p_occ
          3) r ≤ z                    -> p_free
          else                         -> p0
        """
        # world coords of cell center
        x_c, y_c = self.map_to_world(row, col)
        x, y = pose
        # distance r (line 3) table 9.2
        r = math.hypot(x_c - x, y_c - y)

        # CASE 1: cells beyond max range or just behind hit (line 6)
        if r > min(z_max, z + self.alpha/2):
            return self.p0
        # CASE 2: on obstacle shell (line 8)
        if z < z_max and abs(r - z) < self.alpha/2:
            return self.p_occ
        # CASE 3: free cells before obstacle (line 10)
        if r <= z:
            return self.p_free
        # fallback to prior
        return self.p0

    def update_with_scan(self, pose, theta, z_max, walls, landmarks, n_beams=100):
        """
        Bayes update (Eq. 9.5, Table 9.1): (line 4)
        """
        
        if self._last_pose is not None:
            dx = pose[0] - self._last_pose[0]
            dy = pose[1] - self._last_pose[1]
            dtheta = abs((theta - self._last_theta + math.pi) % (2*math.pi) - math.pi)

            if math.hypot(dx, dy) < (self.resolution * 0.5) and dtheta < math.radians(5):
                return  # too little motion


        # cast 360°
        self._last_pose  = pose
        self._last_theta = theta
        x, y = pose
        angles = np.linspace(theta, theta + 2*math.pi, n_beams, endpoint=False)
        
        #loop through each angle beam 
        for ang in angles:
            z = z_max
            step = self.resolution*self.step_factor  #change step 
            dist = step
            while dist <= z_max:
                rx = x + dist * math.cos(ang) #rx and ry to get z
                ry = y + dist * math.sin(ang)
                #check for wall in angle beam
                
                # look for walls
                if any(w.collidepoint(rx, ry) for w in walls):
                    z = dist
                    break
                hit = False
                # look for landmarks
                for lm in landmarks:
                    lx, ly = lm.center
                    if (rx-lx)**2 + (ry-ly)**2 <= lm.radius**2:
                        z = dist
                        hit = True
                        break
                if hit:
                    break

                dist += step
                
            # “for all cells m in perceptual field of this beam”: (occupancy grid mapping algorithm p226 table 9.1)
            # update each cell along ray
            max_steps = int(math.ceil(z_max / self.resolution))
            for k in range(max_steps+1):
                r_k = k * self.resolution
                if r_k > z_max:
                    break
                cx = x + r_k * math.cos(ang) #cx and cy are the grid cells along the specific beam we have calculated (z)
                cy = y + r_k * math.sin(ang)
                row, col = self.world_to_map(cx, cy)
                if not (0 <= row < self.rows and 0 <= col < self.cols):
                    continue
                # get P(m | zx...) and convert to log-odds
                p = self.inverse_sensor_model(row, col, pose, z, z_max)   # table 9.1 p226 get the inverse sensor model probability
                l = math.log(p/(1-p))  # Eq. 9.5 inverse log-odds
                # Bayes update
                self.log_odds[row, col] += (l - self.l0) # This is line 3 in table 9.1 

    def draw(self, screen):

        for r in range(self.rows):
            for c in range(self.cols):               
                
                l = self.log_odds[r, c]
                
                p = 1.0 - 1.0/(1.0 + math.exp(l)) #Eq 9.6 from book p225

                    
                gray = int((1.0 - p) * 255) # color with respect to probabilituy
                pygame.draw.rect(screen,(gray,)*3, pygame.Rect(c*self.resolution, r*self.resolution,
                                self.resolution,
                                self.resolution)
                )
