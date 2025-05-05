# exp1.py

import math
import numpy as np

from constants import WIDTH, HEIGHT, MAX_SENSOR_RANGE
from mapping   import Mapping
from maze      import init_maze, init_landmarks

# 1) Only sweep these resolutions
resolutions = [5, 10, 20, 30]

# 2) Load the recorded trajectory
trajectory = []
with open("trace.txt") as f:
    for line in f:
        x, y, deg = map(float, line.strip().split(","))
        trajectory.append(((x, y), math.radians(deg)))

# 3) Initialize walls & landmarks once
walls     = init_maze()
landmarks = init_landmarks()

# 4) Build ground truth (walls + landmarks)
def build_ground_truth(resolution):
    rows = int(math.ceil(HEIGHT / resolution))
    cols = int(math.ceil(WIDTH  / resolution))
    gt   = np.zeros((rows, cols), dtype=int)

    for r in range(rows):
        for c in range(cols):
            cx = (c + 0.5) * resolution
            cy = (r + 0.5) * resolution

            occupied = False
            # check walls
            for w in walls:
                if w.collidepoint(cx, cy):
                    occupied = True
                    break

            # check landmarks if not already occupied
            if not occupied:
                for lm in landmarks:
                    lx, ly = lm.center
                    if (cx - lx)**2 + (cy - ly)**2 <= lm.radius**2:
                        occupied = True
                        break

            gt[r, c] = 1 if occupied else 0

    return gt

# 5) Numerically stable sigmoid for log-odds → probability
def sigmoid(l):
    # l is a NumPy array
    # stable: for l>=0 use 1/(1+exp(-l)), else exp(l)/(1+exp(l))
    out = np.empty_like(l, dtype=float)
    pos = l >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-l[pos]))
    exp_l   = np.exp(l[neg])
    out[neg] = exp_l / (1.0 + exp_l)
    return out

# 6) Run the sweep and print CSV header
print("resolution_px,mean_abs_error")
for res in resolutions:
    mapping = Mapping(WIDTH, HEIGHT, res)
    gt      = build_ground_truth(res)

    for (x, y), theta in trajectory:
        mapping.update_with_scan((x, y), theta,
                                 MAX_SENSOR_RANGE,
                                 walls, landmarks)

    # convert log-odds → probability safely
    prob = sigmoid(mapping.log_odds)

    error = np.mean(np.abs(prob - gt))
    print(f"{res},{error:.6f}")
