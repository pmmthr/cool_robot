# exp1_step_factor.py

import math
import numpy as np
import time

from constants import WIDTH, HEIGHT, MAX_SENSOR_RANGE
from mapping   import Mapping
from maze      import init_maze, init_landmarks

# 1) Fixed grid resolution (px per cell)
fixed_resolution = 10

# 2) Step‑size factors to sweep (fraction of cell size)
step_factors = [1/8, 1/4, 1/2, 1.0]

# 3) Load the recorded trajectory
trajectory = []
with open("trace.txt") as f:
    for line in f:
        x, y, deg = map(float, line.strip().split(","))
        trajectory.append(((x, y), math.radians(deg)))

# 4) Initialize walls & landmarks once
walls     = init_maze()
landmarks = init_landmarks()

# 5) Build ground truth (walls + landmarks) at fixed_resolution
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

# 6) Numerically stable sigmoid for log-odds → probability
def sigmoid(l):
    out = np.empty_like(l, dtype=float)
    pos = l >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-l[pos]))
    exp_l   = np.exp(l[neg])
    out[neg] = exp_l / (1.0 + exp_l)
    return out

# 7) Run the sweep and print CSV header
print("step_factor,mean_abs_error,runtime_s")
gt = build_ground_truth(fixed_resolution)

for f in step_factors:
    mapping = Mapping(WIDTH, HEIGHT,
                      fixed_resolution,
                      step_factor=f)

    start = time.perf_counter()
    for (x, y), theta in trajectory:
        mapping.update_with_scan((x, y), theta,
                                 MAX_SENSOR_RANGE,
                                 walls, landmarks)
    runtime = time.perf_counter() - start

    prob  = sigmoid(mapping.log_odds)
    error = np.mean(np.abs(prob - gt))
    print(f"{f},{error:.6f},{runtime:.3f}")
