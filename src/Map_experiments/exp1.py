import math
import numpy as np
import matplotlib.pyplot as plt

import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


from constants import WIDTH, HEIGHT, MAX_SENSOR_RANGE
from mapping   import Mapping
from maze      import init_maze, init_landmarks


resolutions = [5, 10, 20, 30]

#  Load the recorded trajectory
trajectory = []
with open("src/trace2.txt") as f:
    for line in f:
        x, y, deg = map(float, line.strip().split(","))
        trajectory.append(((x, y), math.radians(deg)))


# initialize
walls     = init_maze()
landmarks = init_landmarks()


# Build ground truth 
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
                for _, lm in landmarks:
                    lx, ly = lm.center
                    if (cx - lx)**2 + (cy - ly)**2 <= lm.radius**2:
                        occupied = True
                        break

            gt[r, c] = 1 if occupied else 0

    return gt

# logodds to probability
def sigmoid(l):
    # l is a NumPy array
    out = np.empty_like(l, dtype=float)
    pos = l >= 0
    neg = l<0
    out[pos] = 1.0 / (1.0 + np.exp(-l[pos]))
    exp_l   = np.exp(l[neg])
    out[neg] = exp_l / (1.0 + exp_l) 
    return out

#get results
print("resolution_px,mean_abs_error")
errors=[]
for res in resolutions:
    mapping = Mapping(WIDTH, HEIGHT, res)
    gt = build_ground_truth(res)

    for (x, y), theta in trajectory:
        mapping.update_with_scan((x, y), theta, MAX_SENSOR_RANGE, walls, landmarks)

    # convert log-odds to probability with sigmoid
    prob = sigmoid(mapping.log_odds)

    # so observed[r,c] is true if its not grey basically (not scanned)
    observed = (mapping.log_odds != mapping.l0)

    # compute error only on those cells
    if observed.any():
        error = np.mean(np.abs(prob[observed] - gt[observed]))
    else:
        error = float('nan')
    print(f"{res},{error:.6f}")
    errors.append(error)
    
plt.figure()
plt.plot(resolutions, errors, marker='o')
plt.xlabel('Resolution (px per cell)')
plt.ylabel('Mean Absolute Error')
plt.title('Mapping Error vs Grid Resolution')
plt.grid(True)
plt.tight_layout()
plt.show()