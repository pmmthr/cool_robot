
import math
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd


from constants import WIDTH, HEIGHT, MAX_SENSOR_RANGE
from mapping   import Mapping
from maze      import init_maze, init_landmarks

# Fixed grid resolution (px per cell)
fixed_resolution =5

# Step‑size (fraction of cell size) where its resolution * stepsize
step_factors = [1/8, 1/4, 1/2, 1.0]

#Load the recorded trajectory
trajectory = []
with open("trace2.txt") as f:
    for line in f:
        x, y, deg = map(float, line.strip().split(","))
        trajectory.append(((x, y), math.radians(deg)))

# Initialize walls & landmarks 
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
                for lm in landmarks:
                    lx, ly = lm.center
                    if (cx - lx)**2 + (cy - ly)**2 <= lm.radius**2:
                        occupied = True
                        break

            gt[r, c] = 1 if occupied else 0

    return gt

def sigmoid(l):
    out = np.empty_like(l, dtype=float)
    pos = l >= 0
    neg = l<0
    out[pos] = 1.0 / (1.0 + np.exp(-l[pos])) 
    exp_l   = np.exp(l[neg])
    out[neg] = exp_l / (1.0 + exp_l)
    return out

#get results
print("step_factor,mean_abs_error,runtime_s")
gt = build_ground_truth(fixed_resolution)
results = {'step_factor': [], 'error': [], 'runtime_s': []}

for f in step_factors:
    mapping = Mapping(WIDTH, HEIGHT,resolution=fixed_resolution,step_factor=f)

    start = time.perf_counter()
    for (x, y), theta in trajectory:
        mapping.update_with_scan((x, y), theta, MAX_SENSOR_RANGE, walls, landmarks)
    runtime = time.perf_counter() - start

    prob  = sigmoid(mapping.log_odds)
    
    observed = (mapping.log_odds != mapping.l0)
    # compute error only on those cells
    if observed.any():
        error = np.mean(np.abs(prob[observed] - gt[observed]))
    else:
        error = float('nan')
    print(f"{f},{error:.6f},{runtime:.3f}")

    results['step_factor'].append(f)
    results['error'].append(error)
    results['runtime_s'].append(runtime)

df = pd.DataFrame(results)


# Plot Runtime vs Step Factor
plt.figure()
plt.plot(df['step_factor'], df['runtime_s'], marker='o')
plt.xlabel('Step Factor')
plt.ylabel('Runtime (s)')
plt.title('Mapping Runtime vs Step Size')
plt.grid(True)
plt.tight_layout()
plt.show()

