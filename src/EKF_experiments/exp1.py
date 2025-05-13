import math
import numpy as np
import matplotlib.pyplot as plt
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from constants import *
from kalman_filter import KalmanFilter
from maze import init_landmarks, init_maze

# Load the recorded trace
trace = []
with open("traceEKF.txt") as f:
    for line in f:
        x, y, deg = map(float, line.strip().split(","))
        trace.append(((x, y), math.radians(deg)))

# World setup
maze_walls = init_maze()
landmarks  = init_landmarks()   # [(id, Landmark), ...]

# Make three EKFs (for k=1,2,3)
kf_filters = {
    k: KalmanFilter(
        INITIAL_STATE.copy(),
        INITIAL_COVARIANCE.copy(),
        MOTION_NOISE,
        MEASUREMENT_NOISE,
        MAX_SENSOR_RANGE,
        landmarks
    )
    for k in (1, 2, 3)
}
# storage for per-step errors
errors = {1: [], 2: [], 3: []}

# Replay the trace
prev_pose, prev_theta = trace[0]
for i, ((x, y), theta) in enumerate(trace):
    if i == 0:
        # record initial‐state error
        for k, kf in kf_filters.items():
            ex, ey, _ = kf.state
            errors[k].append(math.hypot(ex - x, ey - y))
        continue

    dx = x - prev_pose[0]
    dy = y - prev_pose[1]
    dist = math.hypot(dx, dy)
    v    = dist               # assume dt units of 1
    dθ   = theta - prev_theta
    # wrap to [-π, π]
    dθ   = (dθ + math.pi) % (2*math.pi) - math.pi
    w    = dθ
    dt   = 1.0

    # predict step for all filters
    for kf in kf_filters.values():
        kf.predict(np.array([v, w, dt]))

    # detect features ONCE using the true pose
    detected = kf_filters[1].detect_features(
        landmarks,
        (x, y),
        theta
    )
    detected.sort(key=lambda f: f[3])  # sort by distance

    # update each filter with only k nearest
    for k, kf in kf_filters.items():
        for fid, feat, bearing, dist_obs in detected[:k]:
            dm  = dist_obs 
            bm  = bearing
            kf.update(np.array([dm, bm]), fid)

        # record Euclidean error at this step
        ex, ey, _ = kf.state
        errors[k].append(math.hypot(ex - x, ey - y))

    prev_pose, prev_theta = (x, y), theta

# Compute & plot mean error vs. number of detected features
mean_err = {k: np.mean(errors[k]) for k in errors}
print(mean_err)
ks   = sorted(mean_err.keys())
vals = [mean_err[k] for k in ks]

plt.figure()
plt.plot(ks, vals, marker='o')
plt.xticks(ks)
plt.xlabel('Number of landmarks')
plt.ylabel('Mean localization error')
plt.title('Localization Error vs. number of Detected Landmarks')
plt.grid(True)
plt.tight_layout()
plt.show()
