import numpy as np
from constants import *

def kalman_filter(pos, sigma, velocity, obs):
    r_x, r_y, r_angle = pos
    mu = np.array([r_x, r_y]).T
    u = velocity
    z_x, z_y = obs
    z = np.array([z_x, z_y]).T

    A = np.eye(2)
    B = np.array([np.cos(np.radians(r_angle)), np.sin(np.radians(r_angle))]).T

    sr_x, sr_y, sr_theta = SR_X, SR_Y, SR_THETA
    sq_x, sq_y, sq_theta = SQ_X, SQ_Y, SQ_THETA
    R = np.diag([sr_x**2, sr_y**2])
    Q = np.diag([sq_x**2, sq_y**2])

    mu_pred = A @ mu + B * u
    sigma_pred = A @ sigma @ A.T + R

    C = np.eye(2)
    K = sigma_pred @ C.T @ np.linalg.inv(C @ sigma_pred @ C.T + Q)

    mu = mu_pred + K @ (z - C @ mu_pred)
    sigma = (np.eye(2) - K @ C) @ sigma_pred

    mu = np.random.multivariate_normal(mu, sigma, 1)
    return mu[0, 0], mu[0, 1], sigma
