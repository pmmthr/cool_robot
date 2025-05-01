import numpy as np
from constants import *

def kalman_filter(est, sigma, velocity, obs):
    r_x, r_y, r_angle = est
    mu = np.array([r_x, r_y, r_angle]).T
    u = velocity

    A = np.eye(3)
    B = np.array([np.cos(np.radians(r_angle)), np.sin(np.radians(r_angle)), 0]).T
    mu_pred = A @ mu + B * u

    R = np.diag([SR_X**2, SR_Y**2, SR_THETA**2])
    sigma_pred = A @ sigma @ A.T + R

    C = np.eye(3)
    Q = np.diag([SQ_X**2, SQ_Y**2, SQ_THETA**2])
    K = sigma_pred @ C.T @ np.linalg.inv(C @ sigma_pred @ C.T + Q)

    if obs == None:
        mu_pred = np.random.multivariate_normal(mu_pred, sigma_pred, 1)
        return (mu_pred[0, 0], mu_pred[0, 1], mu_pred[0, 2]), sigma_pred
    
    z_x, z_y, z_angle = obs
    z = np.array([z_x, z_y, z_angle]).T
    mu = mu_pred + K @ (z - C @ mu_pred)
    sigma = (np.eye(3) - K @ C) @ sigma_pred

    mu = np.random.multivariate_normal(mu, sigma, 1)
    return (mu[0, 0], mu[0, 1], mu[0, 2]), sigma
