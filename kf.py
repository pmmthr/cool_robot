import numpy as np

def kalman_filter(mu, u, z):
	A = np.eye(2)
	B = np.array([[], [], []])