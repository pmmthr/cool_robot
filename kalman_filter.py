import numpy as np
import math
import pygame


class KalmanFilter:

    # ALL formulas are from the EKF lecture, except for some which are used for normalization
    
    def __init__(self, initial_state, initial_covariance, motion_noise, meas_noise, sensor_range, features):
        self.state = initial_state  
        self.cov = initial_covariance
        self.motion_noise = motion_noise  
        self.meas_noise= meas_noise
        self.sensor_range = sensor_range
        self.features = features  

    def predict(self, control):
        v, ω, dt = control
        x, y, θ = self.state

        # 1) compute noise-free motion
        if abs(ω) > 1e-6:
            r = v / ω
            θ_new = θ + ω*dt
            x_new = x - r*math.sin(θ) + r*math.sin(θ_new)
            y_new = y + r*math.cos(θ) - r*math.cos(θ_new)
        else:
            # limit ω→0: straight line
            θ_new = θ
            x_new = x + v*dt*math.cos(θ)
            y_new = y + v*dt*math.sin(θ)

        θ_new = math.atan2(math.sin(θ_new), math.cos(θ_new))
        self.state = np.array([x_new, y_new, θ_new])

        # 2) compute G = ∂g/∂state
        if abs(ω) > 1e-6:
            G = np.array([
                [1, 0,
                r*(-math.cos(θ) + math.cos(θ_new))],
                [0, 1,
                r*(-math.sin(θ) + math.sin(θ_new))],
                [0, 0, 1]
            ])
        else:
            G = np.array([
                [1, 0, -v*dt*math.sin(θ)],
                [0, 1,  v*dt*math.cos(θ)],
                [0, 0, 1]
            ])

        # 3) compute V = ∂g/∂[v,ω]
        if abs(ω) > 1e-6:
            θd = θ + ω*dt
            V = np.array([
                [(-math.sin(θ) + math.sin(θd))/ω,
                v*(math.sin(θ) - math.sin(θd))/(ω**2)
                + v*dt*math.cos(θd)/ω],
                [( math.cos(θ) - math.cos(θd))/ω,
                -v*(math.cos(θ) - math.cos(θd))/(ω**2)
                + v*dt*math.sin(θd)/ω],
                [0, dt]
            ])
        else:
            V = np.array([
                [dt*math.cos(θ), -0.5*v*dt*dt*math.sin(θ)],
                [dt*math.sin(θ),  0.5*v*dt*dt*math.cos(θ)],
                [0, dt]
            ])

        # 4) motion noise
        α1, α2, α3, α4 = self.motion_noise
        M = np.diag([
            (α1*abs(v) + α2*abs(ω))**2,
            (α3*abs(v) + α4*abs(ω))**2
        ])

        # 5) covariance prediction
        self.cov = G @ self.cov @ G.T + V @ M @ V.T

        return self.state, self.cov

    def update(self, measurement, feature_id):
        feature_position = self.features[feature_id][1]
        
        dx = feature_position[0] - self.state[0]
        dy = feature_position[1] - self.state[1]
        r2 = dx*dx + dy*dy
        if r2 < 1e-10:
            return self.state, self.cov, np.zeros(2)

        # predicted measurement
        r   = math.sqrt(r2)
        phi = math.atan2(dy, dx) - self.state[2]
        phi = math.atan2(math.sin(phi), math.cos(phi))
        z_pred = np.array([r, phi])

        # measurement Jacobian H (2×3)
        H = np.array([
            [-dx / r,  -dy / r, 0 ],
            [ dy / r2, -dx / r2, -1 ]
        ])

        # measurement as range, bearing
        meas_range, meas_bearing = measurement
        z_meas = np.array([meas_range, meas_bearing])

        # innovation
        y = z_meas - z_pred
        y[1] = math.atan2(math.sin(y[1]), math.cos(y[1]))

        # Pred measurement gain
        S = H @ self.cov @ H.T + self.meas_noise
        # gain
        K = self.cov @ H.T @ np.linalg.inv(S)

        # state update
        self.state = self.state + K @ y


        # covariance update
        I = np.eye(3)
        self.cov = (I - K @ H) @ self.cov

        return self.state, self.cov, y

    def detect_features(self, robot_position, robot_angle):
        detected_features = []
        for fid, feature in self.features:
            dx = feature[0] - robot_position[0]
            dy = feature[1] - robot_position[1]
            distance = math.sqrt(dx**2 + dy**2)
            if distance <= self.sensor_range:
                bearing = math.atan2(dy, dx) - robot_angle
                bearing = math.atan2(math.sin(bearing), math.cos(bearing))  # normalize
                detected_features.append((fid, feature, bearing, distance))
        return detected_features

    def draw_covariance_ellipse(self, screen, color=(0, 255, 0), scale=50.0):

        position = self.state[:2]
        pos_cov = self.cov[:2, :2] * scale
        eigenvals, eigenvecs = np.linalg.eig(pos_cov)

        order = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[order]
        eigenvecs = eigenvecs[:, order]

        major_axis = 2 * math.sqrt(eigenvals[0])
        minor_axis = 2 * math.sqrt(eigenvals[1])
        angle = math.degrees(math.atan2(eigenvecs[1, 0], eigenvecs[0, 0]))

        pygame.draw.ellipse(screen, color,
                            (position[0] - major_axis / 2,
                             position[1] - minor_axis / 2,
                             major_axis, minor_axis), 1)

        end_x = position[0] + major_axis / 2 * math.cos(math.radians(angle))
        end_y = position[1] + major_axis / 2 * math.sin(math.radians(angle))
        pygame.draw.line(screen, color, position, (end_x, end_y), 1)
