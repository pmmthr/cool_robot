import numpy as np
import math
import pygame



class KalmanFilter:

    # All the formulas are taken from lecture 8, Extended kalman filter slides.
    def __init__(self, initial_state, initial_covariance, motion_noise, meas_noise, sensor_range, features):
        self.state = initial_state  
        self.cov = initial_covariance
        self.motion_noise = motion_noise  
        self.meas_noise= meas_noise
        self.sensor_range = sensor_range
        self.features = features  

    def predict(self, control):
        v, w, dt = control
        x, y, theta = self.state
        
        # Avoid division by zero: handle straight-line motion separately (Folie 17, Noise-free Velocity Model)
        if abs(w) < 1e-6:
            # Prediction (straight motion) – Slide 17, eq. (1)
            theta_new = theta
            x_new = x + v * dt * math.cos(theta)
            y_new = y + v * dt * math.sin(theta)

            # Jacobian G = ∂g/∂x for straight motion (linearized) – Slide 17
            G = np.array([
                [1, 0, -v * dt * math.sin(theta)],
                [0, 1,  v * dt * math.cos(theta)],
                [0, 0, 1]
            ])

            # Jacobian V = ∂g/∂u for straight motion (linearized) – derived from Slide 17
            V = np.array([
                [dt * math.cos(theta), -0.5 * v * dt**2 * math.sin(theta)],
                [dt * math.sin(theta),  0.5 * v * dt**2 * math.cos(theta)],
                [0,                  dt]
            ])
        else:
            # Circular motion – Slide 16, eqs. (2-3)
            r = v / w
            theta_new = theta + w * dt
            # Normalize angle
            theta_new = math.atan2(math.sin(theta_new), math.cos(theta_new))
            x_new = x - r * math.sin(theta) + r * math.sin(theta_new)
            y_new = y + r * math.cos(theta) - r * math.cos(theta_new)

            # Jacobian G = ∂g/∂x for circular motion – Slide 16, eq. (G)
            G = np.array([
                [1, 0, r * (-math.cos(theta) + math.cos(theta_new))],
                [0, 1, r * (-math.sin(theta) + math.sin(theta_new))],
                [0, 0, 1]
            ])

            # Jacobian V = ∂g/∂u for circular motion – Slide 16, eq. (V)
            V = np.array([
                [(-math.sin(theta) + math.sin(theta_new)) / w,
                 (v * (math.sin(theta) - math.sin(theta_new))) / (w**2)],
                [( math.cos(theta) - math.cos(theta_new)) / w,
                 (-v * (math.cos(theta) - math.cos(theta_new))) / (w**2)],
                [0, dt]
            ])

        # Update state
        self.state = np.array([x_new, y_new, theta_new])

        # Motion noise covariance M – Slide 16, eq. (M)
        alpha1, alpha2, alpha3, alpha4 = self.motion_noise
        M = np.diag([
            (alpha1 * abs(v) + alpha2 * abs(w))**2,
            (alpha3 * abs(v) + alpha4 * abs(w))**2
        ])

        # Covariance update – Slide 16, eq. (6)
        self.cov = G @ self.cov @ G.T + V @ M @ V.T

        return self.state, self.cov



    def update(self, measurement, feature_id):
        feature_position = self.features[feature_id][1]
        
        _, landmark = self.features[feature_id]
        fx, fy     = float(landmark.center[0]), float(landmark.center[1])
        
        dx = fx - self.state[0]
        dy = fy - self.state[1]
        
        r2 = dx*dx + dy*dy
        if r2 < 1e-10:
            return self.state, self.cov, np.zeros(2)

        # EQUATION 7 SLIDE 15 EKF
        r   = math.sqrt(r2) # top of matrix of equation 7 on slide 15 (EKF)
        phi = math.atan2(dy, dx) - self.state[2] #the bottom of equation 7
        phi = math.atan2(math.sin(phi), math.cos(phi)) #normalizing
        z_pred = np.array([r, phi]) #EQUATION 7 EKF slide 15 (zt)

        # Equation 8 slide 15 (Ht). r is with squareroot. r2 is without. top row is with respect to
        H = np.array([
            [-dx / r,  -dy / r, 0 ],
            [ dy / r2, -dx / r2, -1 ]
        ])

        meas_range, meas_bearing = measurement
        z_meas = np.array([meas_range, meas_bearing])

        y = z_meas - z_pred
        y[1] = math.atan2(math.sin(y[1]), math.cos(y[1]))

        # Equation 10 slide 15 EKF: Pred. measurement covariance
        S = H @ self.cov @ H.T + self.meas_noise
        # Equation 11 slide 15 EKF: Kalman gain
        K = self.cov @ H.T @ np.linalg.inv(S)

        # Equation 12 slide 15 EKF (updated mean)
        self.state = self.state + K @ y 


        I = np.eye(3)
        # Equation 13 slide 15 EKF (updated covariance)
        self.cov = (I - K @ H) @ self.cov

        return self.state, self.cov, y

    def detect_features(self, features, robot_position, robot_angle):
        """
        features: list of (fid, Landmark), where Landmark has .center=(fx,fy) and .radius
        robot_position: (rx, ry)
        robot_angle: heading in radians
        """
        detected = []
        rx, ry = robot_position

        for fid, landmark in features:
            fx, fy = landmark.center
            # Vector from robot to landmark center
            dx = fx - rx
            dy = fy - ry
            distance = math.hypot(dx, dy)
            if distance <= self.sensor_range:
                raw_bearing = math.atan2(dy, dx) - robot_angle
                bearing = math.atan2(math.sin(raw_bearing),
                                    math.cos(raw_bearing))

                detected.append((fid, (fx, fy), bearing, distance))

        return detected

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


    def draw_dotted_line(screen, start, end, color, step=10):
        """
        Draw a dashed line from start to end.
        Aborts early if the distance is NaN/Inf or too small.
        """
        x1, y1 = start
        x2, y2 = end
        dx, dy = x2 - x1, y2 - y1
        distance = math.hypot(dx, dy)
        

        # Bail out if distance is not a finite number or is essentially zero
        if not math.isfinite(distance) or distance < 1e-6:
            return

        ux, uy = dx / distance, dy / distance
        dash = step
        gap = step

        pos = 0.0
        while pos < distance:
            start_pos = pos
            end_pos = min(pos + dash, distance)

            p1 = (int(x1 + ux * start_pos), int(y1 + uy * start_pos))
            p2 = (int(x1 + ux * end_pos),   int(y1 + uy * end_pos))
            pygame.draw.line(screen, color, p1, p2, 1)

            pos += dash + gap