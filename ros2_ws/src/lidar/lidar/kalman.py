import numpy as np

class KalmanFilter:
    # State: [x, y, z, vx, vy, vz, ax, ay, az]^T
    # Measurement: [x, y, z]^T

    def __init__(self, dt=0.1, measurement_var=0.002, process_var=0.5):
        self.dt = dt
        self.process_var = process_var

        #state vector: [x, y, z, vx, vy, vz, ax, ay, az]^T
        self.x = np.zeros((9, 1))

        #state covariance matrix
        self.P = np.eye(9) * 1.0

        #state transition matrix
        self.F = np.eye(9)

        #process noise covariance
        self.Q = np.eye(9)

        #measurement matrix (only measure positions)
        self.H = np.zeros((3, 9))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        #measurement noise covariance
        self.R = np.eye(3) * measurement_var

        self.set_dt(dt)

    def set_dt(self, dt):
        self.dt = max(float(dt), 1e-4)
        dt = self.dt

        self.F = np.eye(9)

        for i in range(3):
            self.F[i, i + 3] = dt
            self.F[i, i + 6] = 0.5 * dt * dt
            self.F[i + 3, i + 6] = dt

        q = self.process_var

        q_pos = 0.25 * dt ** 4 * q
        q_vel = dt ** 2 * q
        q_acc = q

        self.Q = np.eye(9)
        self.Q[0:3, 0:3] *= q_pos
        self.Q[3:6, 3:6] *= q_vel
        self.Q[6:9, 6:9] *= q_acc

    def predict(self):
        self.x = self.F @ self.x #x : predicted state
        self.P = self.F @ self.P @ self.F.T + self.Q #P : covariance matrix of the predicted state
        return self.x[0:3, 0].copy() #predicted position

    def update(self, measurement):
        z = np.array(measurement).reshape(3, 1) #z: measurement [x, y, z]
        innov = z - self.H @ self.x  #innovation
        S = self.H @ self.P @ self.H.T + self.R  #S : innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  #Kalman gain
        self.x = self.x + K @ innov #updated state (pos)
        self.P = (np.eye(9) - K @ self.H) @ self.P #P : covariance matrix
        return self.x[0:3, 0].copy() #updated position

    def set_state(self, position):
        position = np.array(position).reshape(3)
        self.x[:] = 0.0
        self.x[0:3, 0] = position

        self.P = np.eye(9) * 1.0
        self.P[0:3, 0:3] *= 0.05
        self.P[3:6, 3:6] *= 1.0
        self.P[6:9, 6:9] *= 2.0

    def mahalanobis_distance_squared(self, measurement):
        z = np.array(measurement).reshape(3, 1)
        y = z - self.H @ self.x #innovation
        S = self.H @ self.P @ self.H.T + self.R #innovation covariance  
        result = y.T @ np.linalg.inv(S) @ y #squared Mahalanobis, how many squared standard deviation away
        return result.item()