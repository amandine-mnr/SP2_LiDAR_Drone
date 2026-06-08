import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate

LIDAR_POS = np.array([0.5, 1.5, 0.5])

def estimate_delay(time, true_pos, est_pos):

    dt = np.mean(np.diff(time))
    best_lags = []

    for dim in [0, 1]:  # x and y
        true_signal = true_pos[:, dim] - np.mean(true_pos[:, dim])
        est_signal = est_pos[:, dim] - np.mean(est_pos[:, dim])

        corr = correlate(est_signal, true_signal, mode='full')
        lags = np.arange(-len(true_signal) + 1, len(true_signal))

        best_lag = lags[np.argmax(corr)]
        best_lags.append(best_lag)

    best_lag_mean = int(np.round(np.mean(best_lags)))
    delay_sec = best_lag_mean * dt
    best_delays = np.array(best_lags) * dt

    return best_lag_mean, delay_sec, best_delays

#load CSV
# data = np.loadtxt("metrics_log_grid3/drone_log_streamtrack_2026-06-02_15h00m10s.csv", delimiter=",", skiprows=20)
data = np.loadtxt("results/metrics_log_g1/drone_log_streamtrack_2026-06-06_19h55m19s.csv", delimiter=",", skiprows=10)

#read columns
time = data[:, 0]
true_pos = data[:, 1:4]
est_pos = data[:, 4:7]
est_vel = data[:, 7:10]
est_acc = data[:, 10:13]
true_vel = np.zeros((len(true_pos),3))
true_acc = np.zeros((len(true_pos),3))

# Compute true velocity and true acceleration
for i in range(1, len(true_pos)-1):
    dt = time[i+1] - time[i-1]
    if dt > 0:
        true_vel[i] = (true_pos[i+1] - true_pos[i-1]) / dt

    dt1 = time[i] - time[i-1]
    dt2 = time[i+1] - time[i]
    if dt1 > 0 and dt2 > 0:
        true_acc[i] = ((true_pos[i+1] - true_pos[i])/dt2 - (true_pos[i] - true_pos[i-1])/dt1) / ((dt1 + dt2)/2)


# Position errors
errors = np.linalg.norm(true_pos - est_pos, axis=1)
VALID_THRESHOLD = 1.0  #meters
valid_mask = errors <= VALID_THRESHOLD
lost_mask = ~valid_mask
valid_errors = errors[valid_mask]
rmse = np.sqrt(np.mean(valid_errors**2)) if len(valid_errors) else float("nan")
mean_err = np.mean(valid_errors) if len(valid_errors) else float("nan")
max_err = np.max(valid_errors) if len(valid_errors) else float("nan")
print(f"RMSE: {rmse:.3f} m")
print(f"Mean error: {mean_err:.3f} m")
print(f"Max error: {max_err:.3f} m")
#axis-wise error
axis_error = np.abs(true_pos - est_pos)
mean_axis_error = np.mean(axis_error[valid_mask], axis=0)
print(f"Mean axis-wise error (x,y,z): {mean_axis_error}")

# Lost metric
lost_counter = int(np.sum(lost_mask))
dt_array = np.diff(time, prepend=time[0])
lost_time = np.sum(dt_array[lost_mask])
print(f"Lost frames: {lost_counter}")
print(f"Lost time: {lost_time:.3f} s")

# Velocity and acceleration error
speed = np.linalg.norm(true_vel[1:-1], axis=1)
mean_true_vel = np.mean(speed)
print(f"Mean drone speed: {mean_true_vel:.3f} m/s")
vel_error = np.linalg.norm(true_vel - est_vel, axis=1)
acc_error = np.linalg.norm(true_acc - est_acc, axis=1)
vel_rmse = np.sqrt(np.mean(vel_error[valid_mask]**2))
acc_rmse = np.sqrt(np.mean(acc_error[valid_mask]**2))
print(f"Velocity RMSE: {vel_rmse:.3f} m/s")
print(f"Acceleration RMSE: {acc_rmse:.3f} m/s²")

# Mean distance to LiDAR
dist_to_lidar = np.sqrt(
    (true_pos[:,0]- LIDAR_POS[0]) ** 2
    + (true_pos[:,1] - LIDAR_POS[1]) ** 2
    + (true_pos[:,2] - LIDAR_POS[2]) ** 2
)
mean_dist_to_lidar = float(np.mean(dist_to_lidar))
print(f"Mean distance to LiDAR: {mean_dist_to_lidar:.3f} m")

# Delay
best_lag, delay, per_axis_delay = estimate_delay(time, true_pos, est_pos)
print(f"Estimated delay: {delay:.3f} s")
print(f"Delay for x and y axis: {per_axis_delay}")

# -----------------------------
# Plots

# 3D Trajectory
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(121, projection='3d')
ax.plot(true_pos[:,0], true_pos[:,1], true_pos[:,2], label='Ground truth', c='blue')
ax.plot(est_pos[:,0], est_pos[:,1], est_pos[:,2], label='Estimated', c='red', linestyle='--')
ax.set_title("3D Trajectory")
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
ax.set_xlim(-1.8, 1.8)
ax.set_ylim(-1.8, 1.8)
ax.set_zlim(-1.8, 1.8)
ax.legend()
ax.grid(True)

# Position error over time
ax2 = fig.add_subplot(122)
ax2.plot(time, errors, label='Position Error', color='magenta')
ax2.set_title("Position Error over Time")
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Error [m]")
ax2.grid(True)
ax2.legend()

# Velocity plot
plt.figure()

plt.subplot(3,1,1)
plt.plot(time, true_vel[:,0], label="True Vx")
plt.plot(time, est_vel[:,0], '--', label="Est Vx")
plt.legend(); plt.grid(); plt.title("Velocity X")
plt.ylabel("v_x [m/s]")

plt.subplot(3,1,2)
plt.plot(time, true_vel[:,1])
plt.plot(time, est_vel[:,1], '--')
plt.grid(); plt.title("Velocity Y")
plt.ylabel("v_y [m/s]")

plt.subplot(3,1,3)
plt.plot(time, true_vel[:,2])
plt.plot(time, est_vel[:,2], '--')
plt.grid(); plt.title("Velocity Z")
plt.ylabel("v_z [m/s]")

plt.tight_layout(h_pad=2.0, w_pad=2.0)

# Acceleration plot
plt.figure()

plt.subplot(3,1,1)
plt.plot(time, true_acc[:,0], label="True Ax")
plt.plot(time, est_acc[:,0], '--', label="Est Ax")
plt.legend(); plt.grid(); plt.title("Acceleration X")
plt.ylabel("a_x [m/s²]")

plt.subplot(3,1,2)
plt.plot(time, true_acc[:,1])
plt.plot(time, est_acc[:,1], '--')
plt.grid(); plt.title("Acceleration Y")
plt.ylabel("a_y [m/s²]")

plt.subplot(3,1,3)
plt.plot(time, true_acc[:,2])
plt.plot(time, est_acc[:,2], '--')
plt.grid(); plt.title("Acceleration Z")
plt.ylabel("a_z [m/s²]")

plt.tight_layout(h_pad=2.0, w_pad=2.0)

# Position/time plots
plt.figure()

plt.subplot(3,1,1)
plt.plot(time, true_pos[:,0], label="True X")
plt.plot(time, est_pos[:,0], '--', label="Est X")
plt.legend(); plt.grid(); plt.title("Position X")
plt.ylabel("x [m]")

plt.subplot(3,1,2)
plt.plot(time, true_pos[:,1], label="True Y")
plt.plot(time, est_pos[:,1], '--', label="Est Y")
plt.legend(); plt.grid(); plt.title("Position Y")
plt.ylabel("y [m]")

plt.subplot(3,1,3)
plt.plot(time, true_pos[:,2], label="True Z")
plt.plot(time, est_pos[:,2], '--', label="Est Z")
plt.legend(); plt.grid(); plt.title("Position Z")
plt.ylabel("z [m]")

plt.xlabel("Time [s]")


plt.tight_layout(h_pad=2.0, w_pad=2.0)

# # Delay realigned plots
# est_x_corr = np.roll(est_pos[:, 0], -best_lag)
# est_y_corr = np.roll(est_pos[:, 1], -best_lag)
# plt.figure(figsize=(10, 6))
# # x vs time
# plt.subplot(2,1,1)
# plt.plot(time, true_pos[:,0], label="True X", c="blue")
# plt.plot(time, est_pos[:,0], ':', label="Estimated X (raw)", c="green")
# plt.plot(time, est_x_corr, '--', label="Estimated X (aligned)", c="red")
# plt.title("X(t) Alignment")
# plt.ylabel("X [m]")
# plt.grid()
# plt.legend()
# # y vs time
# plt.subplot(2,1,2)
# plt.plot(time, true_pos[:,1], label="True Y", c="blue")
# plt.plot(time, est_pos[:,1], ':', label="Estimated Y (raw)", c="green")
# plt.plot(time, est_y_corr, '--', label="Estimated Y (aligned)", c="red")
# plt.title("Y(t) Alignment")
# plt.xlabel("Time [s]")
# plt.ylabel("Y [m]")
# plt.grid()
# plt.legend()


# plt.tight_layout()
plt.tight_layout(h_pad=2.0, w_pad=2.0)
plt.show()