import os
import numpy as np
from datetime import datetime
from collections import deque
import rclpy
from rclpy.node import Node

from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree

import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker
import time

from lidar.kalman import *

USE_KALMAN = True

def get_timestamped_log_filename(base_name="drone_log_streamtrack", ext=".csv", folder="."):
    timestamp = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
    filename = f"{base_name}_{timestamp}{ext}"
    return os.path.join(folder, filename)

class DroneLocalizer(Node):

    def __init__(self):
        super().__init__('drone_localizer')

        # Parameters
        self.declare_parameter("lidar_x", 0.0)
        self.declare_parameter("lidar_y", 2.0)
        self.declare_parameter("lidar_z", 0.5)

        self.declare_parameter("lidar_noise_std", 0.02)
        self.declare_parameter("bg_threshold", 0.15)

        self.declare_parameter("n_background_frames", 10)

        self.declare_parameter("drone_size", 0.092)
        self.declare_parameter("size_margin", 0.1)

        self.declare_parameter("dbscan_min_samples", 1)

        self.declare_parameter("roi_min_radius", 0.4)
        self.declare_parameter("roi_max_radius", 2.5)
        self.declare_parameter("roi_factor", 1.5)

        self.declare_parameter("memory_len", 5)
        self.declare_parameter("history_predict_min", 3)
        self.declare_parameter("mahalanobis_gate", 11.34)

        # Load parameters

        self.LIDAR_X = self.get_parameter("lidar_x").value
        self.LIDAR_Y = self.get_parameter("lidar_y").value
        self.LIDAR_Z = self.get_parameter("lidar_z").value

        self.LIDAR_NOISE_STD = self.get_parameter("lidar_noise_std").value
        self.BG_THRESHOLD = self.get_parameter("bg_threshold").value

        self.N_BACKGROUND_FRAMES = self.get_parameter("n_background_frames").value

        self.DRONE_SIZE = self.get_parameter("drone_size").value
        self.SIZE_MARGIN = self.get_parameter("size_margin").value

        self.DBSCAN_EPS = self.DRONE_SIZE + self.SIZE_MARGIN
        self.DBSCAN_MIN_SAMPLES = self.get_parameter("dbscan_min_samples").value

        self.ROI_MIN_RADIUS = self.get_parameter("roi_min_radius").value
        self.ROI_MAX_RADIUS = self.get_parameter("roi_max_radius").value
        self.ROI_FACTOR = self.get_parameter("roi_factor").value

        self.MEMORY_LEN = self.get_parameter("memory_len").value
        self.HISTORY_PREDICT_MIN = self.get_parameter("history_predict_min").value
        self.MAHALANOBIS_GATE = self.get_parameter("mahalanobis_gate").value

        # Log file
        log_dir = "ros2_ws/src/lidar/lidar/metrics_log"
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = get_timestamped_log_filename(folder=log_dir)
        self.get_logger().info(f"Log file: {self.log_file}")
        self.log = open(self.log_file, "w")
        self.log.write("time,true_x,true_y,true_z,est_x,est_y,est_z,vel_x,vel_y,vel_z,acc_x,acc_y,acc_z,cluster_size,roi,lost\n")
        self.log.flush()

        # Publisher : Localization ready
        self.ready_publisher = self.create_publisher(Bool, '/localizer_ready', 10)
        self.is_ready = False
        self.shutdown_flag = False

        # Suscriber : Lidar point cloud
        self.subscription = self.create_subscription(PointCloud2, '/my_3D_lidar/points', self.callback, 1)

        # Publisher : Drone position
        self.publisher = self.create_publisher(PointStamped, '/drone_position', 1)

        # Publisher : ROI marker
        self.roi_publisher = self.create_publisher(Marker, '/roi_marker', 1)

        # Suscriber : System shutdown
        self.shutdown_sub = self.create_subscription(Bool, '/system_shutdown', self.shutdown_callback, 1)

        # Suscriber : Odometry
        self.sub = self.create_subscription(Odometry, '/model/crazyflie/odometry', self.odometry_callback, 1)

        # Background
        self.background_frames = []
        self.background = None
        self.background_initialized = False
        self.kdtree = None

        # Motion variables
        self.prev_time = None
        self.velocity = np.zeros(3)
        self.acceleration = np.zeros(3)

        # ROI prediction
        self.roi_radius = 0.5

        # Poses
        self.true_pose = None
        self.est_pose = None

        # Memory
        self.memory = deque(maxlen=self.MEMORY_LEN)
        self.lost_counter = 0

        # Kalman filter
        measurement_var = (self.LIDAR_NOISE_STD ** 2) * 2.0
        self.kf = KalmanFilter(dt=0.1, measurement_var=measurement_var, process_var=0.8)
        self.kf_initialized = False

        self.get_logger().info("Localization node started")
        if USE_KALMAN:
            self.get_logger().info("With Kalman filter")
        else:
            self.get_logger().info("Without Kalman filter")

    def shutdown_callback(self, msg):
        if msg.data:
            self.get_logger().info("System shutdown received, stopping localizer")
            self.shutdown_flag = True
            if hasattr(self, "log") and not self.log.closed:
                self.log.flush()
                self.log.close()

    # Odometry
    def odometry_callback(self, msg):
        if self.shutdown_flag:
            return
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        self.true_pose = (x, y, z)

    # PointCloud conversion
    def convert_pointcloud(self, msg):
        
        points = pc2.read_points_numpy(
            msg,
            field_names=("x", "y", "z"),
            skip_nans=True
        ).astype(np.float32, copy=False)


        if len(points) == 0:
            return None

        points = points[np.isfinite(points).all(axis=1)]

        if len(points) == 0:
            return None

        # Add noise
        points += np.random.normal(0, self.LIDAR_NOISE_STD, points.shape)

        return points

    # Background initialization
    def initialize_background(self, points, current_time):

        self.background_frames.append(points)

        self.get_logger().info(f"Building background: {len(self.background_frames)}/{self.N_BACKGROUND_FRAMES}")

        if len(self.background_frames) < self.N_BACKGROUND_FRAMES:
            self.prev_time = current_time
            return False
        self.background = np.vstack(self.background_frames)
        self.kdtree = cKDTree(self.background)
        self.background_initialized = True
        self.prev_time = current_time

        ready_msg = Bool()
        ready_msg.data = True
        self.ready_publisher.publish(ready_msg)
        self.is_ready = True
        self.get_logger().info("Background initialized")
        return True

    # Prediction
    def predict_from_memory(self, dt):
        if USE_KALMAN and self.kf_initialized:
            kf_pred = self.kf.predict()
        else:
            kf_pred = None
        if len(self.memory) < self.HISTORY_PREDICT_MIN:
            return kf_pred
        pos = np.array([m["pos"] for m in self.memory])
        t = np.array([m["t"] for m in self.memory])
        t = t - t[-1]
        target_t = max(dt if dt is not None else 0.0, 0.0)
        poly_pred = []
        deg = 2
        for d in range(3):
            coeff = np.polyfit(t, pos[:, d], deg) #fit polynome
            poly_pred.append(np.polyval(coeff, target_t)) #evaluate at target t
        poly_pred = np.array(poly_pred)
        if kf_pred is None:
            return poly_pred
        return 0.65 * kf_pred + 0.35 * poly_pred #fuse Kalman prediction and polynome prediction
    
    # ROI radius
    def compute_roi_radius(self):
        if self.kf_initialized:
            P_pos = self.kf.P[:3, :3]
            r = self.ROI_FACTOR * np.sqrt(np.trace(P_pos))
        else:
            r = 0.5 + 0.8 * np.linalg.norm(self.velocity)
        self.roi_radius = float(np.clip(r, self.ROI_MIN_RADIUS, self.ROI_MAX_RADIUS))
    
    # ROI around predicition
    def roi_filter(self, points, predicted, tree=None):
        self.compute_roi_radius()
        if predicted is None or self.lost_counter > 4:
            self.roi_radius = 0.0
            return points
        if tree is None:
            tree = cKDTree(points)
        idx = tree.query_ball_point(predicted, self.roi_radius)
        return points[idx] if len(idx) else np.empty((0, 3))

    # Find moving points
    def subtract_background(self, points, predicted_position):

        if len(points) == 0:
            return np.empty((0,3))

        bg_roi = self.roi_filter(self.background, predicted_position, self.kdtree)

        if len(bg_roi) == 0:
            return points

        bg_tree = cKDTree(bg_roi)
        distances, _ = bg_tree.query(points, k=1)
        moving_points = points[distances > self.BG_THRESHOLD]
        # self.get_logger().info(f"Len moving points : {len(moving_points)}")
        return moving_points

    # Clustering
    def get_clusters(self, moving_points):
        if len(moving_points) < self.DBSCAN_MIN_SAMPLES:
            # self.get_logger().info("Not enough points for clustering")
            return None

        clustering = DBSCAN(eps=self.DBSCAN_EPS, min_samples=self.DBSCAN_MIN_SAMPLES).fit(moving_points)
        labels = clustering.labels_
        clusters = []
        for label in set(labels):
            if label == -1:
                continue

            cluster = moving_points[labels == label]
            center = cluster.mean(axis=0)
            spread = float(np.mean(np.linalg.norm(cluster - center, axis=1)))
            dists = np.linalg.norm(cluster - center, axis=1)
            max_radius = float(np.max(dists))
            max_diameter = 2 * max_radius

            #reject clusters larger than expected drone size
            if (max_diameter > (self.DRONE_SIZE + self.SIZE_MARGIN)):
                # self.get_logger().info("Reject cluster: size too large")
                continue

            clusters.append({"points": cluster, "center": center, "size": len(cluster), "spread": spread, "diameter": max_diameter})

        return clusters

    # Pick best cluster
    def choose_cluster(self, clusters, predicted, dt):
        if not clusters:
            return None
        if predicted is None:
            return max(clusters, key=lambda c: c["size"]) 
        prev_vel = self.memory[-1]["vel"] if len(self.memory) else self.velocity
        prev_pos = self.memory[-1]["pos"] if len(self.memory) else predicted
        scores = []
        for c in clusters:
            dist_score = np.linalg.norm(c["center"] - predicted)
            new_vel = (c["center"] - prev_pos)/ dt
            vel_score = np.linalg.norm(new_vel - prev_vel)
            size_bonus = 0.03 * np.log1p(c["size"])
            spread_penalty = 0.2 * c["spread"]
            scores.append(dist_score + 0.4 * vel_score + spread_penalty - size_bonus)
        return clusters[int(np.argmin(scores))]

    # Update Kalman measurement noise
    def update_measurement_noise(self, cluster):
        n = max(cluster["size"], 1)
        spread = max(cluster["spread"], self.LIDAR_NOISE_STD)
        var = np.clip((spread ** 2 + self.LIDAR_NOISE_STD ** 2) / n, 1e-5, 0.05)
        self.kf.R = np.eye(3) * var

    # Log data
    def log_data(self, t, cluster_size=0):
        if self.true_pose is None or self.est_pose is None:
            return
        self.log.write(f"{t},{self.true_pose[0]},{self.true_pose[1]},{self.true_pose[2]},{self.est_pose[0]},{self.est_pose[1]},{self.est_pose[2]},{self.velocity[0]},{self.velocity[1]},{self.velocity[2]},{self.acceleration[0]},{self.acceleration[1]},{self.acceleration[2]},{cluster_size},{self.roi_radius},{self.lost_counter}\n")
        self.log.flush()
        return

    # Publish drone position
    def publish_position(self, centroid, msg):

        msg_out = PointStamped()
        msg_out.header = msg.header
        msg_out.point.x = float(centroid[0])
        msg_out.point.y = float(centroid[1])
        msg_out.point.z = float(centroid[2])

        self.est_pose = (
            msg_out.point.x + self.LIDAR_X,
            msg_out.point.y + self.LIDAR_Y,
            msg_out.point.z + self.LIDAR_Z
        )

        self.publisher.publish(msg_out)
        # self.get_logger().info(f"Drone position: {centroid}")

    # Publish roi marker (for rviz2)
    def publish_roi_marker(self, predicted_position, msg):

        if predicted_position is None:
            return

        marker = Marker()

        marker.header.frame_id = msg.header.frame_id
        marker.header.stamp = msg.header.stamp

        marker.ns = "drone_roi"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = float(predicted_position[0])
        marker.pose.position.y = float(predicted_position[1])
        marker.pose.position.z = float(predicted_position[2])

        marker.pose.orientation.w = 1.0

        marker.scale.x = float(self.roi_radius * 2.0)
        marker.scale.y = float(self.roi_radius * 2.0)
        marker.scale.z = float(self.roi_radius * 2.0)

        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.15

        self.roi_publisher.publish(marker)

    # Destroy node
    def destroy_node(self):
        if hasattr(self, "log") and not self.log.closed:
            self.log.close()
        super().destroy_node()

    # Callback (when receiving LiDAR point cloud)
    def callback(self, msg):
        

        if self.shutdown_flag: 
            return

        # Time
        current_time = self.get_clock().now().nanoseconds * 1e-9

        # Point cloud
        # t0 = time.perf_counter()
        points = self.convert_pointcloud(msg)
        # t1 = time.perf_counter()

        # self.get_logger().info(f"conversion time: {(t1 - t0)*1000:.2f} ms")
        if points is None:
            self.prev_time = current_time
            return

        # Initialize background
        if not self.background_initialized:
            self.initialize_background(points, current_time)
            return

        # Timestep
        if self.prev_time is not None:
            dt = current_time - self.prev_time
        else:
            dt = None
        if dt is not None and dt > 0:
            self.kf.set_dt(dt)

        # Predict and estimate position
        predicted = self.predict_from_memory(dt)
        
        roi_pts = self.roi_filter(points, predicted, cKDTree(points))
        moving = self.subtract_background(roi_pts, predicted)
        clusters = self.get_clusters(moving)
        chosen = self.choose_cluster(clusters, predicted, dt)

        if chosen is None: #Drone not found in this frame
            self.lost_counter += 1 #Count frames where drone is lost
            if predicted is not None:
                if self.kf_initialized:
                    self.velocity = self.kf.x[3:6, 0].copy()
                    self.acceleration = self.kf.x[6:9, 0].copy()
                    self.kf.P += np.eye(9) * 0.01
                self.publish_position(predicted, msg)
                self.publish_roi_marker(predicted, msg)
                self.log_data(current_time, 0)
            self.prev_time = current_time
            return

        z = chosen["center"]
        
        if USE_KALMAN:
            if not self.kf_initialized:
                self.lost_counter = 0 #Drone detected in this frame
                self.kf.set_state(z)
                self.kf_initialized = True
                filt = z
            else:
                self.update_measurement_noise(chosen)
                if self.kf.mahalanobis_distance_squared(z) > self.MAHALANOBIS_GATE: #Check if result is believable
                    self.lost_counter += 1
                    filt = predicted if predicted is not None else self.kf.x[:3, 0]
                else:
                    self.lost_counter = 0 #Drone detected in this frame
                    filt = self.kf.update(z)
            self.velocity = self.kf.x[3:6, 0].copy()
            self.acceleration = self.kf.x[6:9, 0].copy()
        else:
            filt = z
            self.lost_counter = 0 #Drone detected in this frame

        # Velocity and acceleration
        if len(self.memory):
            last = self.memory[-1]
            dt_mem = max(current_time - last["t"], 1e-4)
            vel = (filt - last["pos"]) / dt_mem
            acc = (vel - last["vel"]) / dt_mem
            self.velocity, self.acceleration = vel, acc

        # Update memory
        self.memory.append({"t": current_time, "pos": filt.copy(), "vel": self.velocity.copy(), "cluster_size": chosen["size"], "spread": chosen["spread"]})
        
        # Publish and log data
        self.publish_position(filt, msg)
        self.publish_roi_marker(predicted, msg)
        self.log_data(current_time, chosen["size"])
        
        self.prev_time = current_time
        # t6 = time.perf_counter()
        # self.get_logger().info(f"Total time : {(t6 - t0)*1000:.2f} ms")


def main(args=None):
    rclpy.init(args=args)
    node = DroneLocalizer()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.shutdown_flag:
                break
    finally:
        node.get_logger().info("Shutting down localization node")
        if hasattr(node, "log") and not node.log.closed:
            node.log.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
