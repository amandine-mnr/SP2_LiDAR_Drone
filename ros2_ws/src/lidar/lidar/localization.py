import os
import numpy as np
from datetime import datetime
from collections import deque
import rclpy
from dataclasses import dataclass, field
import ast
from rclpy.node import Node

from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree

import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker, MarkerArray

from lidar.kalman import KalmanFilter


USE_KALMAN = True

def get_timestamped_log_filename(base_name="drone_log_multitrack", ext=".csv", folder="."):
    timestamp = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
    filename = f"{base_name}_{timestamp}{ext}"
    return os.path.join(folder, filename)

@dataclass
class Track:
    #State for one detected drone
    track_id: int
    kf: KalmanFilter
    memory_len: int
    first_seen: float
    last_seen: float
    memory: deque = field(init=False)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    lost_counter: int = 0
    hits: int = 0
    last_cluster_size: int = 0
    last_spread: float = 0.0
    roi_radius: float = 0.5
    kf_initialized: bool = False
    publisher: object = None

    def __post_init__(self):
        self.memory = deque(maxlen=self.memory_len)

    @property
    def position(self):
        if self.kf_initialized:
            return self.kf.x[0:3, 0].copy()
        if self.memory:
            return self.memory[-1]["pos"].copy()
        return None

class DroneLocalizer(Node):

    def __init__(self):
        super().__init__('drone_localizer')

        # Parameters
        self.declare_parameter("lidar_x", 0.5)
        self.declare_parameter("lidar_y", 1.5)
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

        # Multi-drone parameters
        self.declare_parameter("max_tracks", 10)
        self.declare_parameter("max_lost_frames", 8)
        self.declare_parameter("association_gate", 0.65)
        self.declare_parameter("publish_unconfirmed_tracks", True)
        self.declare_parameter("min_hits_to_confirm", 1)
        self.declare_parameter("odom_topics", ["/model/crazyflie/odometry"])
        self.declare_parameter(
            "drone_start_pose",
            ["(0.0, 0.0, 0.0)",
            "(1.0, 0.0, 0.0)",
            "(0.0, 1.0, 0.0)"]
        )

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
        
        # Multi-drones
        self.MAX_TRACKS = int(self.get_parameter("max_tracks").value)
        self.MAX_LOST_FRAMES = int(self.get_parameter("max_lost_frames").value)
        self.ASSOCIATION_GATE = float(self.get_parameter("association_gate").value)
        self.PUBLISH_UNCONFIRMED_TRACKS = bool(self.get_parameter("publish_unconfirmed_tracks").value)
        self.MIN_HITS_TO_CONFIRM = int(self.get_parameter("min_hits_to_confirm").value)
        self.ODOM_TOPICS = list(self.get_parameter("odom_topics").value)
        # self.DRONE_START_POSE = list(self.get_parameter("drone_start_pose").value)

        raw_poses = self.get_parameter("drone_start_pose").value

        self.DRONE_START_POSE = np.array([
            ast.literal_eval(pose) for pose in raw_poses
        ], dtype=float)

        # Log file
        log_dir = "ros2_ws/src/lidar/lidar/metrics_log"
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = get_timestamped_log_filename(folder=log_dir)
        self.get_logger().info(f"Log file: {self.log_file}")
        self.log = open(self.log_file, "w")
        row = ["time","track_id", "est_x","est_y","est_z","vel_x","vel_y","vel_z","acc_x","acc_y","acc_z","cluster_size","roi","lost,hits"]
        for i in range (len(self.ODOM_TOPICS)):
            row.extend(["true_x", "true_y", "true_z"])
        self.log.write(",".join(map(str, row)) + "\n")
        self.log.flush()

        # Publisher : Localization ready
        self.ready_publisher = self.create_publisher(Bool, '/localizer_ready', 10)
        self.is_ready = False
        self.shutdown_flag = False

        # Suscriber : Lidar point cloud
        self.subscription = self.create_subscription(PointCloud2, '/my_3D_lidar/points', self.callback, 1)

        # Publisher : ROI marker (rviz)
        self.roi_publisher = self.create_publisher(MarkerArray, '/roi_markers', 1)
        # Publisher : Tracks
        self.track_marker_publisher = self.create_publisher(MarkerArray, '/track_markers', 1)
        
        # Suscriber : System shutdown
        self.shutdown_sub = self.create_subscription(Bool, '/system_shutdown', self.shutdown_callback, 1)
        
        # Suscribers : Odometry
        self.true_poses = {}
        self.odom_subs = []
        for i, topic in enumerate(self.ODOM_TOPICS):
            self.odom_subs.append(
                self.create_subscription(Odometry, topic, lambda msg, idx=i: self.odometry_callback(msg, idx), 1)
            )

        # Background
        self.background_frames = []
        self.background = None
        self.background_initialized = False
        self.kdtree = None

        self.prev_time = None

        # Multi drone tracking
        self.tracks = {}
        self.next_track_id = 0
        
        # Kalman filter
        self.measurement_var = (self.LIDAR_NOISE_STD ** 2) * 2.0

        self.get_logger().info("Multi-drone localization node started")
        self.get_logger().info(f"max_tracks={self.MAX_TRACKS}, association_gate={self.ASSOCIATION_GATE} m")

    def shutdown_callback(self, msg):
        if msg.data:
            self.get_logger().info("System shutdown received, stopping localizer")
            self.shutdown_flag = True
            if hasattr(self, "log") and not self.log.closed:
                self.log.flush()
                self.log.close()

    # Odometry
    def odometry_callback(self, msg, idx=0):
        if self.shutdown_flag:
            return
        self.true_poses[idx] = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
        )

    # PointCloud conversion
    def convert_pointcloud(self, msg):
        points = np.array(
            [[p[0], p[1], p[2]] for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)],
            dtype=np.float32,
        )
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

    # Find moving points
    def subtract_background(self, points):

        if len(points) == 0:
            return np.empty((0,3))
        distances, _ = self.kdtree.query(points, k=1)
        return points[distances > self.BG_THRESHOLD]

    # Clustering
    def get_clusters(self, moving_points):
        if len(moving_points) < self.DBSCAN_MIN_SAMPLES:
            return []

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
            max_diameter = float(2.0 * np.max(dists)) if len(dists) else 0.0

            #reject clusters larger than expected drone size
            if (max_diameter > (self.DRONE_SIZE + self.SIZE_MARGIN)):
                # self.get_logger().info("Reject cluster: size too large")
                continue

            clusters.append({"points": cluster, "center": center, "size": len(cluster), "spread": spread, "diameter": max_diameter})

        return clusters

    # Tracking
    def create_track(self, cluster, current_time):
        if len(self.tracks) >= self.MAX_TRACKS:
            return None

        kf = KalmanFilter(dt=0.1, measurement_var=self.measurement_var, process_var=0.8)
        track = Track(
            track_id=self.next_track_id,
            kf=kf,
            memory_len=self.MEMORY_LEN,
            first_seen=current_time,
            last_seen=current_time,
        )
        self.next_track_id += 1

        z = cluster["center"]
        track.kf.set_state(z)
        track.kf_initialized = True
        track.hits = 1
        track.last_cluster_size = cluster["size"]
        track.last_spread = cluster["spread"]
        track.memory.append(
            {"t": current_time, "pos": z.copy(), "vel": track.velocity.copy(), "cluster_size": cluster["size"], "spread": cluster["spread"]}
        )
        track.publisher = self.create_publisher(PointStamped, f'/drone_position/track_{track.track_id}', 1)
        self.tracks[track.track_id] = track
        self.get_logger().info(f"Created track {track.track_id} at {z}")
        return track

    def compute_roi_radius(self, track):
        if track.kf_initialized:
            P_pos = track.kf.P[:3, :3]
            r = self.ROI_FACTOR * np.sqrt(np.trace(P_pos))
        else:
            r = 0.5 + 0.8 * np.linalg.norm(track.velocity)
        track.roi_radius = float(np.clip(r, self.ROI_MIN_RADIUS, self.ROI_MAX_RADIUS))
        return track.roi_radius

    def predict_track(self, track, dt):
        if dt is not None and dt > 0:
            track.kf.set_dt(dt)

        kf_pred = track.kf.predict() if USE_KALMAN and track.kf_initialized else None

        if len(track.memory) < self.HISTORY_PREDICT_MIN:
            return kf_pred

        pos = np.array([m["pos"] for m in track.memory])
        t = np.array([m["t"] for m in track.memory])
        t = t - t[-1]
        target_t = max(dt if dt is not None else 0.0, 0.0)
        deg = min(2, len(track.memory) - 1)
        poly_pred = np.array([np.polyval(np.polyfit(t, pos[:, d], deg), target_t) for d in range(3)])

        if kf_pred is None:
            return poly_pred
        return 0.65 * kf_pred + 0.35 * poly_pred

    def update_measurement_noise(self, track, cluster):
        n = max(cluster["size"], 1)
        spread = max(cluster["spread"], self.LIDAR_NOISE_STD)
        var = np.clip((spread ** 2 + self.LIDAR_NOISE_STD ** 2) / n, 1e-5, 0.05)
        track.kf.R = np.eye(3) * var

    def score_assignment(self, track, cluster, predicted):
        z = cluster["center"]
        dist = np.linalg.norm(z - predicted)
        if dist > max(self.ASSOCIATION_GATE, track.roi_radius):
            return None

        # Mahalanobis gate : reject clusters that are not believable
        if USE_KALMAN and track.kf_initialized:
            maha_dist = track.kf.mahalanobis_distance_squared(z)
            if maha_dist > self.MAHALANOBIS_GATE:
                return None
        else:
            maha_dist = 0.0

        prev_pos = track.memory[-1]["pos"] if track.memory else predicted
        prev_vel = track.memory[-1]["vel"] if track.memory else track.velocity
        new_vel = z - prev_pos
        vel_score = np.linalg.norm(new_vel - prev_vel)
        size_bonus = 0.03 * np.log1p(cluster["size"])
        spread_penalty = 0.2 * cluster["spread"]
        return dist + 0.4 * vel_score + 0.02 * maha_dist + spread_penalty - size_bonus

    # Assign clusters to existing tracks
    def associate(self, clusters, predictions):
        candidates = []
        for tid, predicted in predictions.items(): #tid : track id
            track = self.tracks[tid]
            if predicted is None:
                continue
            self.compute_roi_radius(track)
            for ci, cluster in enumerate(clusters): #ci : cluster index
                score = self.score_assignment(track, cluster, predicted)
                if score is not None:
                    candidates.append((score, tid, ci)) 

        candidates.sort(key=lambda x: x[0])
        assigned_tracks = set()
        assigned_clusters = set()
        assignments = {}
        for _, tid, ci in candidates:
            if tid in assigned_tracks or ci in assigned_clusters:
                continue
            assignments[tid] = ci
            assigned_tracks.add(tid)
            assigned_clusters.add(ci)
        return assignments, assigned_clusters

    def update_track(self, track, cluster, current_time):
        z = cluster["center"]
        track.lost_counter = 0
        track.hits += 1
        track.last_seen = current_time
        track.last_cluster_size = cluster["size"]
        track.last_spread = cluster["spread"]

        if USE_KALMAN:
            self.update_measurement_noise(track, cluster)
            filt = track.kf.update(z)
            track.velocity = track.kf.x[3:6, 0].copy()
            track.acceleration = track.kf.x[6:9, 0].copy()
        else:
            filt = z

        if track.memory:
            last = track.memory[-1]
            dt_mem = max(current_time - last["t"], 1e-4)
            vel = (filt - last["pos"]) / dt_mem
            acc = (vel - last["vel"]) / dt_mem
            track.velocity = vel
            track.acceleration = acc

        track.memory.append(
            {"t": current_time, "pos": filt.copy(), "vel": track.velocity.copy(), "cluster_size": cluster["size"], "spread": cluster["spread"]}
        )
        return filt

    def mark_track_lost(self, track, predicted, current_time):
        track.lost_counter += 1
        track.last_cluster_size = 0
        if track.kf_initialized:
            track.velocity = track.kf.x[3:6, 0].copy()
            track.acceleration = track.kf.x[6:9, 0].copy()
            track.kf.P += np.eye(9) * 0.01
        if predicted is not None:
            track.memory.append(
                {"t": current_time, "pos": predicted.copy(), "vel": track.velocity.copy(), "cluster_size": 0, "spread": 0.0}
            )
        return predicted

    def delete_tracks(self):
        to_delete = [tid for tid, trk in self.tracks.items() if trk.lost_counter > self.MAX_LOST_FRAMES]
        for tid in to_delete:
            self.get_logger().info(f"Deleting lost track {tid}")
            del self.tracks[tid]
            self.delete_marker(tid)

    # Publishing/logging 
    def track_is_publishable(self, track):
        return self.PUBLISH_UNCONFIRMED_TRACKS or track.hits >= self.MIN_HITS_TO_CONFIRM

    def publish_position(self, track, centroid, msg):
        if centroid is None or not self.track_is_publishable(track):
            return

        msg_out = PointStamped()
        msg_out.header = msg.header
        msg_out.point.x = float(centroid[0])
        msg_out.point.y = float(centroid[1])
        msg_out.point.z = float(centroid[2])

        track.publisher.publish(msg_out)

    # Publish markers (for rviz2)
    def publish_markers(self, predictions, msg):
        roi_array = MarkerArray()
        track_array = MarkerArray()

        for tid, track in self.tracks.items():
            predicted = predictions.get(tid, track.position)
            if predicted is not None:
                roi_marker = Marker()
                roi_marker.header.frame_id = msg.header.frame_id
                roi_marker.header.stamp = msg.header.stamp
                roi_marker.ns = "drone_roi"
                roi_marker.id = int(tid)
                roi_marker.type = Marker.SPHERE
                roi_marker.action = Marker.ADD
                roi_marker.pose.position.x = float(predicted[0])
                roi_marker.pose.position.y = float(predicted[1])
                roi_marker.pose.position.z = float(predicted[2])
                roi_marker.pose.orientation.w = 1.0
                roi_marker.scale.x = float(track.roi_radius * 2.0)
                roi_marker.scale.y = float(track.roi_radius * 2.0)
                roi_marker.scale.z = float(track.roi_radius * 2.0)
                roi_marker.color.r = 0.0
                roi_marker.color.g = 1.0
                roi_marker.color.b = 0.0
                roi_marker.color.a = 0.12
                roi_array.markers.append(roi_marker)

            pos = track.position
            if pos is not None:
                marker = Marker()

                marker.header.frame_id = msg.header.frame_id
                marker.header.stamp = msg.header.stamp
                
                marker.ns = "drone_track"
                marker.id = int(tid)
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                
                marker.pose.position.x = float(pos[0])
                marker.pose.position.y = float(pos[1])
                marker.pose.position.z = float(pos[2])
                
                marker.pose.orientation.w = 1.0
                
                marker.scale.x = self.DRONE_SIZE
                marker.scale.y = self.DRONE_SIZE
                marker.scale.z = self.DRONE_SIZE
                
                marker.color.r = 0.5
                marker.color.g = 0.0
                marker.color.b = 2.0
                marker.color.a = 0.9
                track_array.markers.append(marker)

        self.roi_publisher.publish(roi_array)
        self.track_marker_publisher.publish(track_array)

    def delete_marker(self, tid):
        arr = MarkerArray()

        m = Marker()
        m.header.frame_id = "map"
        m.ns = "drone_track"
        m.id = tid
        m.action = Marker.DELETE
        arr.markers.append(m)

        r = Marker()
        r.header.frame_id = "map"
        r.ns = "drone_roi"
        r.id = tid
        r.action = Marker.DELETE
        arr.markers.append(r)

        self.track_marker_publisher.publish(arr)
        self.roi_publisher.publish(arr)

    # Log file
    def log_track(self, t, track):
        pos = track.position
        if pos is None:
            return

        #estimated pose
        est_pose = (
            pos[0] + self.LIDAR_X,
            pos[1] + self.LIDAR_Y,
            pos[2] + self.LIDAR_Z
        )

        #true poses
        true_poses_all = []
        for i in range(len(self.ODOM_TOPICS)):
            raw = self.true_poses.get(i, (np.nan, np.nan, np.nan))

            #offsets
            if i < len(self.DRONE_START_POSE):
                offset = self.DRONE_START_POSE[i]
                true = (
                    raw[0] - offset[0],
                    raw[1] - offset[1],
                    raw[2] - offset[2],
                )
            else:
                true = raw

            true_poses_all.extend(true)

        #write row
        row = [t, track.track_id]

        row.extend([
            est_pose[0], est_pose[1], est_pose[2],
            track.velocity[0], track.velocity[1], track.velocity[2],
            track.acceleration[0], track.acceleration[1], track.acceleration[2],
            track.last_cluster_size,
            track.roi_radius,
            track.lost_counter,
            track.hits
        ])

        row.extend(true_poses_all)

        self.log.write(",".join(map(str, row)) + "\n")

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
        points = self.convert_pointcloud(msg)
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

        # Predict all current tracks
        predictions = {tid: self.predict_track(track, dt) for tid, track in list(self.tracks.items())}

        # Find new clusters and assign them to tracks
        moving = self.subtract_background(points)
        clusters = self.get_clusters(moving)
        assignments, assigned_clusters = self.associate(clusters, predictions)

        # Update tracks
        for tid, ci in assignments.items():
            track = self.tracks[tid]
            filt = self.update_track(track, clusters[ci], current_time)
            predictions[tid] = filt

        # Lost tracks
        for tid, track in list(self.tracks.items()):
            if tid not in assignments:
                pred = predictions.get(tid)
                predictions[tid] = self.mark_track_lost(track, pred, current_time)

        # New tracks
        for ci, cluster in enumerate(clusters):
            if ci not in assigned_clusters:
                track = self.create_track(cluster, current_time)
                if track is not None:
                    predictions[track.track_id] = track.position

        self.delete_tracks()

        # Publish and log data
        for tid, track in sorted(self.tracks.items()):
            self.publish_position(track, track.position, msg)
            self.log_track(current_time, track)
        self.log.flush()
        self.publish_markers(predictions, msg)
        self.prev_time = current_time


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
