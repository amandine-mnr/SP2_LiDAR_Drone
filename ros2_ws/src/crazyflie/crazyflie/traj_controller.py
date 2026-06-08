import sys
import termios
import tty
import select
import math
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool

class CrazyfliePolynomialTrajectory(Node):

    def __init__(self):
        super().__init__('crazyflie_polynomial_trajectory')

        self.declare_parameter('max_length', 1.3)
        self.declare_parameter('max_width', 1.0)
        self.declare_parameter('max_delta_z', 0.4)
        self.declare_parameter('base_altitude', 1.0)

        self.declare_parameter('num_waypoints', 30)
        self.declare_parameter('wp_threshold', 0.15)
        self.declare_parameter('kp', 1.2)
        self.declare_parameter('speed_limit', 0.9)

        self.max_length = self.get_parameter('max_length').value
        self.max_width = self.get_parameter('max_width').value
        self.max_delta_z = self.get_parameter('max_delta_z').value
        self.base_altitude = self.get_parameter('base_altitude').value
        self.num_waypoints = self.get_parameter('num_waypoints').value
        self.wp_threshold = self.get_parameter('wp_threshold').value
        self.kp = self.get_parameter('kp').value
        self.speed_limit = self.get_parameter('speed_limit').value

        self.drones = [
            "crazyflie1",
            "crazyflie2",
            "crazyflie3"
        ]

        self.drones_start_pos = {
            "crazyflie1": np.array([0.0, 0.0, 0.0]),
            "crazyflie2": np.array([1.0, 0.0, 0.0]),
            "crazyflie3": np.array([0.0, -1.0, 0.0])
        }

        #publishers : velocity commands
        self.publishers_dict = {}
        for drone in self.drones:
            self.publishers_dict[drone] = self.create_publisher(Twist, f'/{drone}/gazebo/command/twist',10)

        self.current_pose = {drone: None for drone in self.drones}

        self.current_wp = {drone: 0 for drone in self.drones}

        #subscribers : odometry
        self.odom_sub = []
        for drone in self.drones:
            sub = self.create_subscription(Odometry, f'/model/{drone}/odometry', lambda msg, d=drone: self.odometry_callback(msg, d), 10)
            self.odom_sub.append(sub)

        #suscriber : shutdown (from supervisor)
        self.shutdown_sub = self.create_subscription(Bool, '/system_shutdown', self.shutdown_callback, 10)

        #suscriber : start when localizer ready
        self.ready_sub = self.create_subscription(Bool, '/localizer_ready', self.ready_callback, 10)

        self.timer = self.create_timer(0.02, self.control_loop)

        self.localization_ready = False
        self.shutdown_flag = False

        if sys.stdin.isatty():
            self.settings = termios.tcgetattr(sys.stdin)
        else:
            self.settings = None


        self.generate_trajectory()
        self.get_logger().info("Trajectory controller started")


    def ready_callback(self, msg):
        if msg.data:
            self.localization_ready = True
            self.get_logger().info("Localization ready, starting drone controller")

    def odometry_callback(self, msg, drone):
        self.current_pose[drone] = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        )

    def shutdown_callback(self, msg):
        if msg.data:
            self.get_logger().info("Shutdown received")
            self.shutdown_flag = True
            stop = Twist()
            for drone in self.drones:
                self.publishers_dict[drone].publish(stop)

    def get_key(self):

        if self.settings is None:
            return None

        try:
            tty.setraw(sys.stdin.fileno())
            rlist, _, _ = select.select([sys.stdin], [], [], 0.1)

            if rlist:
                key = sys.stdin.read(1)
            else:
                key = None

            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
            return key

        except Exception:
            return None

    def generate_trajectory(self):

        self.waypoints = []

        for i in range(self.num_waypoints):

            u = i / self.num_waypoints

            theta = 2.0 * np.pi * u

            x = self.max_length * np.cos(theta)

            y = self.max_width * np.sin(theta)

            z = (
                self.base_altitude
                + self.max_delta_z
                * np.sin(2.0 * theta)
            )

            self.waypoints.append(
                np.array([x, y, z])
            )

        self.get_logger().info(
            f"Generated {len(self.waypoints)} waypoints"
        )

    def control_loop(self):

        if self.shutdown_flag:
            return
        if not self.localization_ready:
            return

        key = self.get_key()
        if key == '\x03':
            self.shutdown_flag = True
            return

        for drone in self.drones:

            if self.current_pose[drone] is None:
                continue

            target = (self.waypoints[self.current_wp[drone]] + self.drones_start_pos[drone])

            x, y, z = self.current_pose[drone]

            dx = target[0] - x
            dy = target[1] - y
            dz = target[2] - z

            distance = math.sqrt(
                dx * dx +
                dy * dy +
                dz * dz
            )

            if distance < self.wp_threshold:
                self.current_wp[drone] = (self.current_wp[drone] + 1) % len(self.waypoints)
                continue

            msg = Twist()

            msg.linear.x = np.clip(
                self.kp * dx,
                -self.speed_limit,
                self.speed_limit
            )

            msg.linear.y = np.clip(
                self.kp * dy,
                -self.speed_limit,
                self.speed_limit
            )

            msg.linear.z = np.clip(
                self.kp * dz,
                -self.speed_limit,
                self.speed_limit
            )

            msg.angular.x = 0.0
            msg.angular.y = 0.0
            msg.angular.z = 0.0

            self.publishers_dict[drone].publish(msg)

    def destroy_node(self):
        stop = Twist()
        for drone in self.drones:
            self.publishers_dict[drone].publish(stop)
        super().destroy_node()


def main():
    rclpy.init()
    node = CrazyfliePolynomialTrajectory()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.02)
            if node.shutdown_flag:
                break
    finally:
        node.get_logger().info("Shutting down drone controller node")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()