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

DRONE_START_POSE = np.array([0.0, 0.0, 0.0])

class CrazyflieTrajectory(Node):

    def __init__(self):
        super().__init__('crazyflie_polynomial_trajectory')

        self.declare_parameter('max_length', 1.3)
        self.declare_parameter('max_width', 1.0)
        self.declare_parameter('max_delta_z', 0.4)
        self.declare_parameter('base_altitude', 1.0)

        self.declare_parameter('num_waypoints', 60)
        # self.declare_parameter('wp_threshold', 0.15)
        self.declare_parameter('kp', 1.2)
        self.declare_parameter('speed_limit', 0.9)
        self.declare_parameter('waypoint_spacing', 0.05)

        self.max_length = self.get_parameter('max_length').value
        self.max_width = self.get_parameter('max_width').value
        self.max_delta_z = self.get_parameter('max_delta_z').value
        self.base_altitude = self.get_parameter('base_altitude').value
        self.num_waypoints = self.get_parameter('num_waypoints').value
        # self.wp_threshold = self.get_parameter('wp_threshold').value
        self.kp = self.get_parameter('kp').value
        self.speed_limit = self.get_parameter('speed_limit').value
        self.waypoint_spacing = self.get_parameter('waypoint_spacing').value

        self.wp_threshold = (0.2/2.2)*self.kp

        #publisher : velocity commands
        self.pub = self.create_publisher(Twist, '/crazyflie/gazebo/command/twist', 10)

        #subscriber : odometry
        self.sub = self.create_subscription(Odometry, '/model/crazyflie/odometry', self.odometry_callback, 10)

        #suscriber : shutdown (from supervisor)
        self.shutdown_sub = self.create_subscription(Bool, '/system_shutdown', self.shutdown_callback, 10)

        #suscriber : start when localizer ready
        self.ready_sub = self.create_subscription(Bool, '/localizer_ready', self.ready_callback, 10)

        self.timer = self.create_timer(0.02, self.control_loop)

        self.localization_ready = False
        self.shutdown_flag = False
        self.current_pose = None
        self.current_wp = 0
        self.takeoff_done = False

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

    def odometry_callback(self, msg):
        self.current_pose = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        )

    def shutdown_callback(self, msg):
        if msg.data:
            self.get_logger().info("Shutdown received")
            self.shutdown_flag = True
            stop = Twist()
            self.pub.publish(stop)

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

        theta_dense = np.linspace(0.0, 2.0 * np.pi, 20000)

        x_dense = self.max_length * np.cos(theta_dense)
        y_dense = self.max_width * np.sin(theta_dense)
        z_dense = (self.base_altitude + self.max_delta_z * np.sin(2.0 * theta_dense))

        #3D arc length
        dx = np.diff(x_dense)
        dy = np.diff(y_dense)
        dz = np.diff(z_dense)

        ds = np.sqrt(dx**2 + dy**2 + dz**2)
        cumulative_s = np.concatenate(([0.0], np.cumsum(ds)))

        total_length = cumulative_s[-1]

        #sample at constant distance intervals
        sample_s = np.arange(0.0, total_length, self.waypoint_spacing)

        #convert arc length back to theta
        theta_samples = np.interp(sample_s, cumulative_s, theta_dense)

        #generate waypoints
        self.waypoints = []

        for theta in theta_samples:
            x = self.max_length * np.cos(theta)
            y = self.max_width * np.sin(theta)
            z = (self.base_altitude + self.max_delta_z * np.sin(2.0 * theta))

            self.waypoints.append(np.array([x + DRONE_START_POSE[0], y + DRONE_START_POSE[1], z], dtype=float))

        #close the loop
        if len(self.waypoints) > 0:
            self.waypoints.append(self.waypoints[0].copy())

        #find starting point
        waypoints_np = np.array(self.waypoints)
        self.current_wp = int(np.argmin(np.linalg.norm(waypoints_np - DRONE_START_POSE, axis=1)))

        self.get_logger().info(
            f"Generated {len(self.waypoints)} waypoints "
            f"with {self.waypoint_spacing:.2f} m spacing "
            f"(trajectory length: {total_length:.2f} m)"
        )

    def control_loop(self):

        if self.shutdown_flag:
            return
        if not self.localization_ready:
            return

        if self.current_pose is None:
            return

        key = self.get_key()
        if key == '\x03':
            self.shutdown_flag = True
            return

        target = self.waypoints[self.current_wp]
        next_target = self.waypoints[(self.current_wp + 1) % len(self.waypoints)]

        x, y, z = self.current_pose

        # Takeoff
        if ((z < self.base_altitude-self.max_delta_z) and (self.takeoff_done == False)):
            msg = Twist()
            msg.linear.x = 0.0
            msg.linear.y = 0.0
            msg.linear.z = 0.5
            self.pub.publish(msg)
            return
        self.takeoff_done = True

        dx = target[0] - x
        dy = target[1] - y
        dz = target[2] - z

        dx_next = next_target[0] - x
        dy_next = next_target[1] - y
        dz_next = next_target[2] - z

        distance = math.sqrt(dx*dx + dy*dy + dz*dz)
        distance_next = math.sqrt(dx_next*dx_next + dy_next*dy_next + dz_next*dz_next)

        #switch waypoint when close
        if (distance < self.wp_threshold) or (distance_next < self.wp_threshold):
            self.current_wp = (self.current_wp + 1) % len(self.waypoints)
            return

        msg = Twist()

        msg.linear.x = np.clip(
            self.kp * dx_next,
            -self.speed_limit,
            self.speed_limit
        )

        msg.linear.y = np.clip(
            self.kp * dy_next,
            -self.speed_limit,
            self.speed_limit
        )

        msg.linear.z = np.clip(
            self.kp * dz_next,
            -self.speed_limit,
            self.speed_limit
        )

        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0

        self.pub.publish(msg)

    def destroy_node(self):
        stop = Twist()
        self.pub.publish(stop)
        super().destroy_node()


def main():
    rclpy.init()
    node = CrazyflieTrajectory()

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