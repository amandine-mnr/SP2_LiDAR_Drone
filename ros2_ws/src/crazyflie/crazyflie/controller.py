import sys
import termios
import tty
import select

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class CrazyflieController(Node):

    def __init__(self):
        super().__init__('crazyflie_controller')
        self.get_logger().info("Crazyflie Controller Node Started !")

        self.pub = self.create_publisher(Twist, '/crazyflie/gazebo/command/twist', 10)

        self.timer = self.create_timer(0.1, self.control_loop)
        self.settings = termios.tcgetattr(sys.stdin)
        self.key = None

    def get_key(self):
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = None
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def control_loop(self):
        msg = Twist()

        key = self.get_key()
        if key == 'z':      #Forward
            msg.linear.x = 0.3
        elif key == 's':    #Backward
            msg.linear.x = -0.3
        elif key == 'q':    #Left
            msg.linear.y = 0.3
        elif key == 'd':    #Right
            msg.linear.y = -0.3
        elif key == 'c':    #Up
            msg.linear.z = 0.3
        elif key == 'v':    #Down
            msg.linear.z = -0.3
        elif key == 'a':    #Rotate left
            msg.angular.z = 0.5
        elif key == 'e':    #Rotate right
            msg.angular.z = -0.5
        elif key == '\x03': #Ctrl-C
            rclpy.shutdown()
            return

        if key is not None:
            self.get_logger().info(f"Key pressed: {key}")
            self.pub.publish(msg)


def main():
    rclpy.init()
    node = CrazyflieController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()