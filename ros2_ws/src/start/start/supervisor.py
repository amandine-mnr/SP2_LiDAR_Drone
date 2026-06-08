import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool


class ExperimentSupervisor(Node):

    def __init__(self):
        super().__init__('experiment_supervisor')

        self.duration = 240.0  #sec

        self.ready = False
        self.start_time = None
        self.finished = False

        # Suscriber : localization initialized
        self.ready_sub = self.create_subscription(Bool, '/localizer_ready', self.ready_callback, 10)

        # Publisher : shutdown
        self.shutdown_pub = self.create_publisher(Bool, '/system_shutdown', 10)

        self.timer = self.create_timer(0.1, self.loop)

        self.get_logger().info(f"Experiment Supervisor ready (duration = {self.duration}s)")

    def destroy_node(self):
        super().destroy_node()
        return

    def ready_callback(self, msg: Bool):
        if msg.data and not self.ready:
            self.ready = True
            self.start_time = self.get_clock().now()
            self.get_logger().info("Localization ready, experiment started")

    def loop(self):
        if not self.ready or self.finished:
            return
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9
        if elapsed >= self.duration:
            self.get_logger().info("Experiment finished, sending stop signals")
            self.shutdown_pub.publish(Bool(data=True))
            self.finished = True

            
def main(args=None):
    rclpy.init(args=args)
    node = ExperimentSupervisor()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.finished:
                break
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()