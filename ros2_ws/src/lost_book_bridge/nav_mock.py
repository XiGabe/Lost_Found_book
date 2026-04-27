"""
Mock nav node for testing the ROS2 bridge.

Run this after starting the bridge to simulate the nav node calling
the trigger_cv service each time you press Enter.

  ros2 run lost_book_bridge bridge
  ros2 run lost_book_bridge nav_mock
"""

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


class NavMock(Node):
    def __init__(self):
        super().__init__('nav_mock')
        self._cv_client = self.create_client(Trigger, 'trigger_cv')

    def trigger_cv_and_wait(self):
        if not self._cv_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("trigger_cv service not available")
            return False

        future = self._cv_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future)

        response = future.result()
        if response is None:
            self.get_logger().error("trigger_cv call failed")
            return False

        if response.success:
            self.get_logger().info("CV done, resuming navigation")
        else:
            self.get_logger().error(f"CV failed: {response.message}")

        return response.success


def main():
    rclpy.init()
    node = NavMock()

    try:
        while True:
            input("Press Enter to simulate nav stop and trigger CV...")
            node.trigger_cv_and_wait()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
