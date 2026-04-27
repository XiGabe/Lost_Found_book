import zmq
import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32

CV_TIMEOUT_MS = 10_000

# CV side binds both sockets; bridge connects to them.
# This matches production: CV runs in Docker (stable binder),
# bridge runs natively on the robot (connects to CV's IP).
#
# CV side should use:
#   pull = ctx.socket(zmq.PULL)
#   pull.bind("tcp://*:5555")       # receives "stopped"
#   push = ctx.socket(zmq.PUSH)
#   push.bind("tcp://*:5556")       # bridge connects here to receive "ready"
#
# CV loop:
#   msg = pull.recv_string()        # blocks until bridge sends "stopped"
#   ... process image ...
#   push.send_string("ready")       # signal bridge to resume navigation


class Bridge(Node):
    def __init__(self, push, pull):
        super().__init__('bridge')
        self._push = push
        self._pull = pull
        self._nav_stopped = threading.Event()

        self.create_subscription(Int32, 'nav_status', self._nav_cb, 10)
        self._cv_pub = self.create_publisher(Int32, 'cv_status', 10)

    def _nav_cb(self, msg):
        if msg.data == 1:
            self._nav_stopped.set()

    def run(self):
        while rclpy.ok():
            self._nav_stopped.wait()
            self._nav_stopped.clear()

            self._push.send_string("stopped")

            if self._pull.poll(timeout=CV_TIMEOUT_MS):
                self._pull.recv_string()
                out = Int32()
                out.data = 1
                self._cv_pub.publish(out)
            else:
                self.get_logger().error("CV did not respond within timeout")


def main():
    print("Starting bridge")
    rclpy.init()

    ctx = zmq.Context()
    # CV always runs in Docker on the same machine with -p 5555:5555 -p 5556:5556.
    # For testing, run the bridge container with --network=host so localhost
    # resolves to the host machine where cv_mock.py is running.
    push = ctx.socket(zmq.PUSH)
    push.connect("tcp://localhost:5555")

    pull = ctx.socket(zmq.PULL)
    pull.connect("tcp://localhost:5556")

    while pull.poll(timeout=0):
        pull.recv_string()

    node = Bridge(push, pull)

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
