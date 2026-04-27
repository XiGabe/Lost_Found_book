#!/usr/bin/env python3
import json
import re
import threading
import time
from datetime import datetime
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class PhotoCameraNode(Node):
    """Camera-side node for waypoint-triggered photo capture.

    Protocol:
      /photo_request: std_msgs/String, data is a waypoint id/name
      /photo_done:    std_msgs/String, data is the same waypoint id/name
    """

    def __init__(self):
        super().__init__('photo_camera_node')

        self.declare_parameter('request_topic', '/photo_request')
        self.declare_parameter('done_topic', '/photo_done')
        self.declare_parameter('camera_index', 0)
        self.declare_parameter('output_dir', 'output/photos')
        self.declare_parameter('mock_mode', False)
        self.declare_parameter('mock_delay_sec', 0.0)
        self.declare_parameter('publish_json', False)

        request_topic = self.get_parameter('request_topic').value
        done_topic = self.get_parameter('done_topic').value

        self._done_pub = self.create_publisher(String, done_topic, 10)
        self._request_sub = self.create_subscription(
            String,
            request_topic,
            self._on_photo_request,
            10,
        )

        self._busy_lock = threading.Lock()
        self._busy = False

        self.get_logger().info(
            f'photo camera node ready: {request_topic} -> {done_topic}'
        )

    def _on_photo_request(self, msg: String):
        waypoint_id = msg.data.strip()
        if not waypoint_id:
            self.get_logger().warn('ignored empty photo request')
            return

        with self._busy_lock:
            if self._busy:
                self.get_logger().warn(
                    f'camera busy, ignored photo request for waypoint {waypoint_id}'
                )
                return
            self._busy = True

        worker = threading.Thread(
            target=self._capture_and_publish,
            args=(waypoint_id,),
            daemon=True,
        )
        worker.start()

    def _capture_and_publish(self, waypoint_id: str):
        image_path = ''
        status = 'ok'
        error = ''

        try:
            image_path = self._capture_photo(waypoint_id)
            self.get_logger().info(
                f'photo completed for waypoint {waypoint_id}: {image_path or "mock"}'
            )
        except Exception as exc:
            status = 'error'
            error = str(exc)
            self.get_logger().error(
                f'photo failed for waypoint {waypoint_id}: {error}'
            )
        finally:
            self._publish_done(waypoint_id, status, image_path, error)
            with self._busy_lock:
                self._busy = False

    def _capture_photo(self, waypoint_id: str) -> str:
        mock_mode = bool(self.get_parameter('mock_mode').value)
        mock_delay_sec = float(self.get_parameter('mock_delay_sec').value)

        if mock_mode:
            if mock_delay_sec > 0:
                time.sleep(mock_delay_sec)
            return ''

        import cv2

        camera_index = int(self.get_parameter('camera_index').value)
        output_dir = Path(str(self.get_parameter('output_dir').value))
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(camera_index)
        try:
            if not cap.isOpened():
                raise RuntimeError(f'camera {camera_index} could not be opened')

            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f'camera {camera_index} returned no frame')
        finally:
            cap.release()

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        safe_waypoint_id = self._safe_filename_part(waypoint_id)
        image_path = output_dir / f'{safe_waypoint_id}_{timestamp}.jpg'

        if not cv2.imwrite(str(image_path), frame):
            raise RuntimeError(f'failed to write image: {image_path}')

        return str(image_path)

    def _publish_done(self, waypoint_id: str, status: str, image_path: str, error: str):
        publish_json = bool(self.get_parameter('publish_json').value)
        out = String()

        if publish_json:
            out.data = json.dumps(
                {
                    'waypoint_id': waypoint_id,
                    'status': status,
                    'image_path': image_path,
                    'error': error,
                },
                separators=(',', ':'),
            )
        else:
            out.data = waypoint_id

        self._done_pub.publish(out)
        self.get_logger().info(f'published photo_done for waypoint {waypoint_id}')

    @staticmethod
    def _safe_filename_part(value: str) -> str:
        safe = re.sub(r'[^A-Za-z0-9_.-]+', '_', value).strip('._')
        return safe or 'waypoint'


def main():
    rclpy.init()
    node = PhotoCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
