"""
CV Mock Client for ROS2 Bridge

Listens for "Stoped" command from robot, captures photo, runs vision pipeline,
saves result to output folder, then sends "ready" back.

Usage:
  python3 cv_mock.py
"""

import zmq
import cv2
import sys
from pathlib import Path
from datetime import datetime

# Add project path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.vision_pipeline import VisionPipeline

# Initialize VisionPipeline
print("Loading VisionPipeline...")
vision = VisionPipeline()
print("VisionPipeline ready\n")

ctx = zmq.Context()

pull = ctx.socket(zmq.PULL)
pull.bind("tcp://*:5555")

push = ctx.socket(zmq.PUSH)
push.bind("tcp://*:5556")

# Camera config (change index if needed)
CAMERA_INDEX = 0

while True:
    msg = pull.recv_string()
    print(f"Received: '{msg}'")

    if msg == "stopped":
        # Capture photo from camera
        cap = cv2.VideoCapture(CAMERA_INDEX)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("Failed to capture image from camera")
            push.send_string("ready")
            print("Sent: 'ready'\n")
            continue

        # Send ready back to robot
        push.send_string("ready")
        print("Sent: 'ready' — robot can move to next position\n")

        # Save captured image with unique name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        img_path = Path(__file__).parent.parent / "output" / f"capture_{timestamp}.jpg"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(img_path), frame)
        print(f"Image saved: {img_path}")

        # Run vision pipeline
        print("Running vision.check...")
        result = vision.check(frame)

    else:
        print(f"Unknown command ignored: {msg}")