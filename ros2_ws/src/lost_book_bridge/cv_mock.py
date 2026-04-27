"""
Mock CV client for testing the ROS2 bridge.

In production, the real CV code runs in Docker with -p 5555:5555 -p 5556:5556.
This script mimics that — run it natively, then start the bridge container with
--network=host so its localhost resolves here.

  python3 cv_mock.py
  docker run --network=host my_ros2 python3 bridge.py
"""

import zmq

ctx = zmq.Context()

pull = ctx.socket(zmq.PULL)
pull.bind("tcp://*:5555")

push = ctx.socket(zmq.PUSH)
push.bind("tcp://*:5556")

print("CV mock ready — waiting for bridge...")

while True:
    msg = pull.recv_string() # This is when robot sends to take photo
    print(f"Received: '{msg}'")

    # Code to take and process photo should go here
    input("Press Enter to simulate photo taken and processed...")

    #This is CV saying that it has taken the photo, robot can move
    push.send_string("ready")
    print("Sent: 'ready'\n")
