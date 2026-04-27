#!/bin/bash

#DEVICE_ARGS=""
#for DEV in /dev/ttyUSB0 /dev/ttyUSB1 /dev/ttyACM0 /dev/ttyACM1; do
#  if [ -e "$DEV" ]; then
#    DEVICE_ARGS="--device=$DEV:$DEV"
#    echo "Found device: $DEV"
#    break
#  fi
#done

if docker ps -q -f name=my_ros2_container | grep -q .; then
  docker exec -it -w /root/ros2_ws my_ros2_container bash
else
  docker rm -f my_ros2_container 2>/dev/null
  docker run -it \
  --name my_ros2_container \
  --network=host \
  --privileged \
  -v /home/$SUDO_USER/ros2_ws:/root/ros2_ws \
  my_ros2 \
  bash -c "cd /root/ros2_ws && exec bash"
fi
