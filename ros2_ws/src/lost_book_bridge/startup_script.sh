#!/bin/bash

source ~/.bashrc

source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
source ~/microros_ws/install/setup.bash

# Track PIDs
PIDS=()

# Wait for the TOF micro-ROS MCU to appear by its stable udev symlink
# udev rule: /etc/udev/rules.d/99-microros-tof.rules
# SUBSYSTEM=="tty", ATTRS{idVendor}=="239a", ATTRS{idProduct}=="80f1", ATTRS{serial}=="DF6510674F592330", SYMLINK+="ttyTOF", MODE="0666"
wait_and_start_tof_agent() {
    echo "Waiting for TOF micro-ROS device on /dev/ttyTOF..."
    while [[ ! -e /dev/ttyTOF ]]; do
        sleep 1
    done
    echo "Starting micro_ros_agent for TOF MCU on /dev/ttyTOF"
    ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyTOF -b 115200 &
    PIDS+=($!)
}

# Cleanup function on Ctrl+C
cleanup() {
    echo "Caught Ctrl+C, killing child processes..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null
    done
    kill 0
    exit 0
}

trap cleanup SIGINT

# Start micro-ROS agent for TOF sensors
wait_and_start_tof_agent &
PIDS+=($!)

# Run the docker bridge
ros2 run lost_book_bridge bridge & 
PIDS+=($!)

# Wait for all background jobs
wait

