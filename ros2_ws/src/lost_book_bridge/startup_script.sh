#!/bin/bash

WORKSPACE=/home/ubuntu/Lost_Found_book/ros2_ws
CV_CONTAINER=cv_cont
CV_WORKDIR=/workspace
MAP_NAME=map_01
RVIZ_CONFIG=${WORKSPACE}/src/navigation/rviz/navigation_desktop.rviz
NAV_LAUNCH=${WORKSPACE}/src/navigation/launch/navigation.launch.py

# This script keeps the original startup behavior:
#   - wait for the TOF micro-ROS device
#   - start the micro_ros_agent
#   - start the lost_book_bridge node
# It also opens the CV, navigation, and RViz terminals needed for the
# lost-book waypoint workflow.

bridge_command="
source ~/.bashrc
source /opt/ros/humble/setup.bash
source ${WORKSPACE}/install/lost_book_bridge/share/lost_book_bridge/package.bash
source ~/microros_ws/install/setup.bash

PIDS=()

wait_and_start_tof_agent() {
    echo 'Waiting for TOF micro-ROS device on /dev/ttyTOF...'
    while [[ ! -e /dev/ttyTOF ]]; do
        sleep 1
    done
    echo 'Starting micro_ros_agent for TOF MCU on /dev/ttyTOF'
    ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyTOF -b 115200 &
    PIDS+=(\$!)
}

cleanup() {
    echo 'Caught Ctrl+C, killing child processes...'
    for pid in \"\${PIDS[@]}\"; do
        kill \"\$pid\" 2>/dev/null
    done
    kill 0
    exit 0
}

trap cleanup SIGINT

wait_and_start_tof_agent &
PIDS+=(\$!)

ros2 run lost_book_bridge bridge &
PIDS+=(\$!)

wait
exec bash
"

cv_command="
sudo docker start ${CV_CONTAINER}
sudo docker exec -it ${CV_CONTAINER} /bin/bash -lc 'cd ${CV_WORKDIR}; python modules/cv_mock.py; exec bash'
exec bash
"

nav_command="
cd ${WORKSPACE}
source /opt/ros/humble/setup.bash
source install/navigation/share/navigation/package.bash
export need_compile=False
export HOST=/
export MASTER=/
ros2 launch ${NAV_LAUNCH} map:=${MAP_NAME}
exec bash
"

rviz_command="
source /opt/ros/humble/setup.bash
rviz2 -d ${RVIZ_CONFIG}
exec bash
"

if command -v gnome-terminal >/dev/null 2>&1; then
    gnome-terminal \
        --tab --title="CV real" -- bash -lc "${cv_command}" \
        --tab --title="Bridge + TOF" -- bash -lc "${bridge_command}" \
        --tab --title="Navigation" -- bash -lc "${nav_command}" \
        --tab --title="RViz" -- bash -lc "${rviz_command}"
else
    echo "gnome-terminal is not available."
    echo "Please run these commands manually in four terminals:"
    echo
    echo "Terminal 1 - CV real:"
    echo "sudo docker start ${CV_CONTAINER}"
    echo "sudo docker exec -it ${CV_CONTAINER} /bin/bash"
    echo "cd ${CV_WORKDIR}"
    echo "python modules/cv_mock.py"
    echo
    echo "Terminal 2 - Bridge + TOF:"
    echo "cd ${WORKSPACE}"
    echo "source /opt/ros/humble/setup.bash"
    echo "source install/lost_book_bridge/share/lost_book_bridge/package.bash"
    echo "source ~/microros_ws/install/setup.bash"
    echo "ros2 run lost_book_bridge bridge"
    echo
    echo "Terminal 3 - Navigation:"
    echo "cd ${WORKSPACE}"
    echo "source /opt/ros/humble/setup.bash"
    echo "source install/navigation/share/navigation/package.bash"
    echo "export need_compile=False"
    echo "export HOST=/"
    echo "export MASTER=/"
    echo "ros2 launch ${NAV_LAUNCH} map:=${MAP_NAME}"
    echo
    echo "Terminal 4 - RViz:"
    echo "rviz2 -d ${RVIZ_CONFIG}"
fi
