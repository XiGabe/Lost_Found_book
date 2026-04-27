#!/bin/bash
WORKSPACE=/home/ubuntu/Lost_Found_book/ros2_ws

gnome-terminal \
--tab -e "zsh -c 'source $HOME/.zshrc; source /opt/ros/humble/setup.zsh; sudo systemctl stop start_app_node.service; rviz2 -d ${WORKSPACE}/src/navigation/rviz/navigation_desktop.rviz; exec zsh'" \
--tab -e "zsh -c 'source $HOME/.zshrc; source /opt/ros/humble/setup.zsh; cd ${WORKSPACE}; source install/navigation/share/navigation/package.zsh; export need_compile=False; export HOST=/; export MASTER=/; ros2 launch ${WORKSPACE}/src/navigation/launch/navigation.launch.py map:=map_01; exec zsh'"
