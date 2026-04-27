#!/bin/bash

if ! docker image inspect docker.io/osrf/ros:humble-desktop-jammy > /dev/null 2>&1; then
  echo "Base image not found, pulling..."
  docker pull docker.io/osrf/ros:humble-desktop-jammy
fi

docker build -t my_ros2 .
