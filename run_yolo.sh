#!/bin/bash

# 定义镜像名称变量，方便以后升级
IMAGE="ultralytics/ultralytics:latest-jetson-jetpack6"

# 启动命令
sudo docker run -it \
    --ipc=host \
    --runtime=nvidia \
    --gpus all \
    --name yolo_dev \
    -v /home/$USER/Documents/Lost_Found_book:/workspace \
    -v /tmp/.x11-unix:/tmp/.x11-unix \
    -e DISPLAY=$DISPLAY \
    -w /workspace \
    $IMAGE

#sudo docker start yolo_dev
#sudo docker exec -it yolo_dev /bin/bash