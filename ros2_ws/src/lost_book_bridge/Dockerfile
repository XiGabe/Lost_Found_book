FROM docker.io/osrf/ros:humble-desktop-jammy

COPY --from=docker.io/microros/micro-ros-agent:humble /uros_ws /uros_ws

RUN apt-get update && apt-get install -y python3-pip && rm -rf /var/lib/apt/lists/*

RUN pip3 install pyzmq

# Build custom msgs into the image
RUN mkdir -p /msgs_ws/src && cd /msgs_ws/src && \
    git clone https://github.com/leslier7/lost_book_msgs.git && \
    cd /msgs_ws && \
    . /opt/ros/humble/setup.sh && \
    colcon build && \
    rm -rf build log

RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
RUN echo "source /uros_ws/install/setup.bash" >> ~/.bashrc
RUN echo "source /msgs_ws/install/setup.bash" >> ~/.bashrc
