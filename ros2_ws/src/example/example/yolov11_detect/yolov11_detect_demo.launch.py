import os
from ament_index_python.packages import get_package_share_directory

from launch_ros.actions import Node
from launch import LaunchDescription, LaunchService
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, OpaqueFunction

def launch_setup(context):
    compiled = os.environ['need_compile']
    if compiled == 'True':
        peripherals_package_path = get_package_share_directory('peripherals')
    else:
        peripherals_package_path = '/home/ubuntu/ros2_ws/src/peripherals'

    depth_camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(peripherals_package_path, 'launch/depth_camera.launch.py')),
    )

    camera_topic = '/depth_cam/rgb/image_raw'
    if 'Pro' in os.environ['MACHINE_TYPE']:
        camera_topic = '/usb_cam/image_raw'

    yolov11_detect_demo_node =  Node(
                package='example',
                executable='yolov11_detect_demo',
                name='yolov11_detect_demo',
                parameters=[
                    {'start': True,},
                    {'image_topic': camera_topic},
                ],
                output='screen'
    )

    return [
            depth_camera_launch,
            yolov11_detect_demo_node,
            ]

def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function = launch_setup)
    ])

if __name__ == '__main__':
    # 创建一个LaunchDescription对象(create a LaunchDescription object)
    ld = generate_launch_description()

    ls = LaunchService()
    ls.include_launch_description(ld)
    ls.run()

