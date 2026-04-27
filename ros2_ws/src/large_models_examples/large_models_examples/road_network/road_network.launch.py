import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import OpaqueFunction, IncludeLaunchDescription, ExecuteProcess,DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node


def launch_setup(context):
    compiled = os.environ.get('need_compile', 'False')

    conf = LaunchConfiguration('conf', default='0.45')
    camera_topic = LaunchConfiguration('camera_topic',default='/depth_cam/rgb/image_raw')
    use_yolo_detect = LaunchConfiguration('use_yolo_detect',default='true')

    map_name = LaunchConfiguration('map', default='map_01')
    robot_name = LaunchConfiguration('robot_name',default=os.environ.get('HOST', 'robot'))
    master_name = LaunchConfiguration('master_name',default=os.environ.get('MASTER', 'master'))

    file_name = LaunchConfiguration('file_name', default='road_network')
    file_name_arg = DeclareLaunchArgument('file_name', default_value=file_name)

    camera_type = os.environ.get('DEPTH_CAMERA_TYPE', '')
    if compiled == 'True':
        large_models_example_package_path = get_package_share_directory(
            'large_models_examples'
        )
        controller_package_path = get_package_share_directory('controller')
    else:
        large_models_example_package_path = (
            '/home/ubuntu/ros2_ws/src/large_models_examples/large_models_examples'
        )
        controller_package_path = '/home/ubuntu/ros2_ws/src/controller'


    navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                large_models_example_package_path,
                'road_network/include/navigation.launch.py'
            )
        ),
        launch_arguments={
            'sim': 'false',
            'map': map_name,
            'robot_name': robot_name,
            'master_name': master_name,
            'use_rpp': 'true',
            'use_teb': 'false',
        }.items(),
    )


    rviz_name = 'navigation_road_network.rviz'
    camera_topic = '/depth/rgb/image_raw'
    if 'usb_cam' in camera_type:
        rviz_name = 'navigation_road_network_usb.rviz'
        camera_topic = '/usb_cam/image'

    rviz_node = ExecuteProcess(
        cmd=[
            'rviz2',
            '-d',
            os.path.join(
                large_models_example_package_path,
                'road_network/config',rviz_name
            ),
        ],
        output='screen'
    )

    navigation_controller_node = Node(
        package='large_models_examples',
        executable='nav2_execution_node',
        name='nav2_execution_node',
        output='screen'
    )

    road_network_navigator_node = Node(
        package='large_models_examples',
        executable='road_network_navigator',
        name='road_network_navigator',
        output='screen',
        parameters=[{
            'file_name':file_name,
            'use_yolo_detect': use_yolo_detect,
        }]
    )


    yolov_node = Node(
        package='example',
        executable='yolov11_node',
        output='screen',
        parameters=[{'camera': camera_type, 'task': 'detect', 'engine': 'best_traffic.engine', 'conf': 0.6, 'display_only': False}]
    )

    return [
        file_name_arg,
        navigation_launch,
        rviz_node,
        yolov_node,
        navigation_controller_node,
        road_network_navigator_node,
    ]


def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function=launch_setup)
    ])


if __name__ == '__main__':
    ld = generate_launch_description()

    ls = LaunchService()
    ls.include_launch_description(ld)
    ls.run()
