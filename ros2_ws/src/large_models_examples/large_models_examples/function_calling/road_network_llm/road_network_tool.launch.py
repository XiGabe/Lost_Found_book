import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import OpaqueFunction, IncludeLaunchDescription,DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def launch_setup(context, *args, **kwargs):

    file_name = LaunchConfiguration('file_name',default='self_driving_network')
    file_name_arg = DeclareLaunchArgument('file_name', default_value=file_name)

    camera_type = os.environ.get('DEPTH_CAMERA_TYPE', '')
    if 'usb_cam' in camera_type:
        camera_topic_value = '/usb_cam/image'
    else:
        camera_topic_value = '/depth_cam/rgb/image_raw'

    camera_topic = LaunchConfiguration('camera_topic',default=camera_topic_value)

    llm_agent_progress_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('large_models_examples'),
                'large_models_examples/function_calling/llm_agent_progress.launch.py'
            )
        ),
        launch_arguments={
            'camera_topic': camera_topic,
            'function': 'road_network'
        }.items(),
    )

    road_network_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('large_models_examples'),
                'large_models_examples/road_network/road_network.launch.py'
            )
        ),
        launch_arguments={
            'camera_topic': camera_topic,
            'file_name': file_name,
        }.items(),
    )

    return [
        file_name_arg,
        llm_agent_progress_launch,
        road_network_launch,
    ]


def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function=launch_setup)
    ])
