from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    request_topic_arg = DeclareLaunchArgument(
        'request_topic',
        default_value='/photo_request',
    )
    done_topic_arg = DeclareLaunchArgument(
        'done_topic',
        default_value='/photo_done',
    )
    camera_index_arg = DeclareLaunchArgument(
        'camera_index',
        default_value='0',
    )
    output_dir_arg = DeclareLaunchArgument(
        'output_dir',
        default_value='output/photos',
    )
    mock_mode_arg = DeclareLaunchArgument(
        'mock_mode',
        default_value='false',
    )
    mock_delay_sec_arg = DeclareLaunchArgument(
        'mock_delay_sec',
        default_value='0.0',
    )
    publish_json_arg = DeclareLaunchArgument(
        'publish_json',
        default_value='false',
    )

    camera_node = Node(
        package='lost_book_bridge',
        executable='photo_camera_node',
        name='photo_camera_node',
        output='screen',
        parameters=[{
            'request_topic': LaunchConfiguration('request_topic'),
            'done_topic': LaunchConfiguration('done_topic'),
            'camera_index': ParameterValue(
                LaunchConfiguration('camera_index'),
                value_type=int,
            ),
            'output_dir': LaunchConfiguration('output_dir'),
            'mock_mode': ParameterValue(
                LaunchConfiguration('mock_mode'),
                value_type=bool,
            ),
            'mock_delay_sec': ParameterValue(
                LaunchConfiguration('mock_delay_sec'),
                value_type=float,
            ),
            'publish_json': ParameterValue(
                LaunchConfiguration('publish_json'),
                value_type=bool,
            ),
        }],
    )

    return LaunchDescription([
        request_topic_arg,
        done_topic_arg,
        camera_index_arg,
        output_dir_arg,
        mock_mode_arg,
        mock_delay_sec_arg,
        publish_json_arg,
        camera_node,
    ])
