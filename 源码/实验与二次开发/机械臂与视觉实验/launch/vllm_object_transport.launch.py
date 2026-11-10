import os
from ament_index_python.packages import get_package_share_directory

from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch import LaunchDescription, LaunchService
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, OpaqueFunction


def launch_setup(context):
    mode = LaunchConfiguration('mode', default=1)
    mode_arg = DeclareLaunchArgument('mode', default_value=mode)

    camera_topic = LaunchConfiguration('camera_topic', default='/depth_cam/rgb0/image_raw')
    camera_topic_arg = DeclareLaunchArgument('camera_topic', default_value=camera_topic)

    large_models_package_path = get_package_share_directory('large_models')
    test_stand_package_path = get_package_share_directory('test_stand')

    object_transport_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(test_stand_package_path, 'launch', 'object_transport_node.launch.py')),
        launch_arguments={'camera_topic': camera_topic}.items(),
    )

    large_models_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(large_models_package_path, 'launch/start.launch.py')),
        launch_arguments={
            'mode': mode,
            'camera_topic': camera_topic,
        }.items(),
    )

    camera_topic_str = camera_topic.perform(context)
    camera_info_topic_str = camera_topic_str.replace('image_raw', 'camera_info')
    vllm_object_transport_node = Node(
        package='test_stand',
        executable='vllm_object_transport',
        output='screen',
        parameters=[{
            'camera_topic': camera_topic_str,
            'camera_info_topic': camera_info_topic_str,
        }],
    )

    return [
        mode_arg,
        camera_topic_arg,
        object_transport_launch,
        large_models_launch,
        vllm_object_transport_node,
    ]


def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function=launch_setup),
    ])


if __name__ == '__main__':
    ld = generate_launch_description()
    ls = LaunchService()
    ls.include_launch_description(ld)
    ls.run()
