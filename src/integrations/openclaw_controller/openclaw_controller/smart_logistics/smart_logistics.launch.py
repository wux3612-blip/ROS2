import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription, LaunchService
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context):
    compiled = os.environ['need_compile']
    camera_topic = LaunchConfiguration('camera_topic', default='/depth_cam/rgb0/image_raw')
    camera_topic_arg = DeclareLaunchArgument('camera_topic', default_value=camera_topic)

    enable_display = LaunchConfiguration('enable_display', default='true')
    enable_display_arg = DeclareLaunchArgument('enable_display', default_value=enable_display)

    if compiled == 'True':
        openclaw_controller_package_path = get_package_share_directory('openclaw_controller')
    else:
        openclaw_controller_package_path = '/home/ubuntu/ros2_ws/src/openclaw_controller'

    device_base_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(openclaw_controller_package_path, 'launch/include/device_base.launch.py')),
    )

    config_path = os.path.join(openclaw_controller_package_path, 'openclaw_controller/smart_logistics/config/smart_logistics.yaml')
    if compiled == 'True':
        config_path = os.path.join(openclaw_controller_package_path, 'smart_logistics/config/smart_logistics.yaml')

    smart_logistics_node = Node(
        package='openclaw_controller',
        executable='smart_logistics',
        output='screen',
        parameters=[config_path, {
            'camera_topic': camera_topic,
            'config_path': config_path,
            'enable_display': enable_display,
        }],
    )

    return [
        camera_topic_arg,
        enable_display_arg,
        device_base_launch,
        smart_logistics_node,
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
