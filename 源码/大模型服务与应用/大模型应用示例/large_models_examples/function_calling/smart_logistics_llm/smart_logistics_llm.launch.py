import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def launch_setup(context, *args, **kwargs):
    camera_topic = LaunchConfiguration('camera_topic', default='/depth_cam/rgb0/image_raw')
    camera_topic_arg = DeclareLaunchArgument('camera_topic', default_value=camera_topic)

    mode = LaunchConfiguration('mode', default='1')
    mode_arg = DeclareLaunchArgument('mode', default_value=mode)

    road_network_map = LaunchConfiguration('road_network_map', default='road_network_logistics')
    road_network_map_arg = DeclareLaunchArgument('road_network_map', default_value=road_network_map)

    smart_logistics_enable_display = LaunchConfiguration('smart_logistics_enable_display', default='false')
    smart_logistics_enable_display_arg = DeclareLaunchArgument(
        'smart_logistics_enable_display',
        default_value=smart_logistics_enable_display
    )

    large_models_examples_path = get_package_share_directory('large_models_examples')
    openclaw_controller_path = get_package_share_directory('openclaw_controller')

    smart_scene_navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(openclaw_controller_path, 'launch/smart_scene_navigation.launch.py')
        ),
        launch_arguments={
            'navigation_mode': 'smart_logistics',
            'road_network_map': road_network_map,
            'camera_topic': camera_topic,
            'smart_logistics_enable_display': smart_logistics_enable_display,
        }.items(),
    )

    llm_agent_progress_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                large_models_examples_path,
                'large_models_examples/function_calling/llm_agent_progress.launch.py'
            )
        ),
        launch_arguments={
            'camera_topic': camera_topic,
            'mode': mode,
            'function': 'smart_logistics',
        }.items(),
    )

    return [
        camera_topic_arg,
        mode_arg,
        road_network_map_arg,
        smart_logistics_enable_display_arg,
        smart_scene_navigation_launch,
        llm_agent_progress_launch,
    ]


def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function=launch_setup)
    ])
