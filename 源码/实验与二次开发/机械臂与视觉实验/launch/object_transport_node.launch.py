import os
from ament_index_python.packages import get_package_share_directory

from launch_ros.actions import Node
from launch import LaunchDescription, LaunchService
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, OpaqueFunction


def launch_setup(context):
    compiled = os.environ['need_compile']
    start = LaunchConfiguration('start', default='false')
    start_arg = DeclareLaunchArgument('start', default_value=start)
    display = LaunchConfiguration('display', default='true')
    display_arg = DeclareLaunchArgument('display', default_value=display)
    camera_topic = LaunchConfiguration('camera_topic', default='/depth_cam/rgb0/image_raw')
    camera_topic_arg = DeclareLaunchArgument('camera_topic', default_value=camera_topic)
    use_depth = LaunchConfiguration('use_depth', default='true')
    use_depth_arg = DeclareLaunchArgument('use_depth', default_value=use_depth)
    depth_fuse_rgb_xy = LaunchConfiguration('depth_fuse_rgb_xy', default='true')
    depth_fuse_rgb_xy_arg = DeclareLaunchArgument(
        'depth_fuse_rgb_xy', default_value=depth_fuse_rgb_xy)
    camera_info_topic = camera_topic.perform(context).replace('image_raw', 'camera_info')
    use_depth_val = use_depth.perform(context).lower() in ('true', '1', 'yes')
    depth_fuse_val = depth_fuse_rgb_xy.perform(context).lower() in ('true', '1', 'yes')
    peripherals_package_path = get_package_share_directory('peripherals')
    controller_package_path = get_package_share_directory('controller')
    kinematics_package_path = get_package_share_directory('kinematics')

    depth_camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(peripherals_package_path, 'launch/depth_camera.launch.py')),
    )

    controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(controller_package_path, 'launch/controller.launch.py')),
    )

    kinematics_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(kinematics_package_path, 'launch/kinematics_node.launch.py')),
    )

    object_transport_node = Node(
        package='test_stand',
        executable='object_transport',
        output='screen',
        parameters=[{
            'start': start,
            'display': display,
            'use_depth': use_depth_val,
            'depth_fuse_rgb_xy': depth_fuse_val,
            'rgb_image_topic': camera_topic,
            'rgb_camera_info_topic': camera_info_topic,
            'tf_camera_mount_frame': 'camera_link0',
            'tf_rgb_frame': 'rgb_camera_link',
        }],
    )

    return [
        start_arg,
        display_arg,
        camera_topic_arg,
        use_depth_arg,
        depth_fuse_rgb_xy_arg,
        depth_camera_launch,
        controller_launch,
        kinematics_launch,
        object_transport_node,
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
