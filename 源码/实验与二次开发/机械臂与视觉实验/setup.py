import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'test_stand'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.*'))),
        (os.path.join('share', package_name, 'test_stand'), glob(os.path.join('test_stand', '*.*'))),
        (os.path.join('share', package_name, 'test_stand', 'arm_control'), glob(os.path.join('test_stand', 'arm_control', '*.*'))),
        (os.path.join('share', package_name, 'color_sorting'), glob(os.path.join('test_stand', 'color_sorting', '*.*'))),
        (os.path.join('share', package_name, 'color_sorting', 'config'), glob(os.path.join('test_stand', 'color_sorting', 'config', '*.*'))),
        (os.path.join('share', package_name, 'color_sorting', 'hed_model'), glob(os.path.join('test_stand', 'color_sorting', 'hed_model', '*.*'))),
        (os.path.join('share', package_name, 'test_stand', 'vllm_track_arm'), glob(os.path.join('test_stand', 'vllm_track_arm', '*.*'))),
        (os.path.join('share', package_name, 'test_stand', 'resources', 'models'), glob(os.path.join('test_stand', 'resources', 'models', '*.*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=False,
    maintainer='ubuntu',
    maintainer_email='2436210442@qq.com',
    description='Test stand LLM servo control',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'llm_control_servo = test_stand.arm_control.llm_control_servo:main',
            'llm_control_servo_offline = test_stand.arm_control.llm_control_servo_offline:main',
            'vllm_with_camera = test_stand.vllm_with_camera:main',
            'vllm_track_arm = test_stand.vllm_track_arm.vllm_track_arm:main',
            'llm_color_sorting = test_stand.color_sorting.llm_object_sorting:main',
            'object_sorting = test_stand.color_sorting.object_sorting:main',
            'object_transport = test_stand.object_transport.object_transport:main',
            'vllm_object_transport = test_stand.object_transport.vllm_object_transport:main',
        ],
    },
)
