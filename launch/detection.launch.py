#!/usr/bin/env python3
"""
ROS2 Launch file for Detection and Distance Measurement Node
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    object_prompt_arg = DeclareLaunchArgument(
        'object_prompt',
        default_value='Grey LiPo battery',
        description='Text prompt for object detection (camera_down)'
    )
    
    human_prompt_arg = DeclareLaunchArgument(
        'human_prompt',
        default_value='human',
        description='Text prompt for human detection (camera_front)'
    )
    
    box_threshold_arg = DeclareLaunchArgument(
        'box_threshold',
        default_value='0.30',
        description='Bounding box confidence threshold'
    )
    
    text_threshold_arg = DeclareLaunchArgument(
        'text_threshold',
        default_value='0.20',
        description='Text matching threshold'
    )
    
    drone_frame_arg = DeclareLaunchArgument(
        'drone_frame',
        default_value='drone',
        description='Frame ID for published messages'
    )
    
    # Detection and Distance Node
    detection_node = Node(
        package='detection_and_distance',
        executable='detection_node.py',
        name='detection_and_distance_node',
        output='screen',
        parameters=[{
            'object_prompt': LaunchConfiguration('object_prompt'),
            'human_prompt': LaunchConfiguration('human_prompt'),
            'box_threshold': LaunchConfiguration('box_threshold'),
            'text_threshold': LaunchConfiguration('text_threshold'),
            'drone_frame': LaunchConfiguration('drone_frame'),
        }]
    )
    
    return LaunchDescription([
        object_prompt_arg,
        human_prompt_arg,
        box_threshold_arg,
        text_threshold_arg,
        drone_frame_arg,
        detection_node,
    ])
