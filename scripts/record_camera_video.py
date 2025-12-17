#!/usr/bin/env python3
"""
ROS2 Node: Record Camera Stream to Video
- Subscribes to RealSense camera image topic from OrangePi
- Records frames to MP4 video file
- Displays recording status

Usage:
    ros2 run human_object_detection record_camera_video
    ros2 run human_object_detection record_camera_video --ros-args -p output_file:=my_video.mp4
    
Controls:
    Ctrl+C - Stop recording and save video
"""

import sys
import os
import cv2
import numpy as np
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class RecordCameraVideoNode(Node):
    def __init__(self):
        super().__init__('record_camera_video_node')
        
        # Declare parameters
        self.declare_parameter('image_topic', '/camera/camera_down/color/image_raw')
        self.declare_parameter('output_file', '')  # Empty = auto-generate filename
        self.declare_parameter('fps', 15.0)  # Target FPS for output video
        self.declare_parameter('show_preview', True)  # Show live preview window
        
        # Get parameters
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        output_file = self.get_parameter('output_file').get_parameter_value().string_value
        self.target_fps = self.get_parameter('fps').get_parameter_value().double_value
        self.show_preview = self.get_parameter('show_preview').get_parameter_value().bool_value
        
        # Generate output filename if not provided
        if not output_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f'camera_recording_{timestamp}.mp4'
        
        self.output_file = output_file
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Video writer (will be initialized when first frame arrives)
        self.video_writer = None
        self.frame_count = 0
        self.start_time = None
        self.frame_width = None
        self.frame_height = None
        
        # QoS profile - VOLATILE to match OrangePi RealSense publisher
        camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE
        )
        
        # Subscribe to image topic
        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            camera_qos
        )
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("Camera Video Recorder (ROS2)")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"Subscribing to: {self.image_topic}")
        self.get_logger().info(f"Output file: {self.output_file}")
        self.get_logger().info(f"Target FPS: {self.target_fps}")
        self.get_logger().info(f"Preview: {'Enabled' if self.show_preview else 'Disabled'}")
        self.get_logger().info("=" * 60)
        self.get_logger().info("Waiting for camera frames...")
        self.get_logger().info("Press Ctrl+C to stop recording")
        self.get_logger().info("=" * 60)
    
    def image_callback(self, msg):
        """Process incoming image and write to video."""
        try:
            # Convert ROS image to OpenCV
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Initialize video writer on first frame
            if self.video_writer is None:
                self.frame_height, self.frame_width = frame.shape[:2]
                self.start_time = self.get_clock().now()
                
                # Use mp4v codec for MP4 output
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(
                    self.output_file,
                    fourcc,
                    self.target_fps,
                    (self.frame_width, self.frame_height)
                )
                
                if not self.video_writer.isOpened():
                    self.get_logger().error(f"Failed to open video writer for {self.output_file}")
                    return
                
                self.get_logger().info(f"Recording started: {self.frame_width}x{self.frame_height} @ {self.target_fps} FPS")
            
            # Write frame to video
            self.video_writer.write(frame)
            self.frame_count += 1
            
            # Calculate recording duration
            current_time = self.get_clock().now()
            duration = (current_time - self.start_time).nanoseconds / 1e9
            
            # Log progress every 30 frames (~1-2 seconds)
            if self.frame_count % 30 == 0:
                actual_fps = self.frame_count / duration if duration > 0 else 0
                self.get_logger().info(f"Recording: {self.frame_count} frames, {duration:.1f}s, {actual_fps:.1f} FPS")
            
            # Show preview if enabled
            if self.show_preview:
                # Add recording info overlay
                overlay = frame.copy()
                cv2.putText(overlay, f"REC", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.circle(overlay, (80, 22), 8, (0, 0, 255), -1)  # Red dot
                cv2.putText(overlay, f"Frames: {self.frame_count} | Time: {duration:.1f}s", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow('Recording Preview', overlay)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    self.get_logger().info("Stopping recording (user pressed quit)...")
                    rclpy.shutdown()
            
        except Exception as e:
            self.get_logger().error(f"Error processing frame: {e}")
    
    def cleanup(self):
        """Clean up resources and finalize video."""
        if self.video_writer is not None:
            self.video_writer.release()
            
            if self.frame_count > 0:
                duration = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
                actual_fps = self.frame_count / duration if duration > 0 else 0
                
                self.get_logger().info("=" * 60)
                self.get_logger().info("Recording Complete!")
                self.get_logger().info(f"  Output file: {self.output_file}")
                self.get_logger().info(f"  Total frames: {self.frame_count}")
                self.get_logger().info(f"  Duration: {duration:.2f} seconds")
                self.get_logger().info(f"  Actual FPS: {actual_fps:.1f}")
                self.get_logger().info(f"  Resolution: {self.frame_width}x{self.frame_height}")
                self.get_logger().info("=" * 60)
            else:
                self.get_logger().warn("No frames were recorded!")
        
        if self.show_preview:
            cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = RecordCameraVideoNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup()
        node.get_logger().info("Shutting down video recorder...")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
