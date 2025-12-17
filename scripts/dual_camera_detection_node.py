#!/usr/bin/env python3
"""
ROS2 Node: Dual Camera Human and Object Detection
- Front camera: Object detection (GroundingDINO) - pitched down ~30 degrees
- Down camera: Human detection (GroundingDINO) + pose orientation (MediaPipe) - facing forward, 0° pitch
- Publishes: /local_human_position, /local_object_position, and /global_object_position (PoseStamped)

Subscribes to (Jetson Nano topics):
    - /camera/camera_front/color/image_raw
    - /camera/camera_front/depth/image_rect_raw
    - /camera/camera_front/color/camera_info
    - /camera/camera_down/color/image_raw
    - /camera/camera_down/depth/image_rect_raw
    - /camera/camera_down/color/camera_info
    - /spatial_drone/mavros/local_position/pose (priority) or /spatial_drone/mavros/vision_pose/pose (fallback)
"""

import sys
import os
import numpy as np
import cv2
import time

# Force Matplotlib to use non-interactive backend
import matplotlib
matplotlib.use('Agg')

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.duration import Duration
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import message_filters

# For quaternion from euler
try:
    from tf_transformations import quaternion_from_euler, quaternion_multiply, quaternion_matrix
except ImportError:
    # Fallback implementation if tf_transformations is not available
    import math
    def quaternion_from_euler(roll, pitch, yaw):
        """Convert euler angles to quaternion."""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        q = [0] * 4
        q[0] = sr * cp * cy - cr * sp * sy  # x
        q[1] = cr * sp * cy + sr * cp * sy  # y
        q[2] = cr * cp * sy - sr * sp * cy  # z
        q[3] = cr * cp * cy + sr * sp * sy  # w
        return q
    
    def quaternion_multiply(q1, q2):
        """Multiply two quaternions (x, y, z, w format)."""
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return [
            w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
            w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
            w1*z2 + x1*y2 - y1*x2 + z1*w2,  # z
            w1*w2 - x1*x2 - y1*y2 - z1*z2   # w
        ]
    
    def quaternion_matrix(quaternion):
        """Return homogeneous rotation matrix from quaternion (x, y, z, w)."""
        x, y, z, w = quaternion
        n = w*w + x*x + y*y + z*z
        if n == 0.0:
            return np.identity(4)
        s = 2.0 / n
        wx = s * w * x
        wy = s * w * y
        wz = s * w * z
        xx = s * x * x
        xy = s * x * y
        xz = s * x * z
        yy = s * y * y
        yz = s * y * z
        zz = s * z * z
        return np.array([
            [1.0 - (yy + zz), xy - wz, xz + wy, 0.0],
            [xy + wz, 1.0 - (xx + zz), yz - wx, 0.0],
            [xz - wy, yz + wx, 1.0 - (xx + yy), 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

# Import MediaPipe
import mediapipe as mp

# Add GroundingDINO to path
possible_gdino_paths = [
    '/home/imit-learn/James/catkin_ws/Object-Detection-and-Distance-Measurement/GroundingDINO',
    '/home/imit-learn/James/Object-Detection-and-Distance-Measurement/GroundingDINO',
    os.environ.get('GROUNDINGDINO_PATH', ''),
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'GroundingDINO'),
]

gdino_path = None
for path in possible_gdino_paths:
    if path and os.path.exists(path):
        gdino_path = path
        sys.path.insert(0, gdino_path)
        break

if gdino_path is None:
    print("ERROR: GroundingDINO not found in any expected location")
    print(f"Searched: {[p for p in possible_gdino_paths if p]}")
    sys.exit(1)

# Grounding DINO imports
import torch
from PIL import Image as PILImage
from groundingdino.util.inference import load_model, predict
import groundingdino.datasets.transforms as T


class DualCameraDetectionNode(Node):
    def __init__(self):
        super().__init__('dual_camera_detection_node')
        
        # Declare parameters
        self.declare_parameter('object_prompt', 'Orange  colored cuboid')
        self.declare_parameter('human_prompt', 'human')
        self.declare_parameter('box_threshold', 0.30)
        self.declare_parameter('text_threshold', 0.20)
        self.declare_parameter('min_detection_confidence', 0.5)
        self.declare_parameter('min_tracking_confidence', 0.5)
        self.declare_parameter('drone_frame', 'base_link')
        self.declare_parameter('front_camera_frame', 'camera_front_color_optical_frame')
        self.declare_parameter('down_camera_frame', 'camera_down_color_optical_frame')
        self.declare_parameter('global_frame', 'map')
        
        # Down camera offset relative to drone body frame (meters)
        # Camera is 9cm forward and 8cm above the drone center, facing straight ahead (0° pitch)
        # This camera is ~2cm above and ~3cm behind the front camera
        self.declare_parameter('down_camera_offset_x', 0.09)  # forward (3cm behind front camera)
        self.declare_parameter('down_camera_offset_y', 0.0)   # left/right
        self.declare_parameter('down_camera_offset_z', 0.08)  # up (positive is above)
        self.declare_parameter('down_camera_pitch_deg', 0.0)  # pitch angle in degrees (0 = horizontal)
        
        # Front camera offset relative to drone body frame (meters)
        # Camera is 12cm forward, centered on Y and Z, pitched down ~30 degrees
        self.declare_parameter('front_camera_offset_x', 0.12)  # forward
        self.declare_parameter('front_camera_offset_y', 0.0)   # left/right
        self.declare_parameter('front_camera_offset_z', 0.0)   # up/down
        self.declare_parameter('front_camera_pitch_deg', -30.0) # pitch angle in degrees (negative = looking down)
        
        # Get parameters
        self.object_prompt = self.get_parameter('object_prompt').get_parameter_value().string_value
        self.human_prompt = self.get_parameter('human_prompt').get_parameter_value().string_value
        self.box_threshold = self.get_parameter('box_threshold').get_parameter_value().double_value
        self.text_threshold = self.get_parameter('text_threshold').get_parameter_value().double_value
        self.min_detection_confidence = self.get_parameter('min_detection_confidence').get_parameter_value().double_value
        self.min_tracking_confidence = self.get_parameter('min_tracking_confidence').get_parameter_value().double_value
        self.drone_frame = self.get_parameter('drone_frame').get_parameter_value().string_value
        self.front_camera_frame = self.get_parameter('front_camera_frame').get_parameter_value().string_value
        self.down_camera_frame = self.get_parameter('down_camera_frame').get_parameter_value().string_value
        self.global_frame = self.get_parameter('global_frame').get_parameter_value().string_value
        
        # Camera offset in drone body frame - Down camera
        self.down_camera_offset = np.array([
            self.get_parameter('down_camera_offset_x').get_parameter_value().double_value,
            self.get_parameter('down_camera_offset_y').get_parameter_value().double_value,
            self.get_parameter('down_camera_offset_z').get_parameter_value().double_value
        ])
        self.down_camera_pitch = np.radians(
            self.get_parameter('down_camera_pitch_deg').get_parameter_value().double_value
        )
        
        # Camera offset in drone body frame - Front camera
        self.front_camera_offset = np.array([
            self.get_parameter('front_camera_offset_x').get_parameter_value().double_value,
            self.get_parameter('front_camera_offset_y').get_parameter_value().double_value,
            self.get_parameter('front_camera_offset_z').get_parameter_value().double_value
        ])
        self.front_camera_pitch = np.radians(
            self.get_parameter('front_camera_pitch_deg').get_parameter_value().double_value
        )
        
        self.get_logger().info("=" * 80)
        self.get_logger().info("Dual Camera Human & Object Detection Node (ROS2)")
        self.get_logger().info("=" * 80)
        self.get_logger().info(f"Object detection prompt: '{self.object_prompt}'")
        self.get_logger().info(f"Human detection prompt: '{self.human_prompt}'")
        self.get_logger().info(f"Box threshold: {self.box_threshold}, Text threshold: {self.text_threshold}")
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Camera intrinsics (will be updated from CameraInfo)
        self.front_intrinsics = None
        self.down_intrinsics = None
        
        # Drone pose (will be updated from mavros topics)
        self.drone_pose = None
        self.drone_pose_timestamp = None
        self.using_local_pose = False  # Track which pose source we're using
        
        # Initialize MediaPipe Pose
        self.get_logger().info("[1/3] Initializing MediaPipe Pose...")
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            model_complexity=1
        )
        self.get_logger().info("✓ MediaPipe Pose initialized")
        
        # Initialize Grounding DINO
        self.get_logger().info("[2/3] Loading Grounding DINO model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"  Using device: {self.device}")
        
        gdino_base = gdino_path
        gdino_config = os.path.join(gdino_base, 'groundingdino/config/GroundingDINO_SwinT_OGC.py')
        gdino_checkpoint = os.path.join(gdino_base, 'weights/groundingdino_swint_ogc.pth')
        
        try:
            self.gdino_model = load_model(gdino_config, gdino_checkpoint, device=self.device)
            self.get_logger().info("✓ Grounding DINO loaded")
        except Exception as e:
            self.get_logger().error(f"Failed to load Grounding DINO: {e}")
            sys.exit(1)
        
        # QoS profile for camera image data - matches RealSense camera publisher settings
        camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        
        # QoS profile for camera_info topics (RELIABLE but VOLATILE durability)
        camera_info_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE
        )
        
        # QoS profile for mavros sensor data (uses BEST_EFFORT)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Publishers
        self.human_pose_pub = self.create_publisher(PoseStamped, '/local_human_position', 10)
        self.object_pose_pub = self.create_publisher(PoseStamped, '/local_object_position', 10)
        self.global_human_pose_pub = self.create_publisher(PoseStamped, '/global_human_position', 10)
        self.global_object_pose_pub = self.create_publisher(PoseStamped, '/global_object_position', 10)
        
        # Visualization marker publishers
        self.human_marker_pub = self.create_publisher(MarkerArray, '/detection_markers/human', 10)
        self.object_marker_pub = self.create_publisher(MarkerArray, '/detection_markers/object', 10)
        self.marker_id = 0  # Counter for marker IDs
        
        self.get_logger().info("[3/3] Setting up subscribers...")
        
        # Subscribe to drone pose (priority: local_position, fallback: vision_pose)
        # Use BEST_EFFORT QoS to match mavros publishers
        self.drone_local_pose_sub = self.create_subscription(
            PoseStamped, '/spatial_drone/mavros/local_position/pose',
            self.drone_local_pose_callback, sensor_qos)
        self.drone_vision_pose_sub = self.create_subscription(
            PoseStamped, '/spatial_drone/mavros/vision_pose/pose',
            self.drone_vision_pose_callback, sensor_qos)
        
        # Subscribe to front camera (object detection) - Jetson topic names
        # Use camera_qos to match RealSense publisher (RELIABLE + TRANSIENT_LOCAL)
        front_color_sub = message_filters.Subscriber(
            self, Image, '/camera/camera_front/color/image_raw', qos_profile=camera_qos)
        front_depth_sub = message_filters.Subscriber(
            self, Image, '/camera/camera_front/depth/image_rect_raw', qos_profile=camera_qos)
        self.front_info_sub = self.create_subscription(
            CameraInfo, '/camera/camera_front/color/camera_info', 
            self.front_info_callback, camera_info_qos)
        
        # Subscribe to down camera (human detection) - Jetson topic names
        # Use camera_qos to match RealSense publisher (RELIABLE + TRANSIENT_LOCAL)
        down_color_sub = message_filters.Subscriber(
            self, Image, '/camera/camera_down/color/image_raw', qos_profile=camera_qos)
        down_depth_sub = message_filters.Subscriber(
            self, Image, '/camera/camera_down/depth/image_rect_raw', qos_profile=camera_qos)
        self.down_info_sub = self.create_subscription(
            CameraInfo, '/camera/camera_down/color/camera_info', 
            self.down_info_callback, camera_info_qos)
        
        # Synchronize front camera topics
        self.front_sync = message_filters.ApproximateTimeSynchronizer(
            [front_color_sub, front_depth_sub],
            queue_size=10,
            slop=0.1
        )
        self.front_sync.registerCallback(self.front_camera_callback)
        
        # Synchronize down camera topics
        self.down_sync = message_filters.ApproximateTimeSynchronizer(
            [down_color_sub, down_depth_sub],
            queue_size=10,
            slop=0.1
        )
        self.down_sync.registerCallback(self.down_camera_callback)
        
        self.get_logger().info("✓ Subscribers ready")
        self.get_logger().info("=" * 80)
        self.get_logger().info("Node ready! Waiting for camera data...")
        self.get_logger().info("Subscribed topics (Jetson Nano):")
        self.get_logger().info("  - /camera/camera_front/color/image_raw")
        self.get_logger().info("  - /camera/camera_front/depth/image_rect_raw")
        self.get_logger().info("  - /camera/camera_down/color/image_raw")
        self.get_logger().info("  - /camera/camera_down/depth/image_rect_raw")
        self.get_logger().info("Subscribed topics (Drone pose):")
        self.get_logger().info("  - /spatial_drone/mavros/local_position/pose (priority)")
        self.get_logger().info("  - /spatial_drone/mavros/vision_pose/pose (fallback)")
        self.get_logger().info("Publishing to:")
        self.get_logger().info("  - /local_human_position (PoseStamped)")
        self.get_logger().info("  - /local_object_position (PoseStamped)")
        self.get_logger().info("  - /global_human_position (PoseStamped)")
        self.get_logger().info("  - /global_object_position (PoseStamped)")
        self.get_logger().info("  - /detection_markers/human (MarkerArray)")
        self.get_logger().info("  - /detection_markers/object (MarkerArray)")
        self.get_logger().info(f"Front camera: offset=({self.front_camera_offset[0]:.2f}, {self.front_camera_offset[1]:.2f}, {self.front_camera_offset[2]:.2f})m, pitch={np.degrees(self.front_camera_pitch):.1f}°")
        self.get_logger().info(f"Down camera: offset=({self.down_camera_offset[0]:.2f}, {self.down_camera_offset[1]:.2f}, {self.down_camera_offset[2]:.2f})m, pitch={np.degrees(self.down_camera_pitch):.1f}°")
        self.get_logger().info("=" * 80)
        
    def front_info_callback(self, msg):
        """Store front camera intrinsics."""
        if self.front_intrinsics is None:
            self.front_intrinsics = {
                'fx': msg.k[0],
                'fy': msg.k[4],
                'cx': msg.k[2],
                'cy': msg.k[5],
                'width': msg.width,
                'height': msg.height
            }
            self.get_logger().info(f"Front camera intrinsics received: fx={msg.k[0]:.2f}, fy={msg.k[4]:.2f}")
    
    def down_info_callback(self, msg):
        """Store down camera intrinsics."""
        if self.down_intrinsics is None:
            self.down_intrinsics = {
                'fx': msg.k[0],
                'fy': msg.k[4],
                'cx': msg.k[2],
                'cy': msg.k[5],
                'width': msg.width,
                'height': msg.height
            }
            self.get_logger().info(f"Down camera intrinsics received: fx={msg.k[0]:.2f}, fy={msg.k[4]:.2f}")
    
    def drone_local_pose_callback(self, msg):
        """Store drone pose from local_position (priority source)."""
        self.drone_pose = msg
        self.drone_pose_timestamp = self.get_clock().now()
        if not self.using_local_pose:
            self.using_local_pose = True
            self.get_logger().info("Using local_position/pose for drone localization")
    
    def drone_vision_pose_callback(self, msg):
        """Store drone pose from vision_pose (fallback source)."""
        # Only use vision_pose if we haven't received local_position recently (within 1 second)
        if self.drone_pose is None or not self.using_local_pose:
            self.drone_pose = msg
            self.drone_pose_timestamp = self.get_clock().now()
        elif self.using_local_pose:
            # Check if local_position is stale (> 1 second old)
            now = self.get_clock().now()
            if self.drone_pose_timestamp is not None:
                age = (now - self.drone_pose_timestamp).nanoseconds / 1e9
                if age > 1.0:
                    self.drone_pose = msg
                    self.drone_pose_timestamp = now
                    self.using_local_pose = False
                    self.get_logger().warn("local_position stale, falling back to vision_pose")
    
    def front_camera_callback(self, color_msg, depth_msg):
        """Process front camera: detect object (Orange  colored cuboid)."""
        if self.front_intrinsics is None:
            return
        
        try:
            # Convert ROS images to OpenCV
            color_image = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='16UC1')
            
            # Detect object with Grounding DINO
            boxes, confidences = self.detect_objects(color_image, self.object_prompt)
            
            if len(boxes) == 0:
                return
            
            # Use the highest confidence detection
            best_idx = np.argmax(confidences)
            box = boxes[best_idx]
            confidence = confidences[best_idx]
            
            # Get bounding box center (clamp to image bounds)
            h, w = color_image.shape[:2]
            depth_h, depth_w = depth_image.shape[:2]
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Clamp to depth image bounds (may differ from color image size)
            center_x = max(0, min(center_x, depth_w - 1))
            center_y = max(0, min(center_y, depth_h - 1))
            
            # Get depth at center
            depth_value = depth_image[center_y, center_x]
            if depth_value == 0:
                return
            
            # Convert to 3D position (in camera optical frame)
            depth_m = depth_value * 0.001  # mm to meters
            x_3d, y_3d, z_3d = self.pixel_to_3d(
                center_x, center_y, depth_m, self.front_intrinsics
            )
            
            # Publish local object pose (no orientation)
            self.publish_object_pose(x_3d, y_3d, z_3d, color_msg.header.stamp)
            
            # Compute 3D bounding box size from 2D box and depth
            box_width_3d = abs((x2 - x1) * depth_m / self.front_intrinsics['fx'])
            box_height_3d = abs((y2 - y1) * depth_m / self.front_intrinsics['fy'])
            box_depth_3d = 0.05  # Assume ~5cm depth for small objects like cube
            
            # Transform to global frame and publish if drone pose is available
            if self.drone_pose is not None:
                global_pos = self.transform_to_global_front_camera(x_3d, y_3d, z_3d)
                if global_pos is not None:
                    self.publish_global_object_pose(global_pos[0], global_pos[1], global_pos[2], 
                                                     color_msg.header.stamp)
                    
                    # Publish visualization markers in global frame
                    self.publish_detection_marker(
                        global_pos[0], global_pos[1], global_pos[2],
                        box_width_3d, box_height_3d, box_depth_3d,
                        self.global_frame, color_msg.header.stamp,
                        is_human=False, confidence=confidence, label=self.object_prompt
                    )
                    
                    self.get_logger().info(f"Object detected: local=({x_3d:.3f}, {y_3d:.3f}, {z_3d:.3f}), "
                                           f"global=({global_pos[0]:.3f}, {global_pos[1]:.3f}, {global_pos[2]:.3f}), "
                                           f"conf={confidence:.2f}")
                else:
                    # Publish markers in local frame as fallback
                    self.publish_detection_marker(
                        x_3d, y_3d, z_3d,
                        box_width_3d, box_height_3d, box_depth_3d,
                        self.front_camera_frame, color_msg.header.stamp,
                        is_human=False, confidence=confidence, label=self.object_prompt
                    )
                    self.get_logger().info(f"Object detected: local=({x_3d:.3f}, {y_3d:.3f}, {z_3d:.3f}), "
                                           f"global=N/A, conf={confidence:.2f}")
            else:
                # Publish markers in local frame
                self.publish_detection_marker(
                    x_3d, y_3d, z_3d,
                    box_width_3d, box_height_3d, box_depth_3d,
                    self.front_camera_frame, color_msg.header.stamp,
                    is_human=False, confidence=confidence, label=self.object_prompt
                )
                self.get_logger().info(f"Object detected: local=({x_3d:.3f}, {y_3d:.3f}, {z_3d:.3f}), "
                                       f"global=N/A (no drone pose), conf={confidence:.2f}")
            
        except Exception as e:
            self.get_logger().error(f"Error processing front camera: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def down_camera_callback(self, color_msg, depth_msg):
        """Process down camera: detect human + estimate pose orientation."""
        if self.down_intrinsics is None:
            return
        
        try:
            # Convert ROS images to OpenCV
            color_image = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='16UC1')
            
            # Detect human with Grounding DINO
            boxes, confidences = self.detect_objects(color_image, self.human_prompt)
            
            if len(boxes) == 0:
                return
            
            # Use the highest confidence detection
            best_idx = np.argmax(confidences)
            box = boxes[best_idx]
            confidence = confidences[best_idx]
            
            # Get bounding box center (clamp to image bounds)
            h, w = color_image.shape[:2]
            depth_h, depth_w = depth_image.shape[:2]
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Clamp to depth image bounds (may differ from color image size)
            center_x = max(0, min(center_x, depth_w - 1))
            center_y = max(0, min(center_y, depth_h - 1))
            
            # Get depth at center
            depth_value = depth_image[center_y, center_x]
            if depth_value == 0:
                return
            
            # Convert to 3D position (in camera optical frame)
            depth_m = depth_value * 0.001  # mm to meters
            x_3d, y_3d, z_3d = self.pixel_to_3d(
                center_x, center_y, depth_m, self.down_intrinsics
            )
            
            # Estimate human orientation with MediaPipe
            orientation_angle = self.estimate_human_orientation(color_image)
            
            # Publish local human pose
            self.publish_human_pose(x_3d, y_3d, z_3d, orientation_angle, color_msg.header.stamp)
            
            # Compute 3D bounding box size from 2D box and depth
            box_width_3d = abs((x2 - x1) * depth_m / self.down_intrinsics['fx'])
            box_height_3d = abs((y2 - y1) * depth_m / self.down_intrinsics['fy'])
            box_depth_3d = 0.3  # Assume ~30cm depth for humans
            
            # Transform to global frame and publish if drone pose is available
            if self.drone_pose is not None:
                global_pos = self.transform_to_global_down_camera(x_3d, y_3d, z_3d)
                if global_pos is not None:
                    self.publish_global_human_pose(global_pos[0], global_pos[1], global_pos[2], 
                                                    orientation_angle, color_msg.header.stamp)
                    
                    # Publish visualization markers in global frame
                    self.publish_detection_marker(
                        global_pos[0], global_pos[1], global_pos[2],
                        box_width_3d, box_height_3d, box_depth_3d,
                        self.global_frame, color_msg.header.stamp,
                        is_human=True, confidence=confidence, label=self.human_prompt
                    )
                    
                    self.get_logger().info(f"Human detected: local=({x_3d:.3f}, {y_3d:.3f}, {z_3d:.3f}), "
                                           f"global=({global_pos[0]:.3f}, {global_pos[1]:.3f}, {global_pos[2]:.3f}), "
                                           f"yaw={orientation_angle:.1f}°, conf={confidence:.2f}")
                else:
                    # Publish markers in local frame as fallback
                    self.publish_detection_marker(
                        x_3d, y_3d, z_3d,
                        box_width_3d, box_height_3d, box_depth_3d,
                        self.down_camera_frame, color_msg.header.stamp,
                        is_human=True, confidence=confidence, label=self.human_prompt
                    )
                    self.get_logger().info(f"Human detected: local=({x_3d:.3f}, {y_3d:.3f}, {z_3d:.3f}), "
                                           f"global=N/A, yaw={orientation_angle:.1f}°, conf={confidence:.2f}")
            else:
                # Publish markers in local frame
                self.publish_detection_marker(
                    x_3d, y_3d, z_3d,
                    box_width_3d, box_height_3d, box_depth_3d,
                    self.down_camera_frame, color_msg.header.stamp,
                    is_human=True, confidence=confidence, label=self.human_prompt
                )
                self.get_logger().info(f"Human detected: local=({x_3d:.3f}, {y_3d:.3f}, {z_3d:.3f}), "
                                       f"global=N/A (no drone pose), yaw={orientation_angle:.1f}°, conf={confidence:.2f}")
            
        except Exception as e:
            self.get_logger().error(f"Error processing down camera: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def detect_objects(self, image, text_prompt):
        """Detect objects using Grounding DINO."""
        # Preprocess image
        image_pil = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_transformed, _ = transform(image_pil, None)
        
        # Run detection
        with torch.no_grad():
            boxes, logits, phrases = predict(
                model=self.gdino_model,
                image=image_transformed,
                caption=text_prompt,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                device=self.device
            )
        
        # Convert to numpy
        boxes_np = boxes.cpu().numpy() if len(boxes) > 0 else np.array([])
        confidences_np = logits.cpu().numpy() if len(logits) > 0 else np.array([])
        
        return boxes_np, confidences_np
    
    def estimate_human_orientation(self, image):
        """Estimate human orientation angle using MediaPipe Pose."""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # Process with MediaPipe
        results = self.pose.process(image_rgb)
        
        if not results.pose_landmarks:
            return 0.0  # Default angle
        
        # Calculate orientation angle
        landmarks = results.pose_landmarks.landmark
        angle = self.calculate_yaw_angle(landmarks)
        
        return angle
    
    def calculate_yaw_angle(self, landmarks):
        """Calculate yaw angle from pose landmarks."""
        # Get key landmarks
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        # Calculate shoulder and hip midpoints
        shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_mid_z = (left_shoulder.z + right_shoulder.z) / 2
        hip_mid_x = (left_hip.x + right_hip.x) / 2
        hip_mid_z = (left_hip.z + right_hip.z) / 2
        
        # Torso vector
        torso_vector = np.array([
            hip_mid_x - shoulder_mid_x,
            hip_mid_z - shoulder_mid_z
        ])
        
        # Shoulder vector (left to right)
        shoulder_vector = np.array([
            right_shoulder.x - left_shoulder.x,
            right_shoulder.z - left_shoulder.z
        ])
        
        # Calculate normal vector (cross product in 2D gives scalar)
        # This represents the body facing direction
        normal_x = shoulder_vector[0]
        normal_z = shoulder_vector[1]
        
        # Calculate angle relative to camera (Z-axis pointing away)
        angle_rad = np.arctan2(normal_x, -normal_z)
        angle_deg = np.degrees(angle_rad)
        
        # Normalize to 0-360
        if angle_deg < 0:
            angle_deg += 360
        
        return angle_deg
    
    def pixel_to_3d(self, x_pixel, y_pixel, depth_m, intrinsics):
        """Convert pixel coordinates + depth to 3D coordinates."""
        x = (x_pixel - intrinsics['cx']) * depth_m / intrinsics['fx']
        y = (y_pixel - intrinsics['cy']) * depth_m / intrinsics['fy']
        z = depth_m
        return x, y, z
    
    def publish_human_pose(self, x, y, z, yaw_deg, timestamp):
        """Publish human pose with orientation."""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = timestamp
        pose_msg.header.frame_id = self.front_camera_frame
        
        # Position
        pose_msg.pose.position.x = float(x)
        pose_msg.pose.position.y = float(y)
        pose_msg.pose.position.z = float(z)
        
        # Orientation (yaw angle as quaternion)
        yaw_rad = np.radians(yaw_deg)
        q = quaternion_from_euler(0, 0, yaw_rad)  # roll, pitch, yaw
        pose_msg.pose.orientation.x = float(q[0])
        pose_msg.pose.orientation.y = float(q[1])
        pose_msg.pose.orientation.z = float(q[2])
        pose_msg.pose.orientation.w = float(q[3])
        
        self.human_pose_pub.publish(pose_msg)
    
    def publish_object_pose(self, x, y, z, timestamp):
        """Publish object pose (no orientation)."""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = timestamp
        pose_msg.header.frame_id = self.down_camera_frame
        
        # Position
        pose_msg.pose.position.x = float(x)
        pose_msg.pose.position.y = float(y)
        pose_msg.pose.position.z = float(z)
        
        # Identity orientation (no rotation)
        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = 0.0
        pose_msg.pose.orientation.w = 1.0
        
        self.object_pose_pub.publish(pose_msg)
    
    def transform_to_global_down_camera(self, x_cam, y_cam, z_cam):
        """
        Transform human position from down camera optical frame to global/world frame.
        
        NOTE: The "down" camera is now facing FORWARD with 0° pitch (horizontal).
        
        Coordinate frames:
        - Camera optical frame (ROS standard): X-right, Y-down, Z-forward (into scene)
        - Camera is horizontal (0° pitch) - looking straight ahead
        - Mavros/Vicon world frame (ENU): X-East, Y-North, Z-Up
        - Drone body frame (FLU for mavros): X-Forward, Y-Left, Z-Up
        
        For a camera facing straight FORWARD (0° pitch):
        - Camera Z (depth/forward) -> Body +X (forward)
        - Camera X (right in image) -> Body -Y (right is negative Y in FLU where Y is left)
        - Camera Y (down in image) -> Body -Z (down is negative Z in FLU where Z is up)
        """
        if self.drone_pose is None:
            return None
        
        try:
            # Step 1: Transform from camera optical frame to drone body frame (FLU)
            # Camera optical: X-right, Y-down, Z-forward
            # Camera is horizontal (no pitch to undo)
            # Drone body (FLU): X-forward, Y-left, Z-up
            
            # For forward-facing camera (0° pitch):
            # - cam_z (depth/forward) -> +body_x (forward)
            # - cam_x (right in image) -> -body_y (right = negative Y in FLU)
            # - cam_y (down in image) -> -body_z (down = negative Z in FLU)
            
            x_body = z_cam    # Camera Z (forward) -> Body X (forward)
            y_body = -x_cam   # Camera X (right) -> Body -Y (right in FLU is -Y)
            z_body = -y_cam   # Camera Y (down) -> Body -Z (down in FLU is -Z)
            
            # Add camera offset in body frame
            # Camera is 9cm forward (+X), centered on Y, and 8cm above (+Z in FLU)
            x_body += self.down_camera_offset[0]  # +0.09m forward
            y_body += self.down_camera_offset[1]  # 0 (centered)
            z_body += self.down_camera_offset[2]  # +0.08m (above = positive Z in FLU)
            
            pos_body = np.array([x_body, y_body, z_body])
            
            # Step 2: Transform from drone body frame to world frame (ENU)
            drone_pos = np.array([
                self.drone_pose.pose.position.x,
                self.drone_pose.pose.position.y,
                self.drone_pose.pose.position.z
            ])
            
            drone_quat = [
                self.drone_pose.pose.orientation.x,
                self.drone_pose.pose.orientation.y,
                self.drone_pose.pose.orientation.z,
                self.drone_pose.pose.orientation.w
            ]
            
            # Get rotation matrix from quaternion
            R = quaternion_matrix(drone_quat)[:3, :3]
            
            # Transform position to world frame
            pos_world = R @ pos_body + drone_pos
            
            return pos_world
            
        except Exception as e:
            self.get_logger().error(f"Error transforming down camera to global frame: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            return None
    
    def publish_global_object_pose(self, x, y, z, timestamp):
        """Publish object pose in global/world frame."""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = timestamp
        pose_msg.header.frame_id = self.global_frame
        
        # Position in world frame
        pose_msg.pose.position.x = float(x)
        pose_msg.pose.position.y = float(y)
        pose_msg.pose.position.z = float(z)
        
        # Identity orientation (no rotation - we don't estimate object orientation)
        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = 0.0
        pose_msg.pose.orientation.w = 1.0
        
        self.global_object_pose_pub.publish(pose_msg)
    
    def transform_to_global_front_camera(self, x_cam, y_cam, z_cam):
        """
        Transform human position from front camera optical frame to global/world frame.
        
        Coordinate frames:
        - Camera optical frame (ROS standard): X-right, Y-down, Z-forward (into scene)
        - Camera is pitched down ~30 degrees from horizontal
        - Mavros/Vicon world frame (ENU): X-East, Y-North, Z-Up
        - Drone body frame (FLU for mavros): X-Forward, Y-Left, Z-Up
        """
        if self.drone_pose is None:
            return None
        
        try:
            # Step 1: Transform from camera optical frame to drone body frame (FLU)
            # Camera optical: X-right, Y-down, Z-forward
            # Camera is pitched down by front_camera_pitch (e.g., -30 degrees)
            
            # First undo the camera pitch rotation around X-axis
            pitch = self.front_camera_pitch  # negative value (e.g., -30 degrees in radians)
            cos_p = np.cos(-pitch)  # undo the pitch
            sin_p = np.sin(-pitch)
            
            # Rotation around X-axis to undo pitch
            x_rot = x_cam
            y_rot = cos_p * y_cam - sin_p * z_cam
            z_rot = sin_p * y_cam + cos_p * z_cam
            
            # Now map from camera-aligned frame to drone body frame (FLU)
            # After undoing pitch, camera frame is aligned with drone but axes differ:
            # Camera X (right) -> Body -Y (right is -Y in FLU where Y is left)
            # Camera Y (down) -> Body -Z (down is -Z in FLU where Z is up)
            # Camera Z (forward) -> Body +X (forward)
            
            x_body = z_rot    # Camera Z (forward) -> Body X (forward)
            y_body = -x_rot   # Camera X (right) -> Body -Y (right in FLU is -Y)
            z_body = -y_rot   # Camera Y (down) -> Body -Z (down in FLU is -Z)
            
            # Add camera offset (camera is 12cm forward, centered on Y and Z)
            x_body += self.front_camera_offset[0]  # +0.12m forward
            y_body += self.front_camera_offset[1]  # 0 (centered)
            z_body += self.front_camera_offset[2]  # 0 (centered)
            
            pos_body = np.array([x_body, y_body, z_body])
            
            # Step 2: Transform from drone body frame to world frame (ENU)
            drone_pos = np.array([
                self.drone_pose.pose.position.x,
                self.drone_pose.pose.position.y,
                self.drone_pose.pose.position.z
            ])
            
            drone_quat = [
                self.drone_pose.pose.orientation.x,
                self.drone_pose.pose.orientation.y,
                self.drone_pose.pose.orientation.z,
                self.drone_pose.pose.orientation.w
            ]
            
            # Get rotation matrix from quaternion
            R = quaternion_matrix(drone_quat)[:3, :3]
            
            # Transform position to world frame
            pos_world = R @ pos_body + drone_pos
            
            return pos_world
            
        except Exception as e:
            self.get_logger().error(f"Error transforming front camera to global frame: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            return None
    
    def publish_global_human_pose(self, x, y, z, yaw_deg, timestamp):
        """Publish human pose in global/world frame with orientation."""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = timestamp
        pose_msg.header.frame_id = self.global_frame
        
        # Position in world frame
        pose_msg.pose.position.x = float(x)
        pose_msg.pose.position.y = float(y)
        pose_msg.pose.position.z = float(z)
        
        # Orientation (yaw angle as quaternion) - this is the human's facing direction
        # Note: This yaw is relative to the camera, we should transform it to world frame too
        # For now, we'll add the drone's yaw to get approximate world-frame orientation
        if self.drone_pose is not None:
            # Extract drone yaw from quaternion
            drone_quat = self.drone_pose.pose.orientation
            # Yaw from quaternion: atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
            siny_cosp = 2.0 * (drone_quat.w * drone_quat.z + drone_quat.x * drone_quat.y)
            cosy_cosp = 1.0 - 2.0 * (drone_quat.y * drone_quat.y + drone_quat.z * drone_quat.z)
            drone_yaw_rad = np.arctan2(siny_cosp, cosy_cosp)
            
            # Human yaw in world frame = human yaw relative to camera + drone yaw
            yaw_rad = np.radians(yaw_deg) + drone_yaw_rad
        else:
            yaw_rad = np.radians(yaw_deg)
        
        q = quaternion_from_euler(0, 0, yaw_rad)  # roll, pitch, yaw
        pose_msg.pose.orientation.x = float(q[0])
        pose_msg.pose.orientation.y = float(q[1])
        pose_msg.pose.orientation.z = float(q[2])
        pose_msg.pose.orientation.w = float(q[3])
        
        self.global_human_pose_pub.publish(pose_msg)
    
    def publish_detection_marker(self, x, y, z, width, height, depth, frame_id, timestamp, 
                                  is_human=True, confidence=0.0, label=""):
        """
        Publish visualization markers for RViz showing detection bounding box and label.
        
        Creates:
        - A wireframe cube marker for the bounding box
        - A text marker for the label and confidence
        - A sphere marker at the center point
        """
        marker_array = MarkerArray()
        
        # Colors: Green for human, Blue for object
        if is_human:
            color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8)  # Green
            text_color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
        else:
            color = ColorRGBA(r=0.0, g=0.5, b=1.0, a=0.8)  # Blue
            text_color = ColorRGBA(r=0.0, g=0.5, b=1.0, a=1.0)
        
        # Marker 1: Bounding box (CUBE with wireframe style using LINE_LIST)
        box_marker = Marker()
        box_marker.header.frame_id = frame_id
        box_marker.header.stamp = timestamp
        box_marker.ns = "human_bbox" if is_human else "object_bbox"
        box_marker.id = self.marker_id
        box_marker.type = Marker.CUBE
        box_marker.action = Marker.ADD
        
        box_marker.pose.position.x = float(x)
        box_marker.pose.position.y = float(y)
        box_marker.pose.position.z = float(z)
        box_marker.pose.orientation.w = 1.0
        
        box_marker.scale.x = float(width)
        box_marker.scale.y = float(height)
        box_marker.scale.z = float(depth)
        
        # Semi-transparent fill
        box_marker.color = ColorRGBA(r=color.r, g=color.g, b=color.b, a=0.3)
        box_marker.lifetime = Duration(seconds=0.5).to_msg()
        
        marker_array.markers.append(box_marker)
        self.marker_id += 1
        
        # Marker 2: Wireframe edges (LINE_LIST)
        edge_marker = Marker()
        edge_marker.header.frame_id = frame_id
        edge_marker.header.stamp = timestamp
        edge_marker.ns = "human_edges" if is_human else "object_edges"
        edge_marker.id = self.marker_id
        edge_marker.type = Marker.LINE_LIST
        edge_marker.action = Marker.ADD
        
        edge_marker.pose.orientation.w = 1.0
        edge_marker.scale.x = 0.01  # Line width
        edge_marker.color = color
        edge_marker.lifetime = Duration(seconds=0.5).to_msg()
        
        # Create 12 edges of the bounding box
        hw, hh, hd = width/2, height/2, depth/2
        corners = [
            (x-hw, y-hh, z-hd), (x+hw, y-hh, z-hd),
            (x+hw, y+hh, z-hd), (x-hw, y+hh, z-hd),
            (x-hw, y-hh, z+hd), (x+hw, y-hh, z+hd),
            (x+hw, y+hh, z+hd), (x-hw, y+hh, z+hd)
        ]
        
        # Define edges as pairs of corner indices
        edges = [
            (0,1), (1,2), (2,3), (3,0),  # Bottom face
            (4,5), (5,6), (6,7), (7,4),  # Top face
            (0,4), (1,5), (2,6), (3,7)   # Vertical edges
        ]
        
        for i, j in edges:
            p1 = Point(x=corners[i][0], y=corners[i][1], z=corners[i][2])
            p2 = Point(x=corners[j][0], y=corners[j][1], z=corners[j][2])
            edge_marker.points.append(p1)
            edge_marker.points.append(p2)
        
        marker_array.markers.append(edge_marker)
        self.marker_id += 1
        
        # Marker 3: Center point (SPHERE)
        center_marker = Marker()
        center_marker.header.frame_id = frame_id
        center_marker.header.stamp = timestamp
        center_marker.ns = "human_center" if is_human else "object_center"
        center_marker.id = self.marker_id
        center_marker.type = Marker.SPHERE
        center_marker.action = Marker.ADD
        
        center_marker.pose.position.x = float(x)
        center_marker.pose.position.y = float(y)
        center_marker.pose.position.z = float(z)
        center_marker.pose.orientation.w = 1.0
        
        center_marker.scale.x = 0.05
        center_marker.scale.y = 0.05
        center_marker.scale.z = 0.05
        
        center_marker.color = color
        center_marker.lifetime = Duration(seconds=0.5).to_msg()
        
        marker_array.markers.append(center_marker)
        self.marker_id += 1
        
        # Marker 4: Text label
        text_marker = Marker()
        text_marker.header.frame_id = frame_id
        text_marker.header.stamp = timestamp
        text_marker.ns = "human_label" if is_human else "object_label"
        text_marker.id = self.marker_id
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        
        text_marker.pose.position.x = float(x)
        text_marker.pose.position.y = float(y)
        text_marker.pose.position.z = float(z + height/2 + 0.1)  # Above the box
        text_marker.pose.orientation.w = 1.0
        
        text_marker.scale.z = 0.1  # Text height
        text_marker.color = text_color
        text_marker.text = f"{label} ({confidence:.0%})"
        text_marker.lifetime = Duration(seconds=0.5).to_msg()
        
        marker_array.markers.append(text_marker)
        self.marker_id += 1
        
        # Publish to appropriate topic
        if is_human:
            self.human_marker_pub.publish(marker_array)
        else:
            self.object_marker_pub.publish(marker_array)
        
        # Reset marker ID periodically to avoid overflow
        if self.marker_id > 10000:
            self.marker_id = 0


def main(args=None):
    rclpy.init(args=args)
    node = DualCameraDetectionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down dual camera detection node...")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
