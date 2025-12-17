#!/usr/bin/env python3
"""
ROS2 Node: Detection and Distance Measurement
Dual-camera object detection with depth measurement.

Phase 1: Grounding DINO only (no MediaPipe pose estimation)

Subscribes to (Jetson Nano topics):
    - /camera/camera_down/color/image_raw (sensor_msgs/Image)
    - /camera/camera_down/depth/image_rect_raw (sensor_msgs/Image)
    - /camera/camera_down/color/camera_info (sensor_msgs/CameraInfo)
    - /camera/camera_front/color/image_raw (sensor_msgs/Image)
    - /camera/camera_front/depth/image_rect_raw (sensor_msgs/Image)
    - /camera/camera_front/color/camera_info (sensor_msgs/CameraInfo)

Publishes:
    - /object_position (geometry_msgs/PoseStamped) - Object position from camera_down
    - /human_position (geometry_msgs/PoseStamped) - Human position from camera_front
    - /detection_markers (visualization_msgs/MarkerArray) - RViz visualization
    - /camera/camera_down/detection_image (sensor_msgs/Image) - Annotated image
    - /camera/camera_front/detection_image (sensor_msgs/Image) - Annotated image
"""

import sys
import os

# Force Matplotlib to use non-interactive backend
import matplotlib
matplotlib.use('Agg')

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.duration import Duration
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from typing import Tuple, Optional
import message_filters
import threading

# ROS2 messages
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion, Vector3
from std_msgs.msg import Header, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration as DurationMsg

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


class DetectionAndDistanceNode(Node):
    """ROS2 node for dual-camera detection and distance measurement."""
    
    def __init__(self):
        """Initialize the ROS2 node and all components."""
        super().__init__('detection_and_distance_node')
        
        self.get_logger().info("=" * 80)
        self.get_logger().info("Detection and Distance Measurement Node (ROS2)")
        self.get_logger().info("Phase 1: Grounding DINO Only (No Pose Estimation)")
        self.get_logger().info("=" * 80)
        
        # Declare and get parameters
        self.declare_parameter('object_prompt', 'Grey LiPo battery')
        self.declare_parameter('human_prompt', 'human')
        self.declare_parameter('box_threshold', 0.30)
        self.declare_parameter('text_threshold', 0.20)
        self.declare_parameter('drone_frame', 'drone')
        
        self.object_prompt = self.get_parameter('object_prompt').get_parameter_value().string_value
        self.human_prompt = self.get_parameter('human_prompt').get_parameter_value().string_value
        self.box_threshold = self.get_parameter('box_threshold').get_parameter_value().double_value
        self.text_threshold = self.get_parameter('text_threshold').get_parameter_value().double_value
        self.drone_frame = self.get_parameter('drone_frame').get_parameter_value().string_value
        
        self.get_logger().info(f"Object detection prompt: '{self.object_prompt}'")
        self.get_logger().info(f"Human detection prompt: '{self.human_prompt}'")
        self.get_logger().info(f"Detection thresholds: box={self.box_threshold}, text={self.text_threshold}")
        self.get_logger().info(f"Frame ID: {self.drone_frame}")
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Camera intrinsics and frame IDs (will be updated from camera_info)
        self.camera_down_intrinsics = None
        self.camera_down_frame_id = None
        self.camera_front_intrinsics = None
        self.camera_front_frame_id = None
        
        # Lock for thread-safe detection (callbacks can be concurrent)
        self.detection_lock = threading.Lock()
        
        # Initialize Grounding DINO
        self.get_logger().info("Loading Grounding DINO model...")
        # Force CPU mode to avoid CUDA threading issues with ROS callbacks
        self.device = "cpu"
        self.get_logger().info(f"Using device: {self.device} (forced for ROS compatibility)")
        
        gdino_dir = gdino_path
        self.get_logger().info(f"Found GroundingDINO at: {gdino_dir}")
        
        gdino_config = os.path.join(gdino_dir, "groundingdino/config/GroundingDINO_SwinT_OGC.py")
        gdino_checkpoint = os.path.join(gdino_dir, "weights/groundingdino_swint_ogc.pth")
        
        if not os.path.exists(gdino_config):
            self.get_logger().error(f"Config not found: {gdino_config}")
            sys.exit(1)
        if not os.path.exists(gdino_checkpoint):
            self.get_logger().error(f"Checkpoint not found: {gdino_checkpoint}")
            sys.exit(1)
        
        try:
            self.gdino_model = load_model(gdino_config, gdino_checkpoint, device=self.device)
            self.get_logger().info("✓ Grounding DINO loaded")
        except Exception as e:
            self.get_logger().error(f"Failed to load Grounding DINO: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        # QoS profile for sensor data
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Publishers
        self.object_pos_pub = self.create_publisher(PoseStamped, '/object_position', 10)
        self.human_pos_pub = self.create_publisher(PoseStamped, '/human_position', 10)
        
        # Visualization publishers
        self.marker_pub = self.create_publisher(MarkerArray, '/detection_markers', 10)
        self.image_down_pub = self.create_publisher(Image, '/camera/camera_down/detection_image', 1)
        self.image_front_pub = self.create_publisher(Image, '/camera/camera_front/detection_image', 1)
        
        # Marker ID counter
        self.marker_id = 0
        
        # Subscribers for camera_down (object detection) - using Jetson topic names
        self.camera_down_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera_down/color/camera_info',
            self.camera_down_info_callback,
            sensor_qos
        )
        
        # Synchronized subscribers for camera_down using message_filters
        self.camera_down_color_sub = message_filters.Subscriber(
            self,
            Image,
            '/camera/camera_down/color/image_raw',
            qos_profile=sensor_qos
        )
        self.camera_down_depth_sub = message_filters.Subscriber(
            self,
            Image,
            '/camera/camera_down/depth/image_rect_raw',
            qos_profile=sensor_qos
        )
        
        self.camera_down_sync = message_filters.ApproximateTimeSynchronizer(
            [self.camera_down_color_sub, self.camera_down_depth_sub],
            queue_size=10,
            slop=0.1
        )
        self.camera_down_sync.registerCallback(self.camera_down_callback)
        
        # Subscribers for camera_front (human detection) - using Jetson topic names
        self.camera_front_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera_front/color/camera_info',
            self.camera_front_info_callback,
            sensor_qos
        )
        
        # Synchronized subscribers for camera_front
        self.camera_front_color_sub = message_filters.Subscriber(
            self,
            Image,
            '/camera/camera_front/color/image_raw',
            qos_profile=sensor_qos
        )
        self.camera_front_depth_sub = message_filters.Subscriber(
            self,
            Image,
            '/camera/camera_front/depth/image_rect_raw',
            qos_profile=sensor_qos
        )
        
        self.camera_front_sync = message_filters.ApproximateTimeSynchronizer(
            [self.camera_front_color_sub, self.camera_front_depth_sub],
            queue_size=10,
            slop=0.1
        )
        self.camera_front_sync.registerCallback(self.camera_front_callback)
        
        # Throttle logging
        self.last_down_log_time = self.get_clock().now()
        self.last_front_log_time = self.get_clock().now()
        self.log_interval = Duration(seconds=2)
        
        self.get_logger().info("=" * 80)
        self.get_logger().info("Node initialized successfully. Waiting for camera data...")
        self.get_logger().info(f"  - camera_down: detecting '{self.object_prompt}'")
        self.get_logger().info(f"  - camera_front: detecting '{self.human_prompt}'")
        self.get_logger().info("Subscribed topics (Jetson Nano):")
        self.get_logger().info("  - /camera/camera_down/color/image_raw")
        self.get_logger().info("  - /camera/camera_down/depth/image_rect_raw")
        self.get_logger().info("  - /camera/camera_front/color/image_raw")
        self.get_logger().info("  - /camera/camera_front/depth/image_rect_raw")
        self.get_logger().info("=" * 80)
    
    def camera_down_info_callback(self, msg):
        """Store camera intrinsics and frame ID for downward camera."""
        if self.camera_down_intrinsics is None:
            self.camera_down_intrinsics = {
                'fx': msg.k[0],
                'fy': msg.k[4],
                'cx': msg.k[2],
                'cy': msg.k[5],
                'width': msg.width,
                'height': msg.height
            }
            self.camera_down_frame_id = msg.header.frame_id
            self.get_logger().info(f"Camera Down intrinsics received: fx={msg.k[0]:.2f}, fy={msg.k[4]:.2f}")
            self.get_logger().info(f"Camera Down frame_id: {self.camera_down_frame_id}")
    
    def camera_front_info_callback(self, msg):
        """Store camera intrinsics and frame ID for front camera."""
        if self.camera_front_intrinsics is None:
            self.camera_front_intrinsics = {
                'fx': msg.k[0],
                'fy': msg.k[4],
                'cx': msg.k[2],
                'cy': msg.k[5],
                'width': msg.width,
                'height': msg.height
            }
            self.camera_front_frame_id = msg.header.frame_id
            self.get_logger().info(f"Camera Front intrinsics received: fx={msg.k[0]:.2f}, fy={msg.k[4]:.2f}")
            self.get_logger().info(f"Camera Front frame_id: {self.camera_front_frame_id}")
    
    def camera_down_callback(self, color_msg, depth_msg):
        """Process camera_down images for object (mouse) detection."""
        if self.camera_down_intrinsics is None or self.camera_down_frame_id is None:
            return
        
        try:
            # Convert ROS images to OpenCV
            color_image = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='16UC1')
            
            # Convert depth to meters (RealSense depth is in mm)
            depth_meters = depth_image.astype(np.float32) / 1000.0
            
            # Detect objects
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            detections = self.detect_objects(rgb_image, self.object_prompt)
            
            # Create visualization image
            vis_image = color_image.copy()
            marker_array = MarkerArray()
            h, w = color_image.shape[:2]
            
            for det in detections:
                box = det['box']
                conf = det['confidence']
                label = det['label']
                
                # Draw 2D bounding box
                cx, cy, bw, bh = box
                x1 = int((cx - bw/2) * w)
                y1 = int((cy - bh/2) * h)
                x2 = int((cx + bw/2) * w)
                y2 = int((cy + bh/2) * h)
                
                # Green color for objects
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Get 3D position
                depth_value, px, py = self.get_depth_at_box(depth_meters, box)
                
                if depth_value > 0 and not np.isnan(depth_value):
                    X, Y, Z = self.get_xyz_from_depth(px, py, depth_value, self.camera_down_intrinsics)
                    
                    # Draw label with distance
                    label_text = f"{label}: {conf:.2f} ({Z:.2f}m)"
                    cv2.putText(vis_image, label_text, (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Create 3D marker
                    marker = self.create_marker(X, Y, Z, color_msg.header.stamp, 
                                                "object", (0.0, 1.0, 0.0), label,
                                                self.camera_down_frame_id)
                    marker_array.markers.append(marker)
            
            # Publish visualization image
            try:
                vis_msg = self.bridge.cv2_to_imgmsg(vis_image, encoding='bgr8')
                vis_msg.header = color_msg.header
                self.image_down_pub.publish(vis_msg)
            except CvBridgeError:
                pass
            
            # Publish markers
            if marker_array.markers:
                self.marker_pub.publish(marker_array)
            
            # Publish best detection position
            if detections:
                best_detection = max(detections, key=lambda x: x['confidence'])
                box = best_detection['box']
                depth_value, px, py = self.get_depth_at_box(depth_meters, box)
                
                if depth_value > 0 and not np.isnan(depth_value):
                    X, Y, Z = self.get_xyz_from_depth(px, py, depth_value, self.camera_down_intrinsics)
                    
                    pose_msg = PoseStamped()
                    pose_msg.header.stamp = color_msg.header.stamp
                    pose_msg.header.frame_id = self.camera_down_frame_id
                    pose_msg.pose.position = Point(x=X, y=Y, z=Z)
                    pose_msg.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                    self.object_pos_pub.publish(pose_msg)
                    
                    # Throttled logging
                    now = self.get_clock().now()
                    if (now - self.last_down_log_time) > self.log_interval:
                        self.get_logger().info(f"[DOWN] Object pos=({X:.2f}, {Y:.2f}, {Z:.2f})")
                        self.last_down_log_time = now
        
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge error (camera_down): {e}")
        except Exception as e:
            self.get_logger().error(f"Error processing camera_down: {e}")
    
    def camera_front_callback(self, color_msg, depth_msg):
        """Process camera_front images for human detection."""
        if self.camera_front_intrinsics is None or self.camera_front_frame_id is None:
            return
        
        try:
            # Convert ROS images to OpenCV
            color_image = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='16UC1')
            
            # Convert depth to meters
            depth_meters = depth_image.astype(np.float32) / 1000.0
            
            # Detect humans
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            detections = self.detect_objects(rgb_image, self.human_prompt)
            
            # Create visualization image
            vis_image = color_image.copy()
            marker_array = MarkerArray()
            h, w = color_image.shape[:2]
            
            for det in detections:
                box = det['box']
                conf = det['confidence']
                label = det['label']
                
                # Draw 2D bounding box
                cx, cy, bw, bh = box
                x1 = int((cx - bw/2) * w)
                y1 = int((cy - bh/2) * h)
                x2 = int((cx + bw/2) * w)
                y2 = int((cy + bh/2) * h)
                
                # Blue color for humans
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Get 3D position
                depth_value, px, py = self.get_depth_at_box(depth_meters, box)
                
                if depth_value > 0 and not np.isnan(depth_value):
                    X, Y, Z = self.get_xyz_from_depth(px, py, depth_value, self.camera_front_intrinsics)
                    
                    # Draw label with distance
                    label_text = f"{label}: {conf:.2f} ({Z:.2f}m)"
                    cv2.putText(vis_image, label_text, (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    
                    # Create 3D marker
                    marker = self.create_marker(X, Y, Z, color_msg.header.stamp, 
                                                "human", (0.0, 0.0, 1.0), label,
                                                self.camera_front_frame_id)
                    marker_array.markers.append(marker)
            
            # Publish visualization image
            try:
                vis_msg = self.bridge.cv2_to_imgmsg(vis_image, encoding='bgr8')
                vis_msg.header = color_msg.header
                self.image_front_pub.publish(vis_msg)
            except CvBridgeError:
                pass
            
            # Publish markers
            if marker_array.markers:
                self.marker_pub.publish(marker_array)
            
            # Publish best detection position
            if detections:
                best_detection = max(detections, key=lambda x: x['confidence'])
                box = best_detection['box']
                depth_value, px, py = self.get_depth_at_box(depth_meters, box)
                
                if depth_value > 0 and not np.isnan(depth_value):
                    X, Y, Z = self.get_xyz_from_depth(px, py, depth_value, self.camera_front_intrinsics)
                    
                    pose_msg = PoseStamped()
                    pose_msg.header.stamp = color_msg.header.stamp
                    pose_msg.header.frame_id = self.camera_front_frame_id
                    pose_msg.pose.position = Point(x=X, y=Y, z=Z)
                    pose_msg.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                    self.human_pos_pub.publish(pose_msg)
                    
                    # Throttled logging
                    now = self.get_clock().now()
                    if (now - self.last_front_log_time) > self.log_interval:
                        self.get_logger().info(f"[FRONT] Human pos=({X:.2f}, {Y:.2f}, {Z:.2f})")
                        self.last_front_log_time = now
        
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge error (camera_front): {e}")
        except Exception as e:
            self.get_logger().error(f"Error processing camera_front: {e}")
    
    def create_marker(self, x, y, z, stamp, ns, color, label, frame_id):
        """
        Create a 3D marker for RViz visualization.
        
        Args:
            x, y, z: 3D position
            stamp: ROS timestamp
            ns: Namespace (e.g., 'object' or 'human')
            color: RGB tuple (0-1 range)
            label: Text label
            frame_id: Frame ID for the marker
        
        Returns:
            Marker message
        """
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = ns
        marker.id = self.marker_id
        self.marker_id += 1
        
        # Use CUBE for bounding box visualization
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        
        # Position
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = float(z)
        marker.pose.orientation.w = 1.0
        
        # Size (approximate based on object type)
        if ns == "human":
            marker.scale.x = 0.5  # Width
            marker.scale.y = 0.3  # Depth
            marker.scale.z = 1.7  # Height
        else:
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.05
        
        # Color
        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.color.a = 0.7  # Semi-transparent
        
        # Lifetime (short lifetime so old markers disappear)
        marker.lifetime = DurationMsg(sec=0, nanosec=500000000)  # 0.5 seconds
        
        return marker
    
    def detect_objects(self, rgb_image, text_prompt):
        """
        Detect objects in image using Grounding DINO.
        
        Returns:
            List of detections with 'box', 'confidence', 'label'
        """
        # Use lock to prevent concurrent detection calls
        with self.detection_lock:
            try:
                image_tensor = self.preprocess_image_gdino(rgb_image)
                boxes, logits, phrases = predict(
                    model=self.gdino_model,
                    image=image_tensor,
                    caption=text_prompt,
                    box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold,
                    device=self.device
                )
                
                detections = []
                for box, logit, phrase in zip(boxes, logits, phrases):
                    box_np = box.cpu().numpy() if hasattr(box, 'cpu') else np.array(box)
                    logit_val = logit.item() if hasattr(logit, 'item') else float(logit)
                    
                    # Filter out invalid detections
                    cx, cy, w, h = box_np
                    box_area = w * h  # Normalized area (0-1)
                    
                    # Skip if box is too large (>60% of image) - likely false positive
                    if box_area > 0.6:
                        continue
                    
                    # Skip if box is too small (<0.5% of image) - likely noise
                    if box_area < 0.005:
                        continue
                    
                    # Skip if box center is outside valid range
                    if cx < 0.05 or cx > 0.95 or cy < 0.05 or cy > 0.95:
                        continue
                    
                    detections.append({
                        'box': box_np,
                        'confidence': logit_val,
                        'label': phrase
                    })
                
                return detections
            
            except Exception as e:
                self.get_logger().warning(f"Detection failed: {e}")
                return []
    
    def preprocess_image_gdino(self, image_np):
        """Preprocess image for Grounding DINO."""
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_pil = PILImage.fromarray(image_np)
        image_transformed, _ = transform(image_pil, None)
        return image_transformed
    
    def get_depth_at_box(self, depth_map, box, method='median'):
        """
        Get depth value at bounding box location.
        
        Args:
            depth_map: Depth map in meters
            box: [cx, cy, w, h] in normalized coordinates
            method: 'center', 'median', or 'mean'
        
        Returns:
            (depth_value, pixel_x, pixel_y)
        """
        h, w = depth_map.shape
        
        # Convert normalized box to pixel coordinates
        cx, cy, bw, bh = box
        cx, cy, bw, bh = cx * w, cy * h, bw * w, bh * h
        
        x1 = int(max(0, cx - bw / 2))
        y1 = int(max(0, cy - bh / 2))
        x2 = int(min(w, cx + bw / 2))
        y2 = int(min(h, cy + bh / 2))
        
        # Extract depth region
        depth_region = depth_map[y1:y2, x1:x2]
        
        if method == 'center':
            center_x = int(cx)
            center_y = int(cy)
            return depth_map[center_y, center_x], center_x, center_y
        elif method == 'median':
            valid_depths = depth_region[depth_region > 0]
            if len(valid_depths) > 0:
                return np.median(valid_depths), int(cx), int(cy)
            else:
                return depth_map[int(cy), int(cx)], int(cx), int(cy)
        elif method == 'mean':
            valid_depths = depth_region[depth_region > 0]
            if len(valid_depths) > 0:
                return np.mean(valid_depths), int(cx), int(cy)
            else:
                return depth_map[int(cy), int(cx)], int(cx), int(cy)
    
    def get_xyz_from_depth(self, x_pixel, y_pixel, depth_value, intrinsics):
        """
        Convert pixel coordinates + depth to 3D coordinates (X, Y, Z).
        
        Returns:
            (X, Y, Z) in meters from camera center
        """
        X = (x_pixel - intrinsics['cx']) * depth_value / intrinsics['fx']
        Y = (y_pixel - intrinsics['cy']) * depth_value / intrinsics['fy']
        Z = depth_value
        
        return X, Y, Z


def main(args=None):
    rclpy.init(args=args)
    node = DetectionAndDistanceNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down detection and distance node...")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
