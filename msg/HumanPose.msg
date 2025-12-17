# Human pose estimation message
# Contains pose landmarks and orientation information

std_msgs/Header header

# Pose landmarks (33 landmarks from MediaPipe)
detection_and_distance/Landmark[] landmarks

# Orientation information
string orientation_label    # e.g., "FRONT", "BACK", "LEFT", "RIGHT", "FRONT-LEFT", etc.
float32 orientation_angle   # Orientation angle in degrees (0-360)
float32 orientation_confidence  # Confidence of orientation estimation (0-1)

# Detection confidence
float32 detection_confidence  # Overall pose detection confidence (0-1)
