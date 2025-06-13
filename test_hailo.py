import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
from collections import deque
import threading
import time
from datetime import datetime

from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.pose_estimation_pipeline import GStreamerPoseEstimationApp

# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.frame_buffer = deque(maxlen=900)  # 30 seconds at 30fps
        self.arms_crossed_detected = False
        self.last_arms_crossed_time = None
        self.saving_video = False
        self.video_writer = None
        self.frame_dimensions = None
        self.fps = 30  # Assumed FPS, adjust as needed
        self.cooldown_period = 5  # Seconds before allowing another video save
        self.status_text = "Arms Not Crossed"
        self.lock = threading.Lock()

    def add_frame_to_buffer(self, frame):
        """Add frame to circular buffer"""
        with self.lock:
            self.frame_buffer.append(frame.copy())

    def save_buffer_to_video(self):
        """Save the frame buffer to a video file"""
        if self.saving_video:
            return
        
        with self.lock:
            if len(self.frame_buffer) == 0:
                print("No frames in buffer to save")
                return
            
            self.saving_video = True
            frames_to_save = list(self.frame_buffer)
        
        # Create video filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"arms_crossed_{timestamp}.mp4"
        
        # Get frame dimensions from the first frame
        if len(frames_to_save) > 0:
            h, w = frames_to_save[0].shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filename, fourcc, self.fps, (w, h))
            
            # Write all frames
            for frame in frames_to_save:
                out.write(frame)
            
            out.release()
            print(f"Video saved: {filename} ({len(frames_to_save)} frames)")
        
        self.saving_video = False

# -----------------------------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------------------------

def get_keypoints():
    """Get the COCO keypoints and their left/right flip coorespondence map."""
    keypoints = {
        'nose': 0,
        'left_eye': 1,
        'right_eye': 2,
        'left_ear': 3,
        'right_ear': 4,
        'left_shoulder': 5,
        'right_shoulder': 6,
        'left_elbow': 7,
        'right_elbow': 8,
        'left_wrist': 9,
        'right_wrist': 10,
        'left_hip': 11,
        'right_hip': 12,
        'left_knee': 13,
        'right_knee': 14,
        'left_ankle': 15,
        'right_ankle': 16,
    }
    return keypoints

def check_arms_crossed_above_head(points, bbox, width, height, keypoints):
    """
    Check if arms are crossed above head
    Returns True if both wrists are above the head and crossed
    """
    try:
        # Get keypoint indices
        left_wrist_idx = keypoints['left_wrist']
        right_wrist_idx = keypoints['right_wrist']
        nose_idx = keypoints['nose']
        
        # Get actual coordinates
        left_wrist = points[left_wrist_idx]
        right_wrist = points[right_wrist_idx]
        nose = points[nose_idx]
        
        # Convert to absolute coordinates
        left_wrist_x = (left_wrist.x() * bbox.width() + bbox.xmin()) * width
        left_wrist_y = (left_wrist.y() * bbox.height() + bbox.ymin()) * height
        
        right_wrist_x = (right_wrist.x() * bbox.width() + bbox.xmin()) * width
        right_wrist_y = (right_wrist.y() * bbox.height() + bbox.ymin()) * height
        
        nose_y = (nose.y() * bbox.height() + bbox.ymin()) * height
        
        # Check if both wrists are above the nose (head level)
        wrists_above_head = (left_wrist_y < nose_y) and (right_wrist_y < nose_y)
        
        # Check if wrists are crossed (left wrist to the right of right wrist)
        wrists_crossed = left_wrist_x > right_wrist_x
        
        return wrists_above_head and wrists_crossed
        
    except Exception as e:
        print(f"Error checking arms crossed: {e}")
        return False

# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------

def app_callback(pad, info, user_data):
    # Get the GstBuffer from the probe info
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Using the user_data to count the number of frames
    user_data.increment()
    string_to_print = f"Frame count: {user_data.get_count()}\n"

    # Get the caps from the pad
    format, width, height = get_caps_from_pad(pad)

    # Get video frame
    frame = None
    if format is not None and width is not None and height is not None:
        # Always get frame for buffer and display
        user_data.use_frame = True
        frame = get_numpy_from_buffer(buffer, format, width, height)
        
        # Store frame dimensions
        if user_data.frame_dimensions is None:
            user_data.frame_dimensions = (height, width)

    # Get the detections from the buffer
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # Get the keypoints
    keypoints = get_keypoints()

    # Track if arms are crossed in this frame
    arms_crossed_in_frame = False

    # Parse the detections
    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()
        
        if label == "person":
            # Get track ID
            track_id = 0
            track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            if len(track) == 1:
                track_id = track[0].get_id()
            string_to_print += (f"Detection: ID: {track_id} Label: {label} Confidence: {confidence:.2f}\n")

            # Pose estimation landmarks from detection
            landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
            if len(landmarks) != 0:
                points = landmarks[0].get_points()
                
                # Check if arms are crossed above head
                if check_arms_crossed_above_head(points, bbox, width, height, keypoints):
                    arms_crossed_in_frame = True
                
                # Draw keypoints
                if frame is not None:
                    # Draw all keypoints
                    for keypoint_name, keypoint_index in keypoints.items():
                        if keypoint_index < len(points):
                            point = points[keypoint_index]
                            x = int((point.x() * bbox.width() + bbox.xmin()) * width)
                            y = int((point.y() * bbox.height() + bbox.ymin()) * height)
                            
                            # Different colors for different keypoints
                            if 'wrist' in keypoint_name:
                                color = (0, 255, 255)  # Yellow for wrists
                            elif keypoint_name == 'nose':
                                color = (255, 0, 0)  # Blue for nose
                            else:
                                color = (0, 255, 0)  # Green for others
                            
                            cv2.circle(frame, (x, y), 5, color, -1)

    # Update status and handle video saving
    current_time = time.time()
    
    if arms_crossed_in_frame:
        user_data.status_text = "Arms Crossed Above Head!"
        
        # Check if we should save video
        if not user_data.arms_crossed_detected:
            user_data.arms_crossed_detected = True
            
            # Check cooldown period
            if (user_data.last_arms_crossed_time is None or 
                current_time - user_data.last_arms_crossed_time > user_data.cooldown_period):
                
                user_data.last_arms_crossed_time = current_time
                # Save video in a separate thread to avoid blocking
                threading.Thread(target=user_data.save_buffer_to_video).start()
    else:
        user_data.status_text = "Arms Not Crossed"
        user_data.arms_crossed_detected = False

    if frame is not None:
        # Convert the frame to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Add status text to frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        
        # Add background rectangle for better text visibility
        text_size = cv2.getTextSize(user_data.status_text, font, font_scale, thickness)[0]
        text_x = 10
        text_y = 40
        
        # Draw background rectangle
        cv2.rectangle(frame, 
                     (text_x - 5, text_y - text_size[1] - 5),
                     (text_x + text_size[0] + 5, text_y + 5),
                     (0, 0, 0), -1)
        
        # Draw text
        color = (0, 0, 255) if "Crossed" in user_data.status_text and "Not" not in user_data.status_text else (0, 255, 0)
        cv2.putText(frame, user_data.status_text, (text_x, text_y), 
                   font, font_scale, color, thickness, cv2.LINE_AA)
        
        # Add buffer status
        buffer_text = f"Buffer: {len(user_data.frame_buffer)}/{user_data.frame_buffer.maxlen} frames"
        cv2.putText(frame, buffer_text, (text_x, text_y + 40), 
                   font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Add frame to buffer
        user_data.add_frame_to_buffer(frame)
        
        # Set frame for display
        user_data.set_frame(frame)

    print(string_to_print)
    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    # Create an instance of the user app callback class
    user_data = user_app_callback_class()
    app = GStreamerPoseEstimationApp(app_callback, user_data)
    app.run()