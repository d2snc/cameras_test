import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
import time
import threading
from collections import deque
from datetime import datetime

from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.pose_estimation_pipeline import GStreamerPoseEstimationApp

# -----------------------------------------------------------------------------------------------
# GPIO Configuration
# -----------------------------------------------------------------------------------------------
LED_AVAILABLE = False
try:
    import gpiod
    CHIP_NAME = "gpiochip4"  # On RPi 5, GPIOs are on chip 4
    LED_LINE_OFFSET = 17      # GPIO 17
    chip = gpiod.Chip(CHIP_NAME)
    led_line = chip.get_line(LED_LINE_OFFSET)
    led_line.request(
        consumer="hailo-led",
        type=gpiod.LINE_REQ_DIR_OUT,
        default_vals=[0],
    )
    LED_AVAILABLE = True
    print("GPIO setup for LED on pin 17 successful.")
except Exception as e:
    print(f"LED disabled (gpiod unavailable or setup failed): {e}")

# -----------------------------------------------------------------------------------------------
# Recording and Pose Detection Configuration
# -----------------------------------------------------------------------------------------------
BUFFER_SECONDS = 22  # 20 seconds before, 2 seconds after
RECORDING_FOLDER = "recordings"
FILE_PREFIX = "arms_crossed_"
POSE_CONFIDENCE_THRESHOLD = 0.5 # Confidence for keypoints
POSE_DURATION_SECONDS = 0.8 # How long the pose must be held to trigger

if not os.path.exists(RECORDING_FOLDER):
    os.makedirs(RECORDING_FOLDER)

# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.frame_buffer = deque()
        self.is_recording = False
        self.pose_start_time = None
        self._last_led_pulse_time = 0

    def update_buffer_size(self, fps):
        """Dynamically adjusts the buffer size based on FPS."""
        if fps > 0:
            max_len = int(fps * BUFFER_SECONDS)
            if self.frame_buffer.maxlen != max_len:
                # Recreate the deque with the new maxlen
                current_frames = list(self.frame_buffer)
                self.frame_buffer = deque(current_frames, maxlen=max_len)
                print(f"Frame buffer resized to {max_len} frames for {fps:.1f} FPS.")

    def pulse_led(self, duration=3.0):
        """Turns the LED on for a specified duration."""
        if not LED_AVAILABLE:
            return
        now = time.time()
        # Prevent rapid re-triggering
        if now - self._last_led_pulse_time < duration + 1.0:
            return
        self._last_led_pulse_time = now
        print(f"Turning LED ON for {duration} seconds.")
        led_line.set_value(1)
        # Schedule the LED to turn off
        t = threading.Timer(duration, lambda: led_line.set_value(0))
        t.daemon = True
        t.start()

    def save_video_from_buffer(self, width, height, fps):
        """Saves the buffered frames to a video file."""
        if self.is_recording or len(self.frame_buffer) == 0:
            return
        self.is_recording = True

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(RECORDING_FOLDER, f"{FILE_PREFIX}{timestamp}.mp4")
        
        # Make a copy to prevent issues with the buffer changing during saving
        frames_to_save = list(self.frame_buffer)

        print(f"Starting video save to {filename} with {len(frames_to_save)} frames.")
        
        # Run saving in a separate thread to not block the pipeline
        thread = threading.Thread(target=self._write_video_file, args=(filename, frames_to_save, width, height, fps))
        thread.daemon = True
        thread.start()

    def _write_video_file(self, filename, frames, width, height, fps):
        """The actual video writing logic."""
        # Use MP4V codec for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # Use a reasonable FPS, capping at 30
        save_fps = min(fps if fps > 0 else 30, 30)
        writer = cv2.VideoWriter(filename, fourcc, save_fps, (width, height))
        
        if not writer.isOpened():
            print(f"Error: Could not open video writer for {filename}")
            self.is_recording = False
            return

        for frame in frames:
            # Ensure the frame is in BGR format for OpenCV
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        writer.release()
        print(f"Successfully saved video: {filename}")
        self.is_recording = False


# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------
def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if not buffer:
        return Gst.PadProbeReturn.OK

    user_data.increment()
    
    # Get frame properties
    format, width, height = get_caps_from_pad(pad)
    fps = user_data.get_fps()
    
    # Dynamically update buffer size based on measured FPS
    user_data.update_buffer_size(fps)

    frame = None
    if user_data.use_frame and all((format, width, height)):
        frame = get_numpy_from_buffer(buffer, format, width, height)
        # Add frame to buffer
        user_data.frame_buffer.append(frame.copy())

    # Get detections
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    
    keypoints_map = get_keypoints()
    arms_crossed_detected = False

    for detection in detections:
        if detection.get_label() == "person":
            landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
            if not landmarks:
                continue

            points = landmarks[0].get_points()
            kpts = {}
            for name, index in keypoints_map.items():
                point = points[index]
                if point.confidence() > POSE_CONFIDENCE_THRESHOLD:
                    # Scale points to frame dimensions
                    bbox = detection.get_bbox()
                    kpts[name] = (
                        int((point.x() * bbox.width() + bbox.xmin()) * width),
                        int((point.y() * bbox.height() + bbox.ymin()) * height)
                    )

            # Check for the "arms crossed above head" pose
            required_kpts = ['left_wrist', 'right_wrist', 'left_shoulder', 'right_shoulder', 'nose']
            if all(kpt in kpts for kpt in required_kpts):
                lw, rw = kpts['left_wrist'], kpts['right_wrist']
                ls, rs = kpts['left_shoulder'], kpts['right_shoulder']
                ns = kpts['nose']

                # Condition 1: Wrists are crossed horizontally (left wrist is to the right of the right shoulder)
                wrists_crossed = lw[0] > rs[0] and rw[0] < ls[0]
                # Condition 2: Both wrists are above the nose
                arms_above_head = lw[1] < ns[1] and rw[1] < ns[1]

                if wrists_crossed and arms_above_head:
                    arms_crossed_detected = True
                    break # A person is in the pose, no need to check others

    # State machine for pose detection and triggering actions
    if arms_crossed_detected:
        if user_data.pose_start_time is None:
            user_data.pose_start_time = time.time()
        elif time.time() - user_data.pose_start_time >= POSE_DURATION_SECONDS:
            if not user_data.is_recording:
                print("Arms crossed pose held. Triggering actions.")
                user_data.pulse_led()
                user_data.save_video_from_buffer(width, height, fps)
                # Reset after triggering to prevent immediate re-trigger
                user_data.pose_start_time = None 
    else:
        # If pose is no longer detected, reset the timer
        if user_data.pose_start_time is not None:
            print("Pose broken.")
            user_data.pose_start_time = None
            
    # Optional: Draw keypoints on the frame for visualization
    if frame is not None and user_data.use_frame:
        if arms_crossed_detected:
             cv2.putText(frame, "ARMS CROSSED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(bgr_frame)
        
    return Gst.PadProbeReturn.OK

def get_keypoints():
    """Returns a dictionary mapping keypoint names to their indices."""
    return {
        'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
        'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
        'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
        'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16,
    }

if __name__ == "__main__":
    # The user_data object now manages state for buffering, recording, and pose detection
    user_data = user_app_callback_class()
    
    # Create and run the GStreamer application
    app = GStreamerPoseEstimationApp(app_callback, user_data)
    
    # Enable frame grabbing for visualization and saving
    app.set_use_frame(True) 
    
    app.run()

    # Cleanup GPIO on exit
    if LED_AVAILABLE:
        led_line.set_value(0)
        led_line.release()
        chip.close()
        print("GPIO cleaned up.")
