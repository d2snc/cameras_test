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
import queue
import gc

from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.pose_estimation_pipeline import GStreamerPoseEstimationApp

# -----------------------------------------------------------------------------------------------
# GPIO Configuration using libgpiod
# -----------------------------------------------------------------------------------------------
try:
    import gpiod
    
    CHIP_NAME = "gpiochip0"         # /dev/gpiochip0
    LED_LINE_OFFSET = 17            # GPIO-17 (BCM) = physical pin 11
    chip = gpiod.Chip(CHIP_NAME)
    led_line = chip.get_line(LED_LINE_OFFSET)
    led_line.request(
        consumer="pose-detection-led",
        type=gpiod.LINE_REQ_DIR_OUT,
        default_vals=[0],
    )
    LED_AVAILABLE = True
    print("GPIO17 LED control initialized successfully")
except Exception as e:
    print(f"LED disabled (libgpiod unavailable): {e}")
    LED_AVAILABLE = False

_last_led_pulse = 0.0  # Prevent overlapping pulses

def pulse_led(duration: float = 3.0):
    """Turn on LED for 'duration' seconds, ignoring very close pulses."""
    global _last_led_pulse
    if not LED_AVAILABLE:
        return
    now = time.time()
    if now - _last_led_pulse < 0.2:      # Already blinking recently
        return
    _last_led_pulse = now

    led_line.set_value(1)                # Turn LED on
    print(f"LED turned ON for {duration} seconds")

    def _off():
        led_line.set_value(0)            # Turn LED off
        print("LED turned OFF")

    t = threading.Timer(duration, _off)
    t.daemon = True
    t.start()

# -----------------------------------------------------------------------------------------------
# Video saving configuration
# -----------------------------------------------------------------------------------------------
SAVE_FOLDER = "gravacoes"
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)
    print(f"Created folder: {SAVE_FOLDER}")

# -----------------------------------------------------------------------------------------------
# Optimized frame buffer using memory-mapped arrays
# -----------------------------------------------------------------------------------------------
class OptimizedFrameBuffer:
    def __init__(self, max_frames, frame_shape):
        self.max_frames = max_frames
        self.frame_shape = frame_shape
        self.current_index = 0
        self.is_full = False
        self.lock = threading.Lock()
        
        # Pre-allocate memory for all frames
        self.buffer = np.empty((max_frames, *frame_shape), dtype=np.uint8)
        self.timestamps = np.empty(max_frames, dtype=np.float64)
    
    def add_frame(self, frame):
        with self.lock:
            self.buffer[self.current_index] = frame
            self.timestamps[self.current_index] = time.time()
            self.current_index = (self.current_index + 1) % self.max_frames
            if self.current_index == 0:
                self.is_full = True
    
    def get_frames(self, num_frames):
        with self.lock:
            if not self.is_full and self.current_index < num_frames:
                # Not enough frames yet
                return self.buffer[:self.current_index].copy()
            
            if self.is_full:
                # Buffer is full, get last num_frames
                if num_frames >= self.max_frames:
                    # Return entire buffer in correct order
                    return np.concatenate([
                        self.buffer[self.current_index:],
                        self.buffer[:self.current_index]
                    ])
                else:
                    # Get last num_frames
                    start_idx = (self.current_index - num_frames) % self.max_frames
                    if start_idx < self.current_index:
                        return self.buffer[start_idx:self.current_index].copy()
                    else:
                        return np.concatenate([
                            self.buffer[start_idx:],
                            self.buffer[:self.current_index]
                        ])
            else:
                # Buffer not full yet
                start_idx = max(0, self.current_index - num_frames)
                return self.buffer[start_idx:self.current_index].copy()
    
    def __len__(self):
        return self.max_frames if self.is_full else self.current_index

# -----------------------------------------------------------------------------------------------
# Video writer with queue for non-blocking saves
# -----------------------------------------------------------------------------------------------
class AsyncVideoWriter:
    def __init__(self):
        self.save_queue = queue.Queue(maxsize=2)
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        self.is_saving = False
    
    def _worker(self):
        while True:
            try:
                task = self.save_queue.get(timeout=1)
                if task is None:  # Shutdown signal
                    break
                
                frames, filename, fps, shape = task
                self._save_video(frames, filename, fps, shape)
                self.save_queue.task_done()
                
            except queue.Empty:
                continue
    
    def _save_video(self, frames, filename, fps, shape):
        self.is_saving = True
        try:
            # Use more efficient codec
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filename, fourcc, fps, (shape[1], shape[0]))
            
            if not out.isOpened():
                print(f"Error: Could not initialize VideoWriter for {filename}")
                return
            
            # Write frames in chunks to improve performance
            chunk_size = 30  # Process 30 frames at a time
            total_frames = len(frames)
            
            for i in range(0, total_frames, chunk_size):
                chunk_end = min(i + chunk_size, total_frames)
                for j in range(i, chunk_end):
                    out.write(frames[j])
                
                # Progress update
                progress = (chunk_end / total_frames) * 100
                print(f"Saving progress: {progress:.1f}%", end='\r')
            
            out.release()
            print(f"\nVideo saved: {filename} ({total_frames} frames)")
            
            # Force garbage collection after saving
            del frames
            gc.collect()
            
        except Exception as e:
            print(f"Error saving video: {e}")
        finally:
            self.is_saving = False
    
    def save_video_async(self, frames, filename, fps, shape):
        try:
            self.save_queue.put_nowait((frames, filename, fps, shape))
            return True
        except queue.Full:
            print("Save queue is full, skipping this save")
            return False
    
    def is_busy(self):
        return self.is_saving or not self.save_queue.empty()

# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.arms_crossed_detected = False
        self.last_arms_crossed_time = None
        self.fps = 30  # Assumed FPS, adjust as needed
        self.cooldown_period = 5  # Seconds before allowing another video save
        self.status_text = "Arms Not Crossed"
        self.capture_trigger_time = None
        self.capture_extra_frames = 150  # 5 seconds of extra frames after trigger
        self.frames_after_trigger = 0
        self.triggered = False
        self.total_detections = 0  # Counter for total arms crossed detections
        
        # Initialize optimized frame buffer (will be set after first frame)
        self.frame_buffer = None
        self.buffer_seconds = 35  # 30 seconds before + 5 seconds after
        
        # Async video writer
        self.video_writer = AsyncVideoWriter()
        
        # Performance monitoring
        self.last_fps_update = time.time()
        self.frame_count_fps = 0
        self.current_fps = 30

    def init_buffer(self, frame_shape):
        """Initialize buffer with known frame shape"""
        if self.frame_buffer is None:
            max_frames = self.buffer_seconds * self.fps
            self.frame_buffer = OptimizedFrameBuffer(max_frames, frame_shape)
            print(f"Initialized optimized buffer for {max_frames} frames")

    def add_frame_to_buffer(self, frame):
        """Add frame to optimized buffer"""
        if self.frame_buffer is not None:
            self.frame_buffer.add_frame(frame)

    def save_buffer_to_video(self):
        """Save the frame buffer to a video file in the gravacoes folder"""
        if self.video_writer.is_busy():
            print("Video writer is busy, skipping save")
            return
        
        if self.frame_buffer is None or len(self.frame_buffer) == 0:
            print("No frames in buffer to save")
            return
        
        # Calculate how many frames to save
        total_frames_to_save = min(900 + self.frames_after_trigger, len(self.frame_buffer))
        frames_to_save = self.frame_buffer.get_frames(total_frames_to_save)
        
        if len(frames_to_save) == 0:
            print("No frames retrieved from buffer")
            return
        
        # Create video filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(SAVE_FOLDER, f"arms_crossed_{timestamp}.mp4")
        
        # Get frame dimensions
        h, w = frames_to_save[0].shape[:2]
        
        # Save video asynchronously
        if self.video_writer.save_video_async(frames_to_save, filename, self.fps, (h, w)):
            print(f"Video save queued: {filename}")
        
        self.triggered = False
        self.frames_after_trigger = 0

    def update_fps(self):
        """Update FPS calculation"""
        self.frame_count_fps += 1
        current_time = time.time()
        if current_time - self.last_fps_update >= 1.0:
            self.current_fps = self.frame_count_fps / (current_time - self.last_fps_update)
            self.frame_count_fps = 0
            self.last_fps_update = current_time

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
    
    # Update FPS
    user_data.update_fps()
    
    # Get the caps from the pad
    format, width, height = get_caps_from_pad(pad)

    # Get video frame
    frame = None
    if format is not None and width is not None and height is not None:
        # Always get frame for buffer and display
        user_data.use_frame = True
        frame = get_numpy_from_buffer(buffer, format, width, height)
        
        # Initialize buffer if needed
        if user_data.frame_buffer is None:
            user_data.init_buffer(frame.shape)

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

            # Pose estimation landmarks from detection
            landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
            if len(landmarks) != 0:
                points = landmarks[0].get_points()
                
                # Check if arms are crossed above head
                if check_arms_crossed_above_head(points, bbox, width, height, keypoints):
                    arms_crossed_in_frame = True
                
                # Draw keypoints only if visualization is needed
                if frame is not None and user_data.get_count() % 3 == 0:  # Reduce drawing frequency
                    # Draw only key keypoints for performance
                    key_points = ['left_wrist', 'right_wrist', 'nose']
                    for keypoint_name in key_points:
                        keypoint_index = keypoints[keypoint_name]
                        if keypoint_index < len(points):
                            point = points[keypoint_index]
                            x = int((point.x() * bbox.width() + bbox.xmin()) * width)
                            y = int((point.y() * bbox.height() + bbox.ymin()) * height)
                            
                            # Different colors for different keypoints
                            if 'wrist' in keypoint_name:
                                color = (0, 255, 255)  # Yellow for wrists
                            else:
                                color = (255, 0, 0)  # Blue for nose
                            
                            cv2.circle(frame, (x, y), 5, color, -1)

    # Update status and handle video saving
    current_time = time.time()
    
    if arms_crossed_in_frame:
        user_data.status_text = "Arms Crossed Above Head!"
        
        # Check if we should trigger video capture
        if not user_data.arms_crossed_detected and not user_data.triggered:
            user_data.arms_crossed_detected = True
            
            # Check cooldown period
            if (user_data.last_arms_crossed_time is None or 
                current_time - user_data.last_arms_crossed_time > user_data.cooldown_period):
                
                user_data.capture_trigger_time = current_time
                user_data.triggered = True
                user_data.frames_after_trigger = 0
                user_data.total_detections += 1
                print("Arms crossed detected! Capturing additional frames...")
                
                # Activate LED when arms are crossed
                pulse_led(3.0)
    else:
        user_data.status_text = "Arms Not Crossed"
        user_data.arms_crossed_detected = False
    
    # If we've triggered capture, continue capturing for a few more seconds
    if user_data.triggered:
        user_data.frames_after_trigger += 1
        
        # Check if we've captured enough frames after the trigger
        if user_data.frames_after_trigger >= user_data.capture_extra_frames:
            user_data.last_arms_crossed_time = current_time
            # Save video in a separate thread
            threading.Thread(target=user_data.save_buffer_to_video).start()
            print(f"Saving video with {user_data.frames_after_trigger} frames after trigger")

    if frame is not None:
        # Convert the frame to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Add status text to frame (simplified for performance)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Main status
        color = (0, 0, 255) if "Crossed" in user_data.status_text and "Not" not in user_data.status_text else (0, 255, 0)
        cv2.putText(frame, user_data.status_text, (10, 40), 
                   font, 1, color, 2, cv2.LINE_AA)
        
        # Performance stats
        perf_text = f"FPS: {user_data.current_fps:.1f} | Buffer: {len(user_data.frame_buffer) if user_data.frame_buffer else 0}"
        cv2.putText(frame, perf_text, (10, 70), 
                   font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Capture status if triggered
        if user_data.triggered:
            capture_text = f"Capturing: {user_data.frames_after_trigger}/{user_data.capture_extra_frames}"
            cv2.putText(frame, capture_text, (10, 100), 
                       font, 0.6, (255, 255, 0), 1, cv2.LINE_AA)
        
        # Save status
        if user_data.video_writer.is_busy():
            cv2.putText(frame, "SAVING VIDEO...", (frame.shape[1]//2 - 100, 40), 
                       font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Detection counter
        cv2.putText(frame, f"Detections: {user_data.total_detections}", (10, 130), 
                   font, 0.6, (255, 255, 0), 1, cv2.LINE_AA)
        
        # Add frame to buffer
        user_data.add_frame_to_buffer(frame)
        
        # Set frame for display
        user_data.set_frame(frame)

    return Gst.PadProbeReturn.OK

# -----------------------------------------------------------------------------------------------
# Cleanup function for GPIO
# -----------------------------------------------------------------------------------------------
def cleanup_gpio():
    """Clean up GPIO resources"""
    if LED_AVAILABLE:
        try:
            led_line.set_value(0)  # Make sure LED is off
            print("GPIO cleanup completed")
        except:
            pass

if __name__ == "__main__":
    try:
        # Create an instance of the user app callback class
        user_data = user_app_callback_class()
        app = GStreamerPoseEstimationApp(app_callback, user_data)
        
        print("\n=== Pose Detection with Arms Crossed Detection ===")
        print(f"Videos will be saved to: {os.path.abspath(SAVE_FOLDER)}")
        print(f"GPIO17 LED control: {'Enabled' if LED_AVAILABLE else 'Disabled'}")
        print("Detection: Arms crossed above head")
        print("LED Duration: 3 seconds when detected")
        print("Video: 30 seconds before + moment of crossing")
        print("Press Ctrl+C to exit\n")
        
        app.run()
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        cleanup_gpio()
        gc.collect()