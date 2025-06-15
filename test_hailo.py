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
import gc  # Added missing import
import psutil  # Added missing import

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
    CHIP_NAME = "gpiochip4"
    LED_LINE_OFFSET = 17
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
    print(f"LED disabled: {e}")

# -----------------------------------------------------------------------------------------------
# Recording Configuration
# -----------------------------------------------------------------------------------------------
BUFFER_SECONDS = 20  # Changed to 20 seconds as requested
RECORDING_FOLDER = "recordings"
FILE_PREFIX = "arms_crossed_"
POSE_CONFIDENCE_THRESHOLD = 0.5
POSE_DURATION_SECONDS = 0.8

# Memory optimization settings
BUFFER_FPS = 30  # Store 30 FPS in buffer for smooth video
BUFFER_SCALE = 0.5  # Scale frames to 50% for buffer storage

if not os.path.exists(RECORDING_FOLDER):
    os.makedirs(RECORDING_FOLDER)

# -----------------------------------------------------------------------------------------------
# Direct frame extraction function
# -----------------------------------------------------------------------------------------------
def extract_rgb_frame_from_buffer(buffer, width=1280, height=720):
    """Extract RGB frame directly from GStreamer buffer."""
    expected_size = width * height * 3  # RGB = 3 channels
    
    # Map the buffer to access its data
    success, map_info = buffer.map(Gst.MapFlags.READ)
    if not success:
        print("Failed to map buffer")
        return None
    
    try:
        buffer_size = len(map_info.data)
        
        # Check if buffer size matches expected RGB size
        if buffer_size == expected_size:
            # Perfect match - it's RGB data
            frame_data = np.frombuffer(map_info.data, dtype=np.uint8)
            frame = frame_data.reshape((height, width, 3))
            frame_copy = frame.copy()  # Copy before unmapping
            return frame_copy
            
        elif buffer_size == width * height * 4:
            # RGBA data - extract RGB channels
            frame_data = np.frombuffer(map_info.data, dtype=np.uint8)
            frame = frame_data.reshape((height, width, 4))
            frame_copy = frame[:, :, :3].copy()  # Take only RGB, ignore A
            return frame_copy
            
        elif buffer_size > expected_size:
            # Buffer is larger - try to extract RGB from the beginning
            print(f"Buffer larger than expected: {buffer_size} > {expected_size}")
            frame_data = np.frombuffer(map_info.data[:expected_size], dtype=np.uint8)
            frame = frame_data.reshape((height, width, 3))
            frame_copy = frame.copy()
            return frame_copy
            
        else:
            print(f"Buffer too small: {buffer_size} < {expected_size}")
            return None
            
    except Exception as e:
        print(f"Error extracting frame: {e}")
        return None
    finally:
        buffer.unmap(map_info)

# -----------------------------------------------------------------------------------------------
# Callback class
# -----------------------------------------------------------------------------------------------
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        # Calculate max buffer size based on BUFFER_FPS instead of actual FPS
        self.max_buffer_frames = BUFFER_FPS * BUFFER_SECONDS  # 30 * 20 = 600 frames max
        self.frame_buffer = deque(maxlen=self.max_buffer_frames)
        self.is_recording = False
        self.pose_start_time = None
        self.pose_triggered_this_cycle = False
        self.buffer_lock = threading.Lock()
        self.frame_count = 0
        self.frames_captured = 0
        self.last_fps_time = time.time()
        self.fps = 30.0  # Default FPS
        self.last_buffer_frame_time = 0  # For frame skipping
        self.frame_skip_interval = 1.0 / BUFFER_FPS  # Store frames at BUFFER_FPS rate
        
        # Memory monitoring
        self.last_memory_check = time.time()
        self.memory_warning_shown = False
        
    def update_fps(self):
        """Calculate current FPS."""
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
            print(f"Current FPS: {self.fps:.1f}, Buffer: {len(self.frame_buffer)}/{self.max_buffer_frames} frames")
        return self.fps
        
    def should_store_frame(self):
        """Determine if current frame should be stored based on target buffer FPS."""
        current_time = time.time()
        if current_time - self.last_buffer_frame_time >= self.frame_skip_interval:
            self.last_buffer_frame_time = current_time
            return True
        return False
        
    def pulse_led(self, duration=3.0):
        if not LED_AVAILABLE: return
        print(f"LED ON for {duration} seconds.")
        led_line.set_value(1)
        t = threading.Timer(duration, lambda: led_line.set_value(0))
        t.daemon = True
        t.start()

    def save_video_from_buffer(self):
        with self.buffer_lock:
            if self.is_recording or len(self.frame_buffer) == 0:
                print(f"Cannot save: Recording={self.is_recording}, Buffer size={len(self.frame_buffer)}")
                return
            self.is_recording = True
            frames_to_save = list(self.frame_buffer)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(RECORDING_FOLDER, f"{FILE_PREFIX}{timestamp}.avi")
        duration = len(frames_to_save) / BUFFER_FPS
        print(f"Saving {len(frames_to_save)} frames to {filename} (approximately {duration:.1f} seconds)")
        
        thread = threading.Thread(target=self._write_video, args=(filename, frames_to_save))
        thread.daemon = True
        thread.start()

    def _write_video(self, filename, frames):
        try:
            if not frames:
                print("No frames to write!")
                return
                
            # Get original dimensions from first frame
            h, w = frames[0].shape[:2]
            
            # Scale back to original size if needed
            if BUFFER_SCALE < 1.0:
                w = int(w / BUFFER_SCALE)
                h = int(h / BUFFER_SCALE)
            
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(filename, fourcc, BUFFER_FPS, (w, h))
            
            if not writer.isOpened():
                print(f"Error: Could not open video writer")
                return
                
            for i, frame in enumerate(frames):
                # Scale frame back to original size if it was scaled down
                if BUFFER_SCALE < 1.0:
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
                
                # Convert RGB to BGR for OpenCV
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(bgr_frame)
                
                # Progress update
                if i % 50 == 0:
                    print(f"Writing frame {i}/{len(frames)}...")
                    
            writer.release()
            print(f"Video saved successfully: {filename}")
            
            # Force garbage collection after saving
            gc.collect()
            
        except Exception as e:
            print(f"Error writing video: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_recording = False

    def check_memory_usage(self):
        """Monitor memory usage and warn if getting high."""
        current_time = time.time()
        if current_time - self.last_memory_check > 5.0:  # Check every 5 seconds
            self.last_memory_check = current_time
            try:
                memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                memory_percent = psutil.Process().memory_percent()
                
                if memory_percent > 70 and not self.memory_warning_shown:
                    print(f"WARNING: High memory usage: {memory_mb:.1f}MB ({memory_percent:.1f}%)")
                    self.memory_warning_shown = True
                    
                    # Emergency buffer clear if memory is critically high
                    if memory_percent > 85:
                        print("CRITICAL: Memory usage too high! Clearing oldest 25% of buffer...")
                        with self.buffer_lock:
                            # Keep only the most recent 75% of frames
                            keep_size = int(len(self.frame_buffer) * 0.75)
                            temp_frames = list(self.frame_buffer)[-keep_size:]
                            self.frame_buffer.clear()
                            self.frame_buffer.extend(temp_frames)
                        gc.collect()
                        
                elif memory_percent < 60:
                    self.memory_warning_shown = False
                    
            except Exception as e:
                print(f"Memory check error: {e}")

# -----------------------------------------------------------------------------------------------
# Callback function
# -----------------------------------------------------------------------------------------------
def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if not buffer:
        return Gst.PadProbeReturn.OK

    user_data.increment()
    user_data.frame_count += 1
    fps = user_data.update_fps()
    
    # Check memory usage periodically
    user_data.check_memory_usage()
    
    # Get video properties
    width, height = 1280, 720
    caps_result = get_caps_from_pad(pad)
    if isinstance(caps_result, tuple) and len(caps_result) >= 3:
        format_str, width, height = caps_result
        if user_data.frame_count == 1:
            print(f"Video properties: {width}x{height}, format: {format_str}")
            print(f"Buffer will store frames at {BUFFER_FPS} FPS, scaled to {BUFFER_SCALE*100}%")
    
    # Only process frame extraction if we should store this frame
    should_store = user_data.should_store_frame()
    frame = None
    
    if should_store:
        # Extract frame - try Hailo method first, then direct extraction
        # Method 1: Try the Hailo library function
        try:
            frame = get_numpy_from_buffer(buffer, 'RGB', width, height)
            if frame is not None and user_data.frames_captured == 0:
                print(f"Success: get_numpy_from_buffer worked!")
        except Exception:
            pass
        
        # Method 2: Direct extraction
        if frame is None:
            frame = extract_rgb_frame_from_buffer(buffer, width, height)
            if frame is not None and user_data.frames_captured == 0:
                print(f"Success: Direct buffer extraction worked! Shape: {frame.shape}")
    
    # Add frame to buffer if we got one and should store it
    if frame is not None and should_store:
        # Scale down frame to save memory
        if BUFFER_SCALE < 1.0:
            scaled_height = int(height * BUFFER_SCALE)
            scaled_width = int(width * BUFFER_SCALE)
            scaled_frame = cv2.resize(frame, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
        else:
            scaled_frame = frame
            
        with user_data.buffer_lock:
            user_data.frame_buffer.append(scaled_frame.copy())
            user_data.frames_captured += 1
        
        if user_data.frames_captured % 50 == 0:
            buffer_len = len(user_data.frame_buffer)
            try:
                memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                print(f"Buffer: {buffer_len}/{user_data.max_buffer_frames} frames, Memory: {memory_mb:.1f}MB")
            except:
                print(f"Buffer: {buffer_len}/{user_data.max_buffer_frames} frames")
    
    # Process pose detection (always do this, not just when storing frames)
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    arms_crossed_detected = False
    
    for detection in detections:
        if detection.get_label() == "person":
            landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
            if not landmarks: continue
            
            points = landmarks[0].get_points()
            keypoints = {}
            keypoint_indices = {
                'nose': 0, 'left_wrist': 9, 'right_wrist': 10,
                'left_shoulder': 5, 'right_shoulder': 6
            }
            
            bbox = detection.get_bbox()
            for name, idx in keypoint_indices.items():
                point = points[idx]
                if point.confidence() > POSE_CONFIDENCE_THRESHOLD:
                    keypoints[name] = (
                        int((point.x() * bbox.width() + bbox.xmin()) * width),
                        int((point.y() * bbox.height() + bbox.ymin()) * height)
                    )
            
            if all(k in keypoints for k in keypoint_indices.keys()):
                lw, rw = keypoints['left_wrist'], keypoints['right_wrist']
                ls, rs = keypoints['left_shoulder'], keypoints['right_shoulder']
                ns = keypoints['nose']
                
                wrists_crossed = lw[0] > rs[0] and rw[0] < ls[0]
                arms_above_head = lw[1] < ns[1] and rw[1] < ns[1]
                
                if wrists_crossed and arms_above_head:
                    arms_crossed_detected = True
                    break
    
    # Handle pose detection timing
    if arms_crossed_detected:
        if user_data.pose_start_time is None:
            user_data.pose_start_time = time.time()
            print("Arms crossed pose detected!")
        elif time.time() - user_data.pose_start_time >= POSE_DURATION_SECONDS:
            if not user_data.pose_triggered_this_cycle:
                buffer_len = len(user_data.frame_buffer)
                print(f"Pose held for {POSE_DURATION_SECONDS}s! Triggering save. Buffer: {buffer_len} frames")
                user_data.pulse_led()
                user_data.save_video_from_buffer()
                user_data.pose_triggered_this_cycle = True
    else:
        if user_data.pose_start_time is not None:
            print("Pose broken")
            user_data.pose_start_time = None
            user_data.pose_triggered_this_cycle = False
    
    # Display handling - use original frame or get new one if needed
    if user_data.use_frame:
        display_frame = frame
        if display_frame is None and arms_crossed_detected:
            # Need to extract frame for display
            try:
                display_frame = get_numpy_from_buffer(buffer, 'RGB', width, height)
            except:
                display_frame = extract_rgb_frame_from_buffer(buffer, width, height)
        
        if display_frame is not None:
            if arms_crossed_detected:
                cv2.putText(display_frame, "ARMS CROSSED!", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Convert to BGR for display
            bgr_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
            user_data.set_frame(bgr_frame)
    
    return Gst.PadProbeReturn.OK

# -----------------------------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Initialize GStreamer
    Gst.init(None)
    
    user_data = user_app_callback_class()
    user_data.use_frame = True
    
    print("Starting Hailo pose detection app...")
    print(f"Buffer will hold up to {BUFFER_SECONDS} seconds of video at {BUFFER_FPS} FPS")
    print(f"Frames will be scaled to {BUFFER_SCALE*100}% to save memory")
    print(f"Maximum buffer size: {user_data.max_buffer_frames} frames")
    print(f"Estimated memory usage: ~{(user_data.max_buffer_frames * 640 * 360 * 3) / (1024*1024):.0f}MB")
    print(f"Pose must be held for {POSE_DURATION_SECONDS} seconds to trigger recording")
    print("Cross your arms above your head to trigger recording!")
    
    try:
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        print(f"Initial memory usage: {memory_mb:.1f}MB")
    except:
        pass
    
    app = GStreamerPoseEstimationApp(app_callback, user_data)
    app.run()
    
    print(f"\nShutting down...")
    print(f"Total frames processed: {user_data.frame_count}")
    print(f"Frames captured to buffer: {user_data.frames_captured}")
    
    if LED_AVAILABLE:
        led_line.set_value(0)
        led_line.release()
        chip.close()
        print("GPIO cleaned up.")