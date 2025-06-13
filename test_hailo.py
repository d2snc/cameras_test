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
import gc  # For garbage collection

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
BUFFER_SECONDS = 22
RECORDING_FOLDER = "recordings"
FILE_PREFIX = "arms_crossed_"
POSE_CONFIDENCE_THRESHOLD = 0.5
POSE_DURATION_SECONDS = 0.8

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
        self.frame_buffer = deque()
        self.is_recording = False
        self.pose_start_time = None
        self.pose_triggered_this_cycle = False
        self.buffer_lock = threading.Lock()
        self.total_frames = 0  # Total frames processed
        self.fps_frame_count = 0  # Frames for FPS calculation
        self.frames_captured = 0
        self.last_fps_time = time.time()
        self.fps = 30.0  # Default FPS
        self.max_buffer_frames = 1500  # Maximum buffer size to prevent memory issues
        
    def update_fps(self):
        """Calculate current FPS."""
        current_time = time.time()
        elapsed = current_time - self.last_fps_time
        if elapsed >= 1.0:
            self.fps = self.fps_frame_count / elapsed
            self.fps_frame_count = 0  # Reset only FPS counter, not total frames
            self.last_fps_time = current_time
            
            # Update buffer size based on FPS, but with a maximum limit
            target_size = min(int(self.fps * BUFFER_SECONDS), self.max_buffer_frames)
            if target_size > 0 and self.frame_buffer.maxlen != target_size:
                with self.buffer_lock:
                    # Don't copy all frames if buffer is too large
                    if len(self.frame_buffer) > target_size:
                        # Keep only the most recent frames
                        temp_list = list(self.frame_buffer)
                        self.frame_buffer = deque(temp_list[-target_size:], maxlen=target_size)
                    else:
                        current_frames = list(self.frame_buffer)
                        self.frame_buffer = deque(current_frames, maxlen=target_size)
                print(f"Buffer resized for {self.fps:.1f} FPS: {target_size} frames max")
        return self.fps
        
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
            fps = self.fps

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(RECORDING_FOLDER, f"{FILE_PREFIX}{timestamp}.avi")
        print(f"Saving {len(frames_to_save)} frames to {filename}")
        
        thread = threading.Thread(target=self._write_video, args=(filename, frames_to_save, fps))
        thread.daemon = True
        thread.start()

    def _write_video(self, filename, frames, fps):
        try:
            start_time = time.time()
            if not frames:
                print("No frames to write!")
                return
            
            print(f"Starting video write: {len(frames)} frames")
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out_fps = min(fps, 30.0) if fps > 0 else 30.0
            writer = cv2.VideoWriter(filename, fourcc, out_fps, (w, h))
            
            if not writer.isOpened():
                print(f"Error: Could not open video writer")
                return
            
            # Write frames with progress reporting
            for i, frame in enumerate(frames):
                # Check if writing is taking too long (timeout after 30 seconds)
                if time.time() - start_time > 30:
                    print("Warning: Video writing timeout - saving partial video")
                    break
                
                # Convert RGB to BGR for OpenCV
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(bgr_frame)
                
                # Progress report every 100 frames
                if i % 100 == 0 and i > 0:
                    elapsed = time.time() - start_time
                    fps_write = i / elapsed
                    print(f"Writing progress: {i}/{len(frames)} frames ({fps_write:.1f} fps)")
                    
            writer.release()
            elapsed_time = time.time() - start_time
            print(f"Video saved successfully: {filename}")
            print(f"Write time: {elapsed_time:.1f}s for {len(frames)} frames")
            
            # Clear references to free memory
            frames = None
            
        except Exception as e:
            print(f"Error writing video: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_recording = False

# -----------------------------------------------------------------------------------------------
# Callback function
# -----------------------------------------------------------------------------------------------
def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if not buffer:
        # Periodic garbage collection to prevent memory issues
    if user_data.total_frames % 500 == 0:
        gc.collect()
        if user_data.total_frames % 1000 == 0:
            print(f"System health check - Frames: {user_data.total_frames}, Captured: {user_data.frames_captured}")
    
    return Gst.PadProbeReturn.OK

    user_data.increment()
    user_data.total_frames += 1
    user_data.fps_frame_count += 1
    fps = user_data.update_fps()
    
    # Skip some frames if FPS is too high to reduce load
    if fps > 40 and user_data.total_frames % 2 == 0:
        # Process every other frame when FPS > 40
        skip_frame = True
    else:
        skip_frame = False
    
    # Get video properties
    width, height = 1280, 720
    caps_result = get_caps_from_pad(pad)
    if isinstance(caps_result, tuple) and len(caps_result) >= 3:
        format_str, width, height = caps_result
        if user_data.total_frames == 1:
            print(f"Video properties: {width}x{height}, format: {format_str}")
    
    # Extract frame only if not skipping
    frame = None
    if not skip_frame:
        # Method 1: Try the Hailo library function
        try:
            frame = get_numpy_from_buffer(buffer, 'RGB', width, height)
            if frame is not None and user_data.frames_captured == 0:
                print(f"Success: get_numpy_from_buffer worked!")
        except:
            pass
        
        # Method 2: Direct extraction
        if frame is None:
            frame = extract_rgb_frame_from_buffer(buffer, width, height)
            if frame is not None and user_data.frames_captured == 0:
                print(f"Success: Direct buffer extraction worked! Shape: {frame.shape}")
        
        # Add frame to buffer if we got one
        if frame is not None:
            with user_data.buffer_lock:
                # Check buffer health
                if len(user_data.frame_buffer) >= user_data.max_buffer_frames:
                    print(f"Warning: Buffer at maximum capacity ({user_data.max_buffer_frames} frames)")
                user_data.frame_buffer.append(frame.copy())
                user_data.frames_captured += 1
            
            # Clean up frame reference
            del frame
            
            if user_data.total_frames % 100 == 0:
                buffer_len = len(user_data.frame_buffer)
                print(f"Frame {user_data.total_frames}: Buffer has {buffer_len} frames, FPS: {fps:.1f}")
        else:
            if user_data.total_frames <= 5:
                buffer_size = buffer.get_size()
                print(f"Frame {user_data.total_frames}: Failed to extract. Buffer size: {buffer_size}")
    
    # Process pose detection
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    arms_crossed_detected = False
    
    # Only process pose detection every few frames if FPS is high
    if user_data.total_frames % (2 if fps > 40 else 1) == 0:
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
            if not user_data.pose_triggered_this_cycle and not user_data.is_recording:
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
    
    # Display handling - only update display every few frames if FPS is high
    if frame is not None and user_data.use_frame and (user_data.total_frames % (3 if fps > 40 else 1) == 0):
        display_frame = frame.copy()
        if arms_crossed_detected:
            cv2.putText(display_frame, "ARMS CROSSED!", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Add FPS and buffer info
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (width - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Buffer: {len(user_data.frame_buffer)}", (width - 150, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
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
    print(f"Buffer will hold up to {BUFFER_SECONDS} seconds of video")
    print(f"Pose must be held for {POSE_DURATION_SECONDS} seconds to trigger recording")
    print("Cross your arms above your head to trigger recording!")
    print("Press Ctrl+C to stop...")
    
    try:
        app = GStreamerPoseEstimationApp(app_callback, user_data)
        app.run()
    except KeyboardInterrupt:
        print("\nStopping application...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"\nShutting down...")
        print(f"Total frames processed: {getattr(user_data, 'total_frames', 0)}")
        print(f"Frames captured to buffer: {user_data.frames_captured}")
        
        # Wait for any ongoing recording to finish
        if user_data.is_recording:
            print("Waiting for video recording to complete...")
            timeout = 10
            start = time.time()
            while user_data.is_recording and time.time() - start < timeout:
                time.sleep(0.5)
        
        if LED_AVAILABLE:
            led_line.set_value(0)
            led_line.release()
            chip.close()
            print("GPIO cleaned up.")
        
        # Final garbage collection
        gc.collect()