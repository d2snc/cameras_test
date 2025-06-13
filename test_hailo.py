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
        self.pose_triggered_this_cycle = False
        
        # --- Adopting the locking strategy from the working example ---
        self.buffer_lock = threading.Lock()
        # --------------------------------------------------------------
        
        self._fps_last_time = time.time()
        self._fps_last_frame_count = 0
        self._fps = 0.0
        
        # Store video properties once detected
        self.video_width = None
        self.video_height = None
        self.video_format = None
        self.caps_detected = False
        
        # Debug flags
        self.debug_count = 0
        self.frame_count = 0
        self.caps_printed = False

    def get_fps(self):
        """Calculates and returns the current FPS."""
        total_frames = self.get_count()
        current_time = time.time()
        elapsed_time = current_time - self._fps_last_time
        
        if elapsed_time >= 1.0:
            frames_since_last = total_frames - self._fps_last_frame_count
            self._fps = frames_since_last / elapsed_time
            self._fps_last_time = current_time
            self._fps_last_frame_count = total_frames
        return self._fps

    def update_buffer_size(self, fps):
        """Dynamically adjusts the buffer size based on FPS."""
        if fps > 0:
            max_len = int(fps * BUFFER_SECONDS)
            # Lock is not strictly necessary here as maxlen is atomic, but it's good practice
            with self.buffer_lock:
                if self.frame_buffer.maxlen != max_len:
                    current_frames = list(self.frame_buffer)
                    self.frame_buffer = deque(current_frames, maxlen=max_len)
                    print(f"Frame buffer resized to {max_len} frames for {fps:.1f} FPS.")

    def pulse_led(self, duration=3.0):
        """Turns the LED on for a specified duration."""
        if not LED_AVAILABLE: return
        print(f"Turning LED ON for {duration} seconds.")
        led_line.set_value(1)
        t = threading.Timer(duration, lambda: led_line.set_value(0))
        t.daemon = True
        t.start()

    def save_video_from_buffer(self, width, height, fps):
        """Saves the buffered frames to a video file."""
        with self.buffer_lock:
            if self.is_recording or len(self.frame_buffer) == 0:
                if len(self.frame_buffer) == 0:
                    print("ERROR: Video not saved because the frame buffer is empty.")
                return
            self.is_recording = True
            frames_to_save = list(self.frame_buffer)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(RECORDING_FOLDER, f"{FILE_PREFIX}{timestamp}.avi")
        print(f"Starting video save to {filename} with {len(frames_to_save)} frames.")
        thread = threading.Thread(target=self._write_video_file, args=(filename, frames_to_save, width, height, fps))
        thread.daemon = True
        thread.start()

    def _write_video_file(self, filename, frames, width, height, fps):
        """The actual video writing logic."""
        try:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            save_fps = min(fps if fps > 0 else 30, 30)
            writer = cv2.VideoWriter(filename, fourcc, save_fps, (width, height))
            if not writer.isOpened():
                print(f"Error: Could not open video writer for {filename}")
                self.is_recording = False
                return
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
            print(f"Successfully saved video: {filename}")
        except Exception as e:
            print(f"!!! FATAL ERROR during video writing: {e}")
        finally:
            self.is_recording = False

# -----------------------------------------------------------------------------------------------
# Alternative frame extraction function
# -----------------------------------------------------------------------------------------------
def extract_frame_directly(buffer, width, height, format_str):
    """Try to extract frame directly from buffer."""
    try:
        # Get buffer size and map it
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return None
            
        # Calculate expected size based on format
        if format_str in ['RGB', 'BGR']:
            expected_size = width * height * 3
            channels = 3
        elif format_str in ['RGBA', 'BGRA']:
            expected_size = width * height * 4
            channels = 4
        elif format_str in ['GRAY8', 'GRAY16_LE', 'GRAY16_BE']:
            expected_size = width * height
            channels = 1
        else:
            # For YUV formats, size calculation is different
            buffer.unmap(map_info)
            return None
            
        # Check if buffer size matches
        if len(map_info.data) < expected_size:
            print(f"Buffer size mismatch: got {len(map_info.data)}, expected {expected_size}")
            buffer.unmap(map_info)
            return None
            
        # Create numpy array from buffer data
        frame = np.frombuffer(map_info.data[:expected_size], dtype=np.uint8)
        
        if channels == 1:
            frame = frame.reshape((height, width))
        else:
            frame = frame.reshape((height, width, channels))
            
        # Make a copy before unmapping
        frame_copy = frame.copy()
        buffer.unmap(map_info)
        
        return frame_copy
        
    except Exception as e:
        print(f"Direct extraction failed: {e}")
        try:
            buffer.unmap(map_info)
        except:
            pass
        return None

# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------
def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if not buffer:
        return Gst.PadProbeReturn.OK

    user_data.increment()
    user_data.frame_count += 1
    
    # Extensive debugging for first few frames
    if user_data.debug_count < 5:
        print(f"\n=== DEBUG Frame {user_data.debug_count} ===")
        print(f"Buffer: {buffer}")
        print(f"Buffer size: {buffer.get_size()}")
        
        # Try to get caps information
        caps = get_caps_from_pad(pad)
        if caps and not user_data.caps_printed:
            print(f"Caps string: {caps.to_string()}")
            user_data.caps_printed = True
            
            # Parse caps to get properties
            structure = caps.get_structure(0)
            print(f"Structure name: {structure.get_name()}")
            print(f"Number of fields: {structure.n_fields()}")
            
            # Print all fields
            for i in range(structure.n_fields()):
                field_name = structure.nth_field_name(i)
                field_value = structure.get_value(field_name)
                print(f"  {field_name}: {field_value}")
        
        user_data.debug_count += 1
    
    # Try to get video properties
    if not user_data.caps_detected:
        try:
            caps = get_caps_from_pad(pad)
            if caps:
                structure = caps.get_structure(0)
                user_data.video_width = structure.get_value('width')
                user_data.video_height = structure.get_value('height')
                
                # Try different ways to get format
                format_string = structure.get_value('format')
                if not format_string:
                    # Check if it's under a different name
                    struct_name = structure.get_name()
                    if 'video/x-raw' in struct_name:
                        # Try to extract format from structure
                        for i in range(structure.n_fields()):
                            field_name = structure.nth_field_name(i)
                            if 'format' in field_name.lower():
                                format_string = structure.get_value(field_name)
                                break
                
                user_data.video_format = format_string if format_string else 'RGB'
                user_data.caps_detected = True
                print(f"\nDetected video properties: {user_data.video_width}x{user_data.video_height}, format: {user_data.video_format}")
        except Exception as e:
            print(f"Failed to get caps from pad: {e}")
            # Use fallback values
            user_data.video_width = 1280
            user_data.video_height = 720
            user_data.video_format = 'RGB'
            user_data.caps_detected = True
            print(f"Using fallback video properties: 1280x720, format: RGB")
    
    width = user_data.video_width
    height = user_data.video_height
    format = user_data.video_format
    
    fps = user_data.get_fps()
    user_data.update_buffer_size(fps)

    frame = None
    if user_data.use_frame:
        # First, try the standard method
        try:
            frame = get_numpy_from_buffer(buffer, format, width, height)
            if frame is not None and user_data.debug_count <= 5:
                print(f"SUCCESS: get_numpy_from_buffer worked with format '{format}'")
        except Exception as e:
            if user_data.debug_count <= 5:
                print(f"Exception in get_numpy_from_buffer: {e}")
        
        # If standard method failed, try alternatives
        if frame is None:
            if user_data.debug_count <= 5:
                print(f"Standard extraction failed. Trying alternatives...")
            
            # Try different format strings
            format_alternatives = ['RGB', 'BGR', 'RGBA', 'BGRA', 'I420', 'YV12', 'NV12', 'NV21', 'YUY2', 'UYVY']
            for alt_format in format_alternatives:
                try:
                    frame = get_numpy_from_buffer(buffer, alt_format, width, height)
                    if frame is not None:
                        print(f"SUCCESS: Format '{alt_format}' worked!")
                        user_data.video_format = alt_format
                        break
                except:
                    pass
            
            # If still no success, try direct extraction
            if frame is None:
                if user_data.debug_count <= 5:
                    print("Trying direct buffer extraction...")
                frame = extract_frame_directly(buffer, width, height, format)
                if frame is not None:
                    print("SUCCESS: Direct extraction worked!")
        
        # If we have a frame, add it to buffer
        if frame is not None:
            with user_data.buffer_lock:
                user_data.frame_buffer.append(frame.copy())
            
            # Print buffer status periodically
            if user_data.frame_count % 30 == 0:
                with user_data.buffer_lock:
                    buffer_len = len(user_data.frame_buffer)
                print(f"Frame {user_data.frame_count}: Buffer has {buffer_len} frames")
        else:
            if user_data.debug_count <= 10:
                print(f"WARNING: Frame {user_data.frame_count} - Could not extract frame!")

    # --- The rest of the logic remains the same ---
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    keypoints_map = get_keypoints()
    arms_crossed_detected = False

    for detection in detections:
        if detection.get_label() == "person":
            landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
            if not landmarks: continue
            points = landmarks[0].get_points()
            kpts = {}
            for name, index in keypoints_map.items():
                point = points[index]
                if point.confidence() > POSE_CONFIDENCE_THRESHOLD:
                    bbox = detection.get_bbox()
                    kpts[name] = (
                        int((point.x() * bbox.width() + bbox.xmin()) * width),
                        int((point.y() * bbox.height() + bbox.ymin()) * height)
                    )
            required_kpts = ['left_wrist', 'right_wrist', 'left_shoulder', 'right_shoulder', 'nose']
            if all(kpt in kpts for kpt in required_kpts):
                lw, rw = kpts['left_wrist'], kpts['right_wrist']
                ls, rs = kpts['left_shoulder'], kpts['right_shoulder']
                ns = kpts['nose']
                wrists_crossed = lw[0] > rs[0] and rw[0] < ls[0]
                arms_above_head = lw[1] < ns[1] and rw[1] < ns[1]
                if wrists_crossed and arms_above_head:
                    arms_crossed_detected = True
                    break

    if arms_crossed_detected:
        if user_data.pose_start_time is None:
            user_data.pose_start_time = time.time()
        elif time.time() - user_data.pose_start_time >= POSE_DURATION_SECONDS and not user_data.pose_triggered_this_cycle:
            if not user_data.is_recording:
                with user_data.buffer_lock:
                    buffer_len = len(user_data.frame_buffer)
                print(f"Arms crossed pose held. Buffer has {buffer_len} frames. Triggering actions.")
                user_data.pulse_led()
                user_data.save_video_from_buffer(width, height, fps)
                user_data.pose_triggered_this_cycle = True
    else:
        if user_data.pose_start_time is not None:
            print("Pose broken.")
            user_data.pose_start_time = None
            user_data.pose_triggered_this_cycle = False
            
    # Handle display
    if frame is not None and user_data.use_frame:
        display_frame = frame.copy()
        if arms_crossed_detected:
            cv2.putText(display_frame, "ARMS CROSSED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Convert to BGR for display
        if len(display_frame.shape) == 3 and display_frame.shape[2] == 3:
            bgr_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
        else:
            bgr_frame = display_frame
            
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
    user_data = user_app_callback_class()
    user_data.use_frame = True
    app = GStreamerPoseEstimationApp(app_callback, user_data)
    app.run()
    if LED_AVAILABLE:
        led_line.set_value(0)
        led_line.release()
        chip.close()
        print("GPIO cleaned up.")