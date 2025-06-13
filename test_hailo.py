#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pose estimation + arms-crossed detector with non-blocking
hardware-accelerated video recorder for Raspberry Pi.
2025-06-13 rev.2
"""
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib
import cv2, numpy as np, hailo, os, time, threading, multiprocessing as mp
from collections import deque
from datetime import datetime


from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.pose_estimation_pipeline import GStreamerPoseEstimationApp


# ----------------------------------------------------------
#  GPIO section identical to your original -----------------
# ----------------------------------------------------------
try:
    import gpiod
    _chip = gpiod.Chip("gpiochip0")
    _led  = _chip.get_line(17)
    _led.request(consumer="pose-led", type=gpiod.LINE_REQ_DIR_OUT, default_vals=[0])
    LED_OK = True
except Exception as e:
    print("GPIO17 LED disabled:", e)
    LED_OK = False

def pulse_led(duration=3.0):
    if not LED_OK: return
    _led.set_value(1)
    threading.Timer(duration, lambda: _led.set_value(0), daemon=True).start()

# ----------------------------------------------------------
#  Asynchronous saver process ------------------------------
# ----------------------------------------------------------
def saver_proc(frame_q: mp.Queue, stop_q: mp.Queue, w, h, fps):
    """
    Receives JPEG bytes on frame_q, writes an MP4 with hardware H.264.
    Terminates when stop_q gets any message.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    mp4_path = os.path.join("gravacoes", f"arms_crossed_{ts}.mp4")
    print(f"[Saver] Starting file {mp4_path}")

    # ---------- Try GStreamer (fast, HW) ----------
    gst_cmd = (
        f"gst-launch-1.0 -q appsrc name=appsrc is-live=true block=true "
        f"format=3 do-timestamp=true ! jpegdec ! "
        f"videoconvert ! video/x-raw,format=I420,width={w},height={h},framerate={fps}/1 ! "
        f"v4l2h264enc extra-controls=\"encode,frame_level_rate_control_enable=true\" "
        f"keyframe-period={fps*2} ! h264parse ! mp4mux name=mux ! filesink location={mp4_path}"
    )
    try:
        import subprocess, shlex
        proc = subprocess.Popen(shlex.split(gst_cmd), stdin=subprocess.PIPE)
        use_gst = True
    except Exception as e:
        print("[Saver] HW-encode unavailable, falling back to OpenCV:", e)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(mp4_path, fourcc, fps, (w, h))
        use_gst = False

    frames = 0
    while True:
        if not frame_q.empty():
            jpeg_bytes = frame_q.get()
            if jpeg_bytes is None:   # poison pill
                break
            if use_gst:
                proc.stdin.write(jpeg_bytes)
            else:
                np_buf  = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                frame   = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
                vw.write(frame)
            frames += 1
        elif not stop_q.empty():
            # Main process says "we're done"; send poison pill when queue drained
            frame_q.put(None)

        time.sleep(0.001)

    if use_gst:
        proc.stdin.close()
        proc.wait()
    else:
        vw.release()
    print(f"[Saver] Saved {frames} frames → {mp4_path}")

# ----------------------------------------------------------
#  Pose detection callback class ---------------------------
# ----------------------------------------------------------
class UserCB(app_callback_class):
    FPS               = 30
    BUFFER            = 1200             # ~40 s of 30 fps
    AFTER_TRIGGER     = FPS * 5          # 5 s extra
    COOLDOWN_S        = 5

    def __init__(self):
        super().__init__()
        self.buf = deque(maxlen=self.BUFFER)
        self.last_event   = 0.0
        self.triggered    = False
        self.after_count  = 0
        self.saver: mp.Process|None = None
        self.frame_q, self.stop_q  = None, None
        self.status = "Arms Not Crossed"
        self.tot = 0
        self.dim = None

    # -------------- helpers --------------
    def _launch_saver(self, h, w):
        self.frame_q = mp.Queue(maxsize=500)
        self.stop_q  = mp.Queue()
        self.saver   = mp.Process(target=saver_proc,
                                  args=(self.frame_q, self.stop_q, w, h, self.FPS),
                                  daemon=True)
        self.saver.start()

    def _feed_to_saver(self, frame):
        # Compress once, share bytes
        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok and self.saver is not None:
            self.frame_q.put(jpg.tobytes(), block=False)

# ----------------------------------------------------------
#  Helper: minimal keypoint set ----------------------------
# ----------------------------------------------------------
KP = {"nose":0,"left_wrist":9,"right_wrist":10}

def crossed(points, bbox, W, H):
    try:
        lw = points[KP["left_wrist"]]; rw = points[KP["right_wrist"]]; nz = points[KP["nose"]]
        lwx = (lw.x()*bbox.width()+bbox.xmin())*W
        lwy = (lw.y()*bbox.height()+bbox.ymin())*H
        rwx = (rw.x()*bbox.width()+bbox.xmin())*W
        rwy = (rw.y()*bbox.height()+bbox.ymin())*H
        nzy = (nz.y()*bbox.height()+bbox.ymin())*H
        return (lwy < nzy and rwy < nzy) and (lwx > rwx)
    except Exception:
        return False

# ----------------------------------------------------------
#  GStreamer pad probe -------------------------------------
# ----------------------------------------------------------
def app_cb(pad, info, user: UserCB):
    buf = info.get_buffer();  fmt,W,H = get_caps_from_pad(pad)
    if buf is None or fmt is None: return Gst.PadProbeReturn.OK
    frame = get_numpy_from_buffer(buf, fmt, W, H)

    # ---------- freeze-safe display while writing -----------
    if user.saver and user.saver.is_alive() and not user.triggered:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.putText(bgr, "SAVING…", (60,60), cv2.FONT_HERSHEY_SIMPLEX, 1.5,(0,0,255),3)
        user.set_frame(bgr);  return Gst.PadProbeReturn.OK

    # ---------- detection ----------------------------------
    roi = hailo.get_roi_from_buffer(buf)
    dets = roi.get_objects_typed(hailo.HAILO_DETECTION)
    crossed_now = False
    for d in dets:
        if d.get_label()!="person": continue
        lms = d.get_objects_typed(hailo.HAILO_LANDMARKS)
        if lms and crossed(lms[0].get_points(), d.get_bbox(), W, H):
            crossed_now = True
            break

    t = time.time()
    # ---------- event logic --------------------------------
    if crossed_now:
        user.status = "Arms Crossed!"
        if not user.triggered and t - user.last_event > user.COOLDOWN_S:
            user.triggered = True; user.after_count = 0; user.last_event = t; user.tot += 1
            pulse_led(3); print("*** Arms crossed, start capture ***")
    else:
        user.status = "Arms Not Crossed"

    if user.triggered:
        user.after_count += 1
        if user.after_count >= user.AFTER_TRIGGER:
            user.triggered = False
            user.stop_q.put(True)        # tell saver to finish
            print("*** capture done, writing file ***")

    # ---------- buffer management ---------------------------
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.putText(bgr, user.status, (10,40), cv2.FONT_HERSHEY_SIMPLEX,1,
                (0,0,255) if "Crossed" in user.status else (0,255,0),2)
    cv2.putText(bgr, f"Detections: {user.tot}", (10,75),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),1)

    # Save to buffer & feed saver
    user.buf.append(bgr)
    if user.triggered and user.after_count==1:
        # first frame after trigger: launch saver & send 30 s history
        if user.dim is None: user.dim = bgr.shape[:2]
        h,w = user.dim
        user._launch_saver(h,w)
        for f in list(user.buf)[-UserCB.BUFFER:]:
            user._feed_to_saver(f)
    elif user.saver and user.saver.is_alive():
        user._feed_to_saver(bgr)

    user.set_frame(bgr)
    return Gst.PadProbeReturn.OK

# ----------------------------------------------------------
#  Main -----------------------------------------------------
# ----------------------------------------------------------
if __name__ == "__main__":
    Gst.init(None)
    os.makedirs("gravacoes", exist_ok=True)
    try:
        user = UserCB()
        app  = GStreamerPoseEstimationApp(app_cb, user)
        print("Running. Ctrl-C to quit.")
        app.run()
    finally:
        if LED_OK: _led.set_value(0)
