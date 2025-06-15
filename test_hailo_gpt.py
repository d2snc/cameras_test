#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hailo Pose-Estimation — braços cruzados
╰▶ buffer circular redimensionável + gravação em thread
Jun/2025 (rev. 1b)
"""

import gi, os, cv2, hailo, time, threading, numpy as np, psutil, gc
from collections import deque
from datetime import datetime
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

# ───────── HAILO APPS INFRA ────────────────────────────────────────────────
from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.pose_estimation_pipeline import GStreamerPoseEstimationApp

# ───────── CONFIGURAÇÃO GERAL ──────────────────────────────────────────────
BUFFER_SECONDS       = 20
DETECTION_INTERVAL   = 0.25          # 4 Hz
POSE_HOLD_SECONDS    = 0.8
OUTPUT_DIR           = "recordings"
FILE_PREFIX_AUTO     = "arms_"
FILE_PREFIX_MANUAL   = "manual_"
MAX_RECORD_FPS       = 25
BUFFER_SCALE         = 0.5

# ───────── LED via libgpiod (opcional) ─────────────────────────────────────
LED_AVAILABLE = False
try:
    import gpiod
    CHIP, LINE = "gpiochip4", 17
    chip  = gpiod.Chip(CHIP)
    led   = chip.get_line(LINE)
    led.request(consumer="hailo-led",
                type=gpiod.LINE_REQ_DIR_OUT,
                default_vals=[0])
    LED_AVAILABLE = True
except Exception as e:
    print("[LED] desativado:", e)

def pulse_led(sec=3.0):
    if not LED_AVAILABLE: return
    led.set_value(1)
    t = threading.Timer(sec, lambda: led.set_value(0))
    t.daemon = True
    t.start()

# ───────── Fallback para extrair RGB de GstBuffer ─────────────────────────
def gstbuffer_to_rgb(buf, w, h):
    ok, info = buf.map(Gst.MapFlags.READ)
    if not ok: return None
    try:
        exp = w*h*3
        if len(info.data) < exp: return None
        arr = np.frombuffer(info.data[:exp], dtype=np.uint8)\
                .reshape((h, w, 3))
        return arr.copy()
    finally:
        buf.unmap(info)

# ───────── Classe de estado (herda utilidades da infra) ───────────────────
class UserState(app_callback_class):
    def __init__(self):
        super().__init__()
        self.buf_sec      = BUFFER_SECONDS
        self.fps_est      = 30.0
        self.frames       = deque(maxlen=int(self.fps_est*self.buf_sec))
        self.stamps       = deque(maxlen=int(self.fps_est*self.buf_sec))
        self.lock         = threading.Lock()

        # detecção & gravação
        self.pose_start   = None
        self.pose_ready   = False
        self.recording    = False
        self.trigger_fr   = None
        self.auto_cnt     = 0
        self.manual_cnt   = 0

# instancia estado
S = UserState()
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ───────── Thread de gravação ─────────────────────────────────────────────
def writer_thread(frames, fname, fps):
    try: os.nice(-10)
    except: pass
    if not frames: 
        S.recording = False; return
    h, w = frames[0].shape[:2]
    if BUFFER_SCALE < 1.0:
        w, h = int(w/BUFFER_SCALE), int(h/BUFFER_SCALE)
    fps = max(10.0, min(fps, MAX_RECORD_FPS))
    vw  = cv2.VideoWriter(
            fname,
            cv2.VideoWriter_fourcc(*'XVID'),
            fps, (w, h))
    if not vw.isOpened():
        print("[ERR] VideoWriter"); S.recording=False; return
    for i,f in enumerate(frames):
        if BUFFER_SCALE < 1.0:
            f = cv2.resize(f,(w,h),interpolation=cv2.INTER_LINEAR)
        vw.write(cv2.cvtColor(f,cv2.COLOR_RGB2BGR))
        if i and i%max(1,len(frames)//10)==0:
            print(f"[SAVE] {i/len(frames)*100:.0f}%…")
    vw.release()
    print("[SAVE] ok:", fname)
    S.recording=False
    with S.lock:
        S.frames.clear(); S.stamps.clear()
    gc.collect()

def save_buffer(manual=False):
    if S.recording: return
    with S.lock:
        if not S.frames: return
        out_frames=[f.copy() for f in S.frames]
        if S.trigger_fr is not None:
            out_frames.append(S.trigger_fr.copy())
    tag = FILE_PREFIX_MANUAL if manual else FILE_PREFIX_AUTO
    fname=os.path.join(OUTPUT_DIR,f"{tag}{datetime.now():%Y%m%d_%H%M%S}.avi")
    S.recording=True
    threading.Thread(target=writer_thread,
                     args=(out_frames,fname,S.fps_est),
                     daemon=True).start()
    pulse_led()

# ───────── Callback GStreamer ─────────────────────────────────────────────
def app_callback(pad, info, _user):
    buf = info.get_buffer();  now = time.time()
    if not buf: return Gst.PadProbeReturn.OK

    # FPS médio + resize do deque
    S.increment()  # contador da infra
    if not hasattr(app_callback,"_last_fps"): 
        app_callback._last_fps, app_callback._cnt = time.time(), 0
    app_callback._cnt += 1
    if now - app_callback._last_fps >= 1.0:
        S.fps_est = app_callback._cnt / (now - app_callback._last_fps)
        app_callback._cnt = 0
        app_callback._last_fps = now
        new_len = int(S.fps_est * S.buf_sec)
        if new_len != S.frames.maxlen:
            with S.lock:
                S.frames = deque(S.frames, maxlen=new_len)
                S.stamps = deque(S.stamps, maxlen=new_len)
            print(f"[BUF] → {new_len} quadros ({S.buf_sec}s @ {S.fps_est:.1f}fps)")

    # obtém props vídeo
    fmt, w, h = get_caps_from_pad(pad)

    # extrai (opcional) e guarda em buffer
    frame = None
    need_frame = (BUFFER_SCALE < 1.0) or (now - getattr(app_callback,"_last_det",0) < DETECTION_INTERVAL)
    if need_frame:
        try:
            frame = get_numpy_from_buffer(buf,"RGB",w,h)
        except: frame = gstbuffer_to_rgb(buf,w,h)
    if frame is not None and BUFFER_SCALE < 1.0:
        frame = cv2.resize(frame,(int(w*BUFFER_SCALE),int(h*BUFFER_SCALE)),
                           interpolation=cv2.INTER_AREA)
    if frame is not None:
        with S.lock:
            S.frames.append(frame); S.stamps.append(now)

    # detecção sub-amostrada
    if now - getattr(app_callback,"_last_det",0) >= DETECTION_INTERVAL:
        app_callback._last_det = now
        roi  = hailo.get_roi_from_buffer(buf)
        dets = roi.get_objects_typed(hailo.HAILO_DETECTION)
        crossed=False
        for d in dets:
            if d.get_label()!="person": continue
            lms = d.get_objects_typed(hailo.HAILO_LANDMARKS)
            if not lms: continue
            pts=lms[0].get_points()
            need=[0,5,6,9,10]               # nose, shoulders, wrists
            if any(pts[i].confidence()<0.5 for i in need): continue
            bbox=d.get_bbox()
            def XY(p): return ((p.x()*bbox.width()+bbox.xmin())*w,
                               (p.y()*bbox.height()+bbox.ymin())*h)
            nose,ls,rs,lw,rw = map(XY,[pts[i] for i in need])
            if lw[0]>rs[0] and rw[0]<ls[0] and lw[1]<nose[1] and rw[1]<nose[1]:
                crossed=True; break
        del roi,dets
        if crossed:
            if S.pose_start is None:
                S.pose_start=now
            elif now-S.pose_start>=POSE_HOLD_SECONDS and not S.pose_ready:
                S.pose_ready=True; S.auto_cnt+=1
                with S.lock: S.trigger_fr=frame.copy() if frame is not None else None
                save_buffer(manual=False)
        else:
            S.pose_start=None; S.pose_ready=False

    # GC/mem
    if S.get_count()%200==0:
        rss=psutil.Process().memory_info().rss/1024/1024
        if rss>700:
            print(f"[MEM] {rss:.0f} MB -> GC"); gc.collect()
        if rss>900 and len(S.frames)>int(0.5*S.frames.maxlen):
            with S.lock:
                keep=int(0.5*len(S.frames))
                S.frames=deque(list(S.frames)[-keep:],maxlen=S.frames.maxlen)
                S.stamps=deque(list(S.stamps)[-keep:],maxlen=S.stamps.maxlen)
            print("[MEM] buffer reduzido emergencialmente")

    return Gst.PadProbeReturn.OK

# ───────── Interface / janela OpenCV ───────────────────────────────────────
def ui_loop():
    print("[TECLAS] q  g  +  -")
    while True:
        with S.lock:
            disp=S.frames[-1].copy() if S.frames else None
        if disp is None:
            time.sleep(0.01); continue
        h,w=disp.shape[:2]
        txt=f"Auto:{S.auto_cnt}  Manual:{S.manual_cnt}  Buf:{len(S.frames)}/{S.frames.maxlen}"
        cv2.putText(disp,txt,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        if S.recording:
            cv2.putText(disp,"SALVANDO...",(w//2-120,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
        cv2.imshow("Hailo Pose",cv2.cvtColor(disp,cv2.COLOR_RGB2BGR))
        k=cv2.waitKey(1)&0xFF
        if k==ord('q'): break
        elif k in (ord('+'),ord('=')): S.buf_sec+=5
        elif k in (ord('-'),ord('_')) and S.buf_sec>5: S.buf_sec-=5
        elif k==ord('g'):
            with S.lock: S.trigger_fr=disp.copy()
            save_buffer(manual=True); S.manual_cnt+=1
        time.sleep(0.003)
    cv2.destroyAllWindows()

# ───────── MAIN ────────────────────────────────────────────────────────────
if __name__=="__main__":
    print(f"[BOOT] buf={BUFFER_SECONDS}s  det={1/DETECTION_INTERVAL:.1f}/s  scale={BUFFER_SCALE}")
    Gst.init(None)
    app = GStreamerPoseEstimationApp(app_callback, S)
    try:
        threading.Thread(target=app.run, daemon=True).start()
        ui_loop()
    finally:
        if LED_AVAILABLE:
            led.set_value(0); led.release(); chip.close()
        print("[EXIT] encerrado")
