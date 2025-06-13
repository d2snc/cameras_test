#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline Hailo + GStreamer que:
  • mantém buffer circular de 30 s;
  • grava os 30 s anteriores quando detectar braços cruzados;
  • sobrepõe texto “BRAÇOS CRUZADOS” (vermelho) ou “Braços não cruzados” (verde).
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import os, time, threading
from collections import deque
import numpy as np
import cv2
import hailo

from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.pose_estimation_pipeline import GStreamerPoseEstimationApp

# ═══════════════════════════════════════════════════════════════════════════
# Utilidades de pose
# ═══════════════════════════════════════════════════════════════════════════
def coco_keypoints():
    return {
        'nose': 0,  'left_eye': 1,  'right_eye': 2,
        'left_ear': 3,  'right_ear': 4,
        'left_shoulder': 5, 'right_shoulder': 6,
        'left_elbow': 7,   'right_elbow': 8,
        'left_wrist': 9,   'right_wrist': 10,
        'left_hip': 11,    'right_hip': 12,
        'left_knee': 13,   'right_knee': 14,
        'left_ankle': 15,  'right_ankle': 16,
    }


def arms_crossed(points, kpts, margin=0.12):
    ls = points[kpts['left_shoulder']]
    rs = points[kpts['right_shoulder']]
    lw = points[kpts['left_wrist']]
    rw = points[kpts['right_wrist']]

    mid_y = 0.5 * (ls.y() + rs.y())
    if lw.y() > mid_y and rw.y() > mid_y:
        x_min, x_max = min(ls.x(), rs.x()) - margin, max(ls.x(), rs.x()) + margin
        return (x_min < lw.x() < x_max) and (x_min < rw.x() < x_max)
    return False


# ═══════════════════════════════════════════════════════════════════════════
# Classe com buffer circular
# ═══════════════════════════════════════════════════════════════════════════
class UserCallback(app_callback_class):
    def __init__(self, fps=30, clip_sec=30, out_dir="gravacoes"):
        super().__init__()
        self.fps = fps
        self.clip_len = fps * clip_sec
        self.buffer = deque(maxlen=self.clip_len)
        self.kpts = coco_keypoints()

        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.clip_idx = 0

        self.saving = False
        self.last_event_ts = 0.0      # histerese de 2 s
        self.overlay_frames = 0       # manter texto vermelho ~1 s

        self.use_frame = True         # AVISO: precisa estar ligado

    # -------------------- gravação assíncrona ---------------------------------
    def _save_clip(self, frames, w, h):
        ts = time.strftime("%Y%m%d_%H%M%S")
        fname = f"{self.out_dir}/cruzou_{ts}_{self.clip_idx:04}.mp4"
        vw = cv2.VideoWriter(fname,
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             self.fps, (w, h))
        for f in frames:
            vw.write(f)
        vw.release()
        print(f"[INFO] Clip salvo: {fname}")
        self.clip_idx += 1
        self.saving = False


# ═══════════════════════════════════════════════════════════════════════════
# Callback de pad
# ═══════════════════════════════════════════════════════════════════════════
def app_callback(pad, info, user: UserCallback):
    buf = info.get_buffer()
    if buf is None:
        return Gst.PadProbeReturn.OK

    user.increment()
    fmt, w, h = get_caps_from_pad(pad)

    # 1. converte para numpy RGB --------------------------------------------
    frame_rgb = None
    if user.use_frame and fmt and w and h:
        frame_rgb = get_numpy_from_buffer(buf, fmt, w, h)

    # 2. inferência ----------------------------------------------------------
    arms_event = False
    if frame_rgb is not None:
        roi = hailo.get_roi_from_buffer(buf)
        for det in roi.get_objects_typed(hailo.HAILO_DETECTION):
            if det.get_label() != "person":
                continue
            lmks = det.get_objects_typed(hailo.HAILO_LANDMARKS)
            if not lmks:
                continue
            if arms_crossed(lmks[0].get_points(), user.kpts):
                arms_event = True
                break  # basta um

    # 3. buffer circular (sempre em BGR) ------------------------------------
    if frame_rgb is not None:
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        user.buffer.append(frame_bgr.copy())
    else:
        frame_bgr = None

    # 4. disparo de gravação -------------------------------------------------
    now = time.time()
    if arms_event and not user.saving and (now - user.last_event_ts > 2):
        user.saving = True
        user.last_event_ts = now
        frames_to_save = list(user.buffer)
        threading.Thread(target=user._save_clip,
                         args=(frames_to_save, w, h),
                         daemon=True).start()
        user.overlay_frames = user.fps             # mostra vermelho 1 s

    # 5. overlay de texto ----------------------------------------------------
    if frame_bgr is not None:
        show_cross = user.overlay_frames > 0
        if user.overlay_frames > 0:
            user.overlay_frames -= 1

        txt   = "BRAÇOS CRUZADOS" if show_cross else "Braços não cruzados"
        color = (0, 0, 255) if show_cross else (0, 255, 0)   # BGR

        # fundo opaco p/ visibilidade
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
        cv2.rectangle(frame_bgr, (5, 10), (5+tw+10, 10+th+10),
                      (0, 0, 0), -1)
        cv2.putText(frame_bgr, txt, (10, 10+th),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA)

        # entrega para visualização
        user.set_frame(frame_bgr)

    return Gst.PadProbeReturn.OK


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    user_data = UserCallback(fps=30, clip_sec=30, out_dir="gravacoes")

    app = GStreamerPoseEstimationApp(app_callback, user_data)
    app.run()
