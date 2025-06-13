#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GStreamer + Hailo Pose-Estimation pipeline que:
  • mantém um buffer circular dos últimos 30 s de vídeo;
  • detecta quando uma pessoa cruza os braços;
  • grava em .mp4 os 30 s *anteriores* ao evento;
  • sobrepõe no frame a mensagem “BRAÇOS CRUZADOS” / “Braços não cruzados”.
Ajuste FPS, resoluções e heurística conforme o seu cenário.
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


# ────────────────────────────────────────────────────────────────────────────
# Utilidades de pose/keypoints
# ────────────────────────────────────────────────────────────────────────────
def coco_keypoints():
    """Mapeia nomes → índice de keypoints seguindo COCO."""
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


def arms_crossed(points, kpts, margin=0.12) -> bool:
    """
    Heurística simples:
      – pulsos abaixo da linha dos ombros
      – x dos pulsos dentro da faixa entre ombro esq./dir. (±margin)
    Ajuste 'margin' ou troque a lógica se necessário.
    """
    ls = points[kpts['left_shoulder']]
    rs = points[kpts['right_shoulder']]
    lw = points[kpts['left_wrist']]
    rw = points[kpts['right_wrist']]

    mid_y = 0.5 * (ls.y() + rs.y())
    if lw.y() > mid_y and rw.y() > mid_y:
        x_min, x_max = min(ls.x(), rs.x()) - margin, max(ls.x(), rs.x()) + margin
        return (x_min < lw.x() < x_max) and (x_min < rw.x() < x_max)
    return False


# ────────────────────────────────────────────────────────────────────────────
# Classe de callback com buffer circular
# ────────────────────────────────────────────────────────────────────────────
class UserCallback(app_callback_class):
    def __init__(self, fps: int = 30, clip_seconds: int = 30, out_dir="gravacoes"):
        super().__init__()
        self.fps = fps
        self.clip_len = fps * clip_seconds
        self.buffer: deque[np.ndarray] = deque(maxlen=self.clip_len)
        self.kpts = coco_keypoints()

        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.clip_idx = 0

        # Controle de disparo
        self.saving = False
        self.last_event_ts = 0.0

        # Controle de overlay (exibir msg por 1 s após evento)
        self.overlay_frames = 0

    # ────────────────────────────────────────────────────────────────────────
    # Gravação assíncrona
    # ────────────────────────────────────────────────────────────────────────
    def _save_clip(self, frames, width: int, height: int):
        ts = time.strftime("%Y%m%d_%H%M%S")
        fname = f"{self.out_dir}/cruzou_{ts}_{self.clip_idx:04}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(fname, fourcc, self.fps, (width, height))
        for f in frames:
            vw.write(f)
        vw.release()
        print(f"[INFO] Clip salvo: {fname}")
        self.clip_idx += 1
        self.saving = False


# ────────────────────────────────────────────────────────────────────────────
# Callback do pad
# ────────────────────────────────────────────────────────────────────────────
def app_callback(pad, info, user: UserCallback):
    buf = info.get_buffer()
    if buf is None:
        return Gst.PadProbeReturn.OK

    user.increment()
    fmt, width, height = get_caps_from_pad(pad)

    # Converte frame para numpy (RGB) se desejar
    frame = None
    if user.use_frame and fmt and width and height:
        frame = get_numpy_from_buffer(buf, fmt, width, height)

    # Inferência Hailo
    roi = hailo.get_roi_from_buffer(buf)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    arms_event = False
    for det in detections:
        if det.get_label() != "person":
            continue

        lmks = det.get_objects_typed(hailo.HAILO_LANDMARKS)
        if not lmks:
            continue
        pts = lmks[0].get_points()

        if arms_crossed(pts, user.kpts):
            arms_event = True

        # Desenho opcional dos pulsos (debug visual)
        if user.use_frame and frame is not None:
            for name in ('left_wrist', 'right_wrist'):
                p = pts[user.kpts[name]]
                x = int((p.x() * det.get_bbox().width() + det.get_bbox().xmin()) * width)
                y = int((p.y() * det.get_bbox().height() + det.get_bbox().ymin()) * height)
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

    # ────────────── Buffer circular ──────────────
    if user.use_frame and frame is not None:
        # Guardamos cópia BGR para gravar direto
        user.buffer.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    # ────────────── Disparo de gravação ───────────
    now = time.time()
    if arms_event and not user.saving and (now - user.last_event_ts > 2):
        user.saving = True
        user.last_event_ts = now
        frames_to_save = list(user.buffer)          # cópia estável
        threading.Thread(target=user._save_clip,
                         args=(frames_to_save, width, height),
                         daemon=True).start()

    # ────────────── Overlay de texto ──────────────
    if user.use_frame and frame is not None:
        if arms_event:
            user.overlay_frames = user.fps          # mostra ~1 s
        show_cross = user.overlay_frames > 0
        if user.overlay_frames > 0:
            user.overlay_frames -= 1

        txt   = "BRAÇOS CRUZADOS" if show_cross else "Braços não cruzados"
        color = (0, 0, 255) if show_cross else (0, 255, 0)  # BGR
        cv2.putText(frame, txt, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA)

        # Envia frame convertido de volta
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user.set_frame(frame_bgr)

    return Gst.PadProbeReturn.OK


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Escolha FPS real se souber; caso contrário use 30 e ajuste
    user_data = UserCallback(fps=30, clip_seconds=30, out_dir="gravacoes")
    user_data.use_frame = True  # Necessário p/ overlay e buffer

    app = GStreamerPoseEstimationApp(app_callback, user_data)
    app.run()
