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


# ---------------------------------------------------------------------------------
# Utilidades ----------------------------------------------------------------------
# ---------------------------------------------------------------------------------
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


def arms_crossed(points, kpts, th=0.12):
    """Heurística simples: pulsos passam à frente da linha
    tórax-ombro-tórax (aproximada). Ajuste th conforme as imagens."""
    ls = points[kpts['left_shoulder']]
    rs = points[kpts['right_shoulder']]
    lw = points[kpts['left_wrist']]
    rw = points[kpts['right_wrist']]

    # linha horizontal aproximada na metade do tórax
    mid_y = 0.5 * (ls.y() + rs.y())
    # pulsos abaixo da linha do ombro (evita falsos positivos quando mãos levantadas)
    if lw.y() > mid_y and rw.y() > mid_y:
        # pulsos à frente do tórax (x entre ombros, com folga th)
        x_min, x_max = min(ls.x(), rs.x()) - th, max(ls.x(), rs.x()) + th
        return (x_min < lw.x() < x_max) and (x_min < rw.x() < x_max)
    return False


# ---------------------------------------------------------------------------------
# Classe estendida que guarda o buffer circular -----------------------------------
# ---------------------------------------------------------------------------------
class UserCallback(app_callback_class):
    def __init__(self, fps=30, clip_sec=30, out_dir="gravacoes"):
        super().__init__()
        self.fps = fps
        self.clip_len = int(fps * clip_sec)
        self.buffer = deque(maxlen=self.clip_len)        # NEW
        self.kpts = coco_keypoints()
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.clip_idx = 0
        self.saving = False            # trava para não gravar 2× o mesmo evento
        self.last_event_ts = 0

    # Thread-seguro: dispara a gravação num thread separado -----------------------
    def _save_clip(self, frames, w, h):
        ts = time.strftime("%Y%m%d_%H%M%S")
        fname = f"{self.out_dir}/cruzou_{ts}_{self.clip_idx:04}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(fname, fourcc, self.fps, (w, h))
        for f in frames:
            vw.write(f)
        vw.release()
        print(f"[INFO] Clip salvo: {fname}")
        self.clip_idx += 1
        self.saving = False

# ---------------------------------------------------------------------------------
# Callback principal --------------------------------------------------------------
# ---------------------------------------------------------------------------------
def app_callback(pad, info, user: UserCallback):
    buf = info.get_buffer()
    if buf is None:
        return Gst.PadProbeReturn.OK

    user.increment()
    fmt, w, h = get_caps_from_pad(pad)
    frame = None
    if user.use_frame and fmt and w and h:
        frame = get_numpy_from_buffer(buf, fmt, w, h)

    roi = hailo.get_roi_from_buffer(buf)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    arms_event = False
    for det in detections:
        if det.get_label() != "person":
            continue

        lmks = det.get_objects_typed(hailo.HAILO_LANDMARKS)
        if not lmks:
            continue
        points = lmks[0].get_points()

        if arms_crossed(points, user.kpts):
            arms_event = True

        if user.use_frame and frame is not None:
            # desenha pulsos (debug visual)
            for name in ('left_wrist', 'right_wrist'):
                p = points[user.kpts[name]]
                x = int((p.x() * det.get_bbox().width() + det.get_bbox().xmin()) * w)
                y = int((p.y() * det.get_bbox().height() + det.get_bbox().ymin()) * h)
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

    # NEW ― mantém buffer circular com cópia BGR da imagem ------------------------
    if user.use_frame and frame is not None:
        user.buffer.append(frame.copy())

    # Se braços cruzados → grava os 30 s anteriores --------------------------------
    if arms_event and not user.saving:
        now = time.time()
        if now - user.last_event_ts > 2:           # 2 s de histerese
            user.saving = True
            user.last_event_ts = now
            frames_to_save = list(user.buffer)     # cópia estável
            threading.Thread(target=user._save_clip,
                             args=(frames_to_save, w, h),
                             daemon=True).start()

    # Exibe / repassa frame se necessário
    if user.use_frame and frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user.set_frame(frame)

    return Gst.PadProbeReturn.OK


# ---------------------------------------------------------------------------------
# Main ----------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
if __name__ == "__main__":
    # escolha o FPS real do pipeline (ou extraia dos caps se preferir)
    user_data = UserCallback(fps=30, clip_sec=30)
    # Defina use_frame=True para ter acesso aos frames RGB no callback
    user_data.use_frame = True

    app = GStreamerPoseEstimationApp(app_callback, user_data)
    app.run()
