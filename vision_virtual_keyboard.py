import time
import threading
import queue
from dataclasses import dataclass

import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


# ----------------------------
# Configuration / Tuning
# ----------------------------

CAM_INDEX = 0
CAM_WIDTH = 960
CAM_HEIGHT = 540

UI_UPDATE_MS = 30
CAM_PREVIEW_W = 600
CAM_PREVIEW_H = 380

# Head step (more stable)
NOSE_Y_SMOOTH_ALPHA = 0.30
STEP_THRESHOLD = 0.032     # step trigger (try 0.028..0.040)
RETURN_THRESHOLD = 0.012   # must return inside this to re-arm
STEP_COOLDOWN_SEC = 0.22   # prevents double steps

# Auto scanning
SCAN_INTERVAL_SEC = 0.50
SCAN_PAUSE_AFTER_BLINK_SEC = 0.18

# Blink
EAR_THRESHOLD = 0.20
BLINK_CONSEC_FRAMES = 2
BLINK_COOLDOWN_SEC = 0.40

LEFT_EYE = [33, 160, 159, 133, 145, 144]
RIGHT_EYE = [362, 387, 386, 263, 373, 374]
NOSE_LM = 1


class EMA1D:
    def __init__(self, alpha: float):
        self.alpha = float(alpha)
        self.value = None

    def update(self, x: float) -> float:
        x = float(x)
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1.0 - self.alpha) * self.value
        return float(self.value)


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def ear_from_6_points(p1, p2, p3, p4, p5, p6) -> float:
    num = euclidean(p2, p6) + euclidean(p3, p5)
    den = 2.0 * euclidean(p1, p4)
    if den <= 1e-6:
        return 0.0
    return num / den


@dataclass
class VisionEvent:
    kind: str  # "ROW_STEP", "BLINK", "FRAME"
    data: dict


# ----------------------------
# Improved head step detector (no drifting neutral)
# ----------------------------

class HeadRowStepStable:
    """
    Stable step-based row control:

    - neutral_y set at start
    - if dy crosses STEP_THRESHOLD => step once
    - after step: neutral_y = current y (auto-neutral)
    - disarm until dy returns within RETURN_THRESHOLD
    - neutral does NOT drift (prevents wrong steps)
    """
    def __init__(self):
        self.y_ema = EMA1D(alpha=NOSE_Y_SMOOTH_ALPHA)
        self.neutral_y = None
        self.armed = True
        self.last_step_t = 0.0
        self.last_dy = 0.0

    def update(self, nose_y_norm: float, now: float) -> str:
        y = self.y_ema.update(nose_y_norm)

        if self.neutral_y is None:
            self.neutral_y = y
            self.armed = True
            self.last_dy = 0.0
            return "NONE"

        dy = y - self.neutral_y
        self.last_dy = float(dy)

        # re-arm logic
        if not self.armed:
            if abs(dy) <= RETURN_THRESHOLD:
                self.armed = True
            return "NONE"

        # cooldown
        if (now - self.last_step_t) < STEP_COOLDOWN_SEC:
            return "NONE"

        if dy > STEP_THRESHOLD:
            self.last_step_t = now
            self.armed = False
            self.neutral_y = y  # auto-neutral after step
            return "DOWN"

        if dy < -STEP_THRESHOLD:
            self.last_step_t = now
            self.armed = False
            self.neutral_y = y
            return "UP"

        return "NONE"


class BlinkDetector:
    def __init__(self):
        self.below_count = 0
        self.last_blink_t = 0.0
        self.last_ear = 1.0

    def update(self, left_eye_pts, right_eye_pts, now: float) -> bool:
        left_ear = ear_from_6_points(*left_eye_pts)
        right_ear = ear_from_6_points(*right_eye_pts)
        ear = (left_ear + right_ear) / 2.0
        self.last_ear = float(ear)

        if (now - self.last_blink_t) < BLINK_COOLDOWN_SEC:
            self.below_count = 0
            return False

        if ear < EAR_THRESHOLD:
            self.below_count += 1
        else:
            self.below_count = 0

        if self.below_count >= BLINK_CONSEC_FRAMES:
            self.below_count = 0
            self.last_blink_t = now
            return True

        return False


class KeyboardRenderer:
    def __init__(self, parent: ttk.Frame, text_widget: tk.Text):
        self.text = text_widget
        self.shift = False
        self.caps = True

        self.active_row = 0
        self.scan_col = 0
        self.scan_last_t = time.time()
        self.scan_pause_until = 0.0

        self.keys = [
            ["`","1","2","3","4","5","6","7","8","9","0","-","=","BACK"],
            ["TAB","Q","W","E","R","T","Y","U","I","O","P","[","]","\\"],
            ["CAPS","A","S","D","F","G","H","J","K","L",";","'","ENTER"],
            ["SHIFT","Z","X","C","V","B","N","M",",",".","/","SHIFT"],
            ["CTRL","WIN","ALT","SPACE","ALT","WIN","CTRL"],
        ]

        self.btn_map = {}
        for r, row in enumerate(self.keys):
            row_frame = ttk.Frame(parent)
            row_frame.pack(anchor="center", pady=6)
            for c, key in enumerate(row):
                w = 4
                if key in ("SHIFT", "BACK", "TAB", "CAPS"):
                    w = 7
                if key == "ENTER":
                    w = 10
                if key == "SPACE":
                    w = 20
                if key == "\\":
                    w = 6
                lbl = tk.Label(
                    row_frame, text=key, font=("Segoe UI", 14, "bold"),
                    width=w, relief="ridge", bd=2, padx=6, pady=8,
                    bg="#202225", fg="white"
                )
                lbl.pack(side="left", padx=5)
                self.btn_map[(r, c)] = lbl

        self.update_highlights()

    def set_row_step(self, move: str):
        if move == "UP":
            new_row = max(0, self.active_row - 1)
        elif move == "DOWN":
            new_row = min(len(self.keys) - 1, self.active_row + 1)
        else:
            return

        if new_row != self.active_row:
            self.active_row = new_row
            self.scan_col = 0
            self.scan_last_t = time.time()
            self.update_highlights()

    def tick_scan(self, now: float):
        if now < self.scan_pause_until:
            return
        if (now - self.scan_last_t) >= SCAN_INTERVAL_SEC:
            self.scan_last_t = now
            row_len = len(self.keys[self.active_row])
            if row_len > 0:
                self.scan_col = (self.scan_col + 1) % row_len
                self.update_highlights()

    def update_highlights(self):
        for lbl in self.btn_map.values():
            lbl.configure(bg="#202225", fg="white")
        for c in range(len(self.keys[self.active_row])):
            self.btn_map[(self.active_row, c)].configure(bg="#2E3B4E", fg="white")
        if 0 <= self.scan_col < len(self.keys[self.active_row]):
            self.btn_map[(self.active_row, self.scan_col)].configure(bg="#00D1FF", fg="black")
        if self.shift:
            for (r, c), lbl in self.btn_map.items():
                if lbl.cget("text") == "SHIFT":
                    lbl.configure(bg="#FFD000", fg="black")

    def current_key(self) -> str:
        row = self.keys[self.active_row]
        if not row:
            return ""
        self.scan_col = max(0, min(self.scan_col, len(row) - 1))
        return row[self.scan_col]

    def on_blink_select(self):
        key = self.current_key()
        self.scan_pause_until = time.time() + SCAN_PAUSE_AFTER_BLINK_SEC

        if key == "":
            return
        if key == "SHIFT":
            self.shift = not self.shift
            self.update_highlights()
            return
        if key == "CAPS":
            self.caps = not self.caps
            return
        if key == "BACK":
            current = self.text.get("1.0", "end-1c")
            if current:
                self.text.delete("1.0", "end")
                self.text.insert("1.0", current[:-1])
            return
        if key == "SPACE":
            self.text.insert("end", " ")
            return
        if key == "ENTER":
            self.text.insert("end", "\n")
            return
        if key == "TAB":
            self.text.insert("end", "\t")
            return

        ch = key
        if len(ch) == 1 and ch.isalpha():
            ch = ch.upper() if (self.caps ^ self.shift) else ch.lower()
            self.shift = False
            self.update_highlights()
        self.text.insert("end", ch)


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Hands-free Keyboard: Stable Head Row + Auto Scan + Blink Select")
        self.root.geometry("1320x680")

        self.event_queue = queue.Queue()

        container = ttk.Frame(root, padding=10)
        container.pack(fill=tk.BOTH, expand=True)
        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(0, weight=1)

        left = ttk.Frame(container)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        ttk.Label(left, text="Camera", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.camera_label = tk.Label(left, bg="black")
        self.camera_label.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        self._tk_img_ref = None

        right = ttk.Frame(container)
        right.grid(row=0, column=1, sticky="nsew")

        ttk.Label(right, text="Typed Text", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.text = tk.Text(right, height=4, font=("Consolas", 16), wrap="word")
        self.text.pack(fill=tk.X, pady=(6, 10))

        ttk.Label(right, text="Nod up/down: change row | keys scan | blink select", font=("Segoe UI", 10)).pack(anchor="w")

        kb_frame = ttk.Frame(right)
        kb_frame.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        self.keyboard = KeyboardRenderer(kb_frame, self.text)

        self._poll_events()
        self._scan_tick()

    def push_event(self, ev: VisionEvent):
        self.event_queue.put(ev)

    def _poll_events(self):
        try:
            while True:
                ev = self.event_queue.get_nowait()
                if ev.kind == "ROW_STEP":
                    self.keyboard.set_row_step(ev.data["move"])
                elif ev.kind == "BLINK":
                    self.keyboard.on_blink_select()
                elif ev.kind == "FRAME":
                    self._update_camera(ev.data["frame_bgr"])
        except queue.Empty:
            pass
        self.root.after(UI_UPDATE_MS, self._poll_events)

    def _scan_tick(self):
        self.keyboard.tick_scan(time.time())
        self.root.after(20, self._scan_tick)

    def _update_camera(self, frame_bgr: np.ndarray):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self._tk_img_ref = imgtk
        self.camera_label.configure(image=imgtk)


class VisionThread(threading.Thread):
    def __init__(self, app: App):
        super().__init__(daemon=True)
        self.app = app
        self.stop_flag = threading.Event()

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.head = HeadRowStepStable()
        self.blink = BlinkDetector()

    def stop(self):
        self.stop_flag.set()

    def run(self):
        cap = cv2.VideoCapture(CAM_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        if not cap.isOpened():
            print("ERROR: Could not open webcam.")
            return

        while not self.stop_flag.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            now = time.time()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.face_mesh.process(rgb)

            move = "NONE"
            ear_val = 0.0

            if res.multi_face_landmarks:
                face = res.multi_face_landmarks[0]

                def lm_xy(i):
                    lm = face.landmark[i]
                    return np.array([lm.x * w, lm.y * h], dtype=float)

                nose_y = float(face.landmark[NOSE_LM].y)
                move = self.head.update(nose_y, now)
                if move in ("UP", "DOWN"):
                    self.app.push_event(VisionEvent(kind="ROW_STEP", data={"move": move}))

                lp = [lm_xy(i) for i in LEFT_EYE]
                rp = [lm_xy(i) for i in RIGHT_EYE]
                did_blink = self.blink.update(lp, rp, now)
                ear_val = self.blink.last_ear
                if did_blink:
                    self.app.push_event(VisionEvent(kind="BLINK", data={}))

                nx, ny = int(face.landmark[NOSE_LM].x * w), int(face.landmark[NOSE_LM].y * h)
                cv2.circle(frame, (nx, ny), 4, (0, 255, 255), -1)

            cv2.putText(frame, f"dy={self.head.last_dy:+.3f} armed={self.head.armed} step={move}",
                        (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"thr={STEP_THRESHOLD:.3f} return={RETURN_THRESHOLD:.3f} cooldown={STEP_COOLDOWN_SEC:.2f}s",
                        (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"EAR={ear_val:.3f} thr={EAR_THRESHOLD:.2f} frames={BLINK_CONSEC_FRAMES}",
                        (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            frame_ui = cv2.resize(frame, (CAM_PREVIEW_W, CAM_PREVIEW_H))
            self.app.push_event(VisionEvent(kind="FRAME", data={"frame_bgr": frame_ui}))

        cap.release()
        self.face_mesh.close()


def main():
    root = tk.Tk()
    app = App(root)

    vision = VisionThread(app)
    vision.start()

    def on_close():
        vision.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()