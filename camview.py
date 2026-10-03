"""
Voice-controlled camera arm (zoom / pan / centre / follow / home).

What the doctor gets: the same thing a human camera assistant does - "zoom
in", "left", "follow the tool", "hold", "back to start".

How it works with a RECORDED video: the camera cannot really see new parts
of the abdomen, so the monitors show a zoomed / shifted window of the full
recorded frame (digital zoom and pan). At the same time the camera arm in the
simulation really moves: it tilts to aim at the centre of that window and
goes deeper when zoomed in, through its trocar (RCM), so the robot side is
real kinematics.

What does NOT change: YOLO, the safety layer and arm 2 keep working on the
FULL frame, so a zoomed view can never hide a danger warning.
"""
import numpy as np

import config as cfg
from rcm import RCMState


class CameraVoice:
    ZOOM_MIN, ZOOM_MAX, ZOOM_STEP = 1.0, 3.0, 1.25
    # Scope light, as brightness of the monitor picture (100 % = recorded).
    # The AI always sees the original video, so this never changes YOLO.
    LIGHT_MIN, LIGHT_MAX, LIGHT_STEP = 0.5, 1.6, 0.15

    def __init__(self, camera_model, home_state):
        self.cam = camera_model                 # pixel <-> tissue (fixed)
        self.home = home_state                  # recorded camera pose
        self.W, self.H = camera_model.W, camera_model.H
        self.light = 1.0
        self.home_view()
        # what the monitors show right now (smoothed towards the goal)
        self.shown = [self.zoom, self.cx, self.cy]

    # ------------------------------------------------------------ commands
    def home_view(self):
        self.zoom, self.cx, self.cy = 1.0, 0.5, 0.5
        self.follow = False
        self.msg = "home view"

    def command(self, c, obs=None):
        """Apply one command. Returns True if it was a camera command."""
        if c == "ZOOM_IN":
            self.zoom = min(self.ZOOM_MAX, self.zoom * self.ZOOM_STEP)
            self.msg = f"zoom {self.zoom:.1f}x"
        elif c == "ZOOM_OUT":
            self.zoom = max(self.ZOOM_MIN, self.zoom / self.ZOOM_STEP)
            self.msg = f"zoom {self.zoom:.1f}x"
        elif c in ("PAN_LEFT", "PAN_RIGHT", "PAN_UP", "PAN_DOWN"):
            if self.zoom < self.ZOOM_STEP:
                # at 1x the whole recorded field is already on screen; a
                # small zoom gives the view room to move
                self.zoom = self.ZOOM_STEP
            step = 0.15 / self.zoom              # 15 % of the visible width
            dx = {"PAN_LEFT": -step, "PAN_RIGHT": step}.get(c, 0.0)
            dy = {"PAN_UP": -step, "PAN_DOWN": step}.get(c, 0.0)
            self.cx += dx
            self.cy += dy
            self.follow = False
            self.msg = c.split("_")[1].lower()
        elif c == "CENTER":
            p = self._tool_px(obs)
            if p is not None:
                self.cx, self.cy = p[0] / self.W, p[1] / self.H
                self.msg = "centred on tool"
            else:
                self.cx, self.cy = 0.5, 0.5
                self.msg = "centred (no tool seen)"
        elif c == "CAM_FOLLOW":
            self.follow = True
            if self.zoom < self.ZOOM_STEP:
                self.zoom = self.ZOOM_STEP
            self.msg = "following the tool"
        elif c in ("CAM_HOLD", "STOP"):
            self.follow = False
            self.msg = "holding"
        elif c == "HOME":
            self.home_view()
        elif c in ("LIGHT_UP", "LIGHT_DOWN"):
            d = self.LIGHT_STEP if c == "LIGHT_UP" else -self.LIGHT_STEP
            self.light = float(np.clip(round(self.light + d, 2),
                                       self.LIGHT_MIN, self.LIGHT_MAX))
            self.msg = f"light {self.light * 100:.0f} %"
        else:
            return False
        self._clamp()
        return True

    # ------------------------------------------------------------- update
    @staticmethod
    def _tool_px(obs):
        t = getattr(obs, "instrument", None) if obs is not None else None
        if t is None:
            return None
        if t.tip is not None and getattr(t, "tip_visible", True):
            return t.tip
        return t.centroid

    def _clamp(self):
        half = 0.5 / self.zoom
        self.cx = float(np.clip(self.cx, half, 1.0 - half))
        self.cy = float(np.clip(self.cy, half, 1.0 - half))

    def update(self, dt, obs):
        """Follow the tool (if asked) and glide the shown view smoothly."""
        if self.follow:
            p = self._tool_px(obs)
            if p is not None:
                a = min(1.0, dt * 2.0)               # gentle, no jitter
                self.cx += (p[0] / self.W - self.cx) * a
                self.cy += (p[1] / self.H - self.cy) * a
                self._clamp()
        a = min(1.0, dt * 6.0)
        goal = (self.zoom, self.cx, self.cy)
        self.shown = [s + (g - s) * a for s, g in zip(self.shown, goal)]

    def crop(self):
        """Visible window as fractions of the full frame (x0, y0, x1, y1)."""
        z, cx, cy = self.shown
        half = 0.5 / max(z, 1.0)
        cx = float(np.clip(cx, half, 1 - half))
        cy = float(np.clip(cy, half, 1 - half))
        return cx - half, cy - half, cx + half, cy + half

    def is_home(self):
        return abs(self.shown[0] - 1.0) < 1e-3

    def status(self):
        z = self.shown[0]
        mode = "FOLLOW" if self.follow else ("HOME" if self.zoom == 1.0 and
                                             not self.follow else "HOLD")
        light = (f", light {self.light * 100:.0f}%"
                 if abs(self.light - 1.0) > 1e-3 and
                 not self.msg.startswith("light") else "")
        return mode, f"{self.msg} - zoom {z:.1f}x{light}"

    # --------------------------------------------------- camera arm pose
    def arm_target(self):
        """RCM state of the camera arm for the current view: aim at the
        view centre on the tissue, go deeper the more it is zoomed in."""
        z, cx, cy = self.shown
        P = self.cam.pixel_to_tissue(cx * self.W, cy * self.H)
        if P is None:
            return self.home
        v = np.asarray(P, float) - np.asarray(cfg.TROCAR_CAMERA, float)
        tx = float(np.arctan2(v[0], -v[2]))
        ty = float(np.arctan2(v[1], -v[2]))
        lo, hi = cfg.INSERTION_LIMITS
        ins = float(np.clip(self.home.insertion + 0.025 * (z - 1.0), lo,
                            min(hi, self.home.insertion + 0.05)))
        return RCMState(cfg.TROCAR_CAMERA, tilt_x=tx, tilt_y=ty,
                        insertion=ins)

    def drive_arm(self, state, dt):
        """Move the camera arm state towards the target at a safe speed."""
        tgt = self.arm_target()
        max_tilt = 0.35 * dt                     # rad per step
        max_ins = 0.03 * dt                      # m per step
        state.tilt_x += float(np.clip(tgt.tilt_x - state.tilt_x,
                                      -max_tilt, max_tilt))
        state.tilt_y += float(np.clip(tgt.tilt_y - state.tilt_y,
                                      -max_tilt, max_tilt))
        state.insertion += float(np.clip(tgt.insertion - state.insertion,
                                         -max_ins, max_ins))
        return state
