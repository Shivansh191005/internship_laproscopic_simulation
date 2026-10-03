"""
Free camera + separate zoomable window for the operating-room view.

Press  o  in the console to open the big "Operating room (3D)" window:

    left-drag          orbit around the point you are looking at
    right-drag         pan  (also: middle-drag, or Shift + left-drag)
    mouse wheel        zoom in / out   (double-click: zoom in)
    v / b              next / previous preset angle
    r                  back to the current preset
    o                  close the window

The small panel in the console always shows the same camera.
"""
import numpy as np
import cv2
import pybullet as p

import config as cfg

WINDOW = "Operating room (3D)"


def _clamp_target(t):
    hx, hy, hz = cfg.ROOM_SIZE
    return [float(np.clip(t[0], -hx + 0.2, hx - 0.2)),
            float(np.clip(t[1], -hy + 0.2, hy - 0.2)),
            float(np.clip(t[2], 0.05, 2 * hz - 0.2))]


def _wheel_delta(flags):
    """Wheel direction (+ = away from you). cv2.getMouseWheelDelta is
    missing in some OpenCV builds; the delta is the signed high word."""
    f = getattr(cv2, "getMouseWheelDelta", None)
    if f is not None:
        return f(flags)
    d = (int(flags) >> 16) & 0xFFFF
    return d - 0x10000 if d >= 0x8000 else d


class OrbitCamera:
    """Camera = (distance, yaw, pitch, target), like PyBullet's debug cam."""

    def __init__(self):
        self.preset = 0
        self.free = False
        self.apply_preset(0)

    # -- presets -----------------------------------------------------------
    @property
    def name(self):
        n = cfg.OR_VIEWS[self.preset % len(cfg.OR_VIEWS)][0]
        return f"{n} (free)" if self.free else n

    def apply_preset(self, i):
        self.preset = i % len(cfg.OR_VIEWS)
        _, d, y, pch, t = cfg.OR_VIEWS[self.preset]
        self.dist, self.yaw, self.pitch = float(d), float(y), float(pch)
        self.target = list(map(float, t))
        self.free = False

    def next(self, step=1):
        self.apply_preset(self.preset + step)

    def spec(self):
        return (self.dist, self.yaw, self.pitch, list(self.target))

    # -- mouse moves -------------------------------------------------------
    def orbit(self, dx, dy):
        self.yaw = (self.yaw - dx * 0.30) % 360.0
        self.pitch = float(np.clip(self.pitch - dy * 0.25, -89.0, 15.0))
        self.free = True

    def zoom(self, factor):
        self.dist = float(np.clip(self.dist * factor, 0.25, 7.0))
        self.free = True

    def pan(self, dx, dy):
        V = np.array(p.computeViewMatrixFromYawPitchRoll(
            self.target, self.dist, self.yaw, self.pitch, 0, 2)).reshape(4, 4).T
        right, up = V[0, :3], V[1, :3]
        k = self.dist * 0.0016
        t = np.array(self.target) - right * dx * k + up * dy * k
        self.target = _clamp_target(t)
        self.free = True


class ORWindow:
    """The separate zoomable window and its mouse handling."""

    def __init__(self, cam):
        self.cam = cam
        self.open = False
        self._drag = None
        self._last = (0, 0)
        self.dirty = True              # camera changed -> redraw soon

    def toggle(self):
        if self.open:
            self.close()
        else:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL |
                           getattr(cv2, "WINDOW_GUI_NORMAL", 0))
            w, h = cfg.OR_WINDOW_SIZE
            cv2.resizeWindow(WINDOW, w, h)
            cv2.moveWindow(WINDOW, int(__import__("os").environ.get("ORWIN_X", 60)), 60)
            cv2.setMouseCallback(WINDOW, self._mouse)
            self.open = True
            self.dirty = True
        return self.open

    def close(self):
        if self.open:
            try:
                cv2.destroyWindow(WINDOW)
            except cv2.error:
                pass
        self.open = False

    def closed_by_user(self):
        """True if the user clicked the window's X."""
        if not self.open:
            return False
        try:
            return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            return True

    def show(self, bgr, hint=True):
        if not self.open or bgr is None:
            return
        img = bgr
        if hint:
            img = bgr.copy()
            txt = (f"view: {self.cam.name}   |   drag: orbit   right-drag: "
                   f"pan   wheel: zoom   v/b: angles   r: reset   o: close")
            cv2.rectangle(img, (0, img.shape[0] - 26), (img.shape[1],
                                                         img.shape[0]),
                          (20, 22, 26), -1)
            cv2.putText(img, txt, (10, img.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (215, 220, 225), 1,
                        cv2.LINE_AA)
        cv2.imshow(WINDOW, img)

    def _mouse(self, event, x, y, flags, _param):
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN,
                     cv2.EVENT_MBUTTONDOWN):
            pan = (event != cv2.EVENT_LBUTTONDOWN or
                   flags & cv2.EVENT_FLAG_SHIFTKEY)
            self._drag = "pan" if pan else "orbit"
            self._last = (x, y)
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP,
                       cv2.EVENT_MBUTTONUP):
            self._drag = None
        elif event == cv2.EVENT_MOUSEMOVE and self._drag:
            dx, dy = x - self._last[0], y - self._last[1]
            self._last = (x, y)
            if self._drag == "orbit":
                self.cam.orbit(dx, dy)
            else:
                self.cam.pan(dx, dy)
            self.dirty = True
        elif event in (cv2.EVENT_MOUSEWHEEL, getattr(cv2, "EVENT_MOUSEHWHEEL",
                                                     -99)):
            delta = _wheel_delta(flags)
            self.cam.zoom(0.88 if delta > 0 else 1.0 / 0.88)
            self.dirty = True
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            self.cam.zoom(0.7)
            self.dirty = True
