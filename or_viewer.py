"""
Big zoomable 3D window of the operating room (key  o  or --or-window).

Runs as its OWN process with a real on-screen OpenGL window (pyglet, via
pyrender's Viewer), so rotating / zooming never waits for the console:

  * the picture goes straight to the screen - no read-back of pixels from
    the GPU, no copying images between processes;
  * the mouse is handled by the window itself at its own frame rate, so the
    camera answers immediately even while YOLO and the console are busy;
  * 4x hardware anti-aliasing (MSAA) instead of rendering twice as big;
  * the main program only sends the arm poses (~6 KB) ~30 times a second.

Controls (in the 3D window):
    left-drag            orbit (turntable: the floor stays level)
    right-drag / Shift   pan
    mouse wheel          zoom (smooth)
    v / b                next / previous preset angle (glides there)
    r                    back to the current preset
    Esc / close button   close the window
"""
import os
import sys
import time

import numpy as np

import config as cfg

_PAN_K = 0.0016


def _preset(i):
    _, d, y, p_, t = cfg.OR_VIEWS[i % len(cfg.OR_VIEWS)]
    return float(d), float(y), float(p_), np.array(t, float)


def _viewer_main(conn, preset):
    """Child process: build the scene once, then run the window."""
    try:
        if sys.platform.startswith("linux") and \
                "PYOPENGL_PLATFORM" in os.environ and \
                os.environ.get("DISPLAY"):
            os.environ.pop("PYOPENGL_PLATFORM")      # on-screen, not EGL
        import pyrender
        import pybullet as p
        import pyglet
        from scene import Scene
        from or_render import ORRenderer, _quat_to_mat
        sc = Scene(gui=False)
        r = ORRenderer(sc, offscreen=False)
        keys = sorted({(b, l) for _, b, l, _ in r.nodes})
        idx = {k: i for i, k in enumerate(keys)}
    except Exception as e:                                  # noqa: BLE001
        import traceback
        conn.send(("error", f"{type(e).__name__}: {e}",
                   traceback.format_exc()))
        return

    def view_matrix(d, yaw, pitch, tgt):
        V = np.array(p.computeViewMatrixFromYawPitchRoll(
            list(tgt), d, yaw, pitch, 0, 2)).reshape(4, 4).T
        return np.linalg.inv(V)

    class LiveViewer(pyrender.Viewer):
        def __init__(self):
            self.cur = list(_preset(preset))      # shown camera
            self.goal = list(_preset(preset))     # where it is gliding to
            self.preset = preset
            self.drag_mode = None
            self.last_t = time.perf_counter()
            self.frames = 0
            self.fps_t = time.perf_counter()
            self.fps = 0.0
            r.scene.set_pose(r.cam_node, view_matrix(*self.cur))
            flags = {"shadows": False, "cull_faces": True}
            super().__init__(
                r.scene, viewport_size=tuple(cfg.OR_WINDOW_SIZE),
                render_flags=flags,
                viewer_flags={"refresh_rate": float(cfg.OR_WINDOW_FPS),
                              "window_title": "Operating room (3D)",
                              "use_raymond_lighting": False,
                              "use_direct_lighting": False})

        # -- camera ------------------------------------------------------
        def _reset_view(self):
            from pyrender.trackball import Trackball
            pose = view_matrix(*self.cur)
            self._camera_node.matrix = pose
            self._trackball = Trackball(pose, self.viewport_size, 1.0,
                                        self.cur[3])

        def _title(self):
            name = cfg.OR_VIEWS[self.preset % len(cfg.OR_VIEWS)][0]
            self.set_caption(
                f"Operating room (3D) - {name} - {self.fps:.0f} fps   |   "
                f"drag: orbit   right-drag: pan   wheel: zoom   "
                f"v/b: angles   r: reset   Esc: close")

        def go_preset(self, i):
            self.preset = i % len(cfg.OR_VIEWS)
            self.goal = list(_preset(self.preset))
            self._title()

        def on_mouse_press(self, x, y, buttons, modifiers):
            shift = modifiers & pyglet.window.key.MOD_SHIFT
            self.drag_mode = "pan" if (buttons != pyglet.window.mouse.LEFT
                                       or shift) else "orbit"

        def on_mouse_release(self, x, y, button, modifiers):
            self.drag_mode = None

        def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
            g = self.goal
            if self.drag_mode == "orbit":
                g[1] = g[1] - dx * 0.30
                g[2] = float(np.clip(g[2] + dy * 0.25, -89.0, 15.0))
                self.cur[1], self.cur[2] = g[1], g[2]   # direct, no lag
            else:
                M = view_matrix(*self.cur)
                right, up = M[:3, 0], M[:3, 1]
                k = g[0] * _PAN_K
                g[3] = g[3] - right * dx * k - up * dy * k
                hx, hy, hz = cfg.ROOM_SIZE
                g[3] = np.clip(g[3], [-hx + .2, -hy + .2, .05],
                               [hx - .2, hy - .2, 2 * hz - .2])
                self.cur[3] = g[3].copy()

        def on_mouse_scroll(self, x, y, dx, dy):
            self.goal[0] = float(np.clip(self.goal[0] * (0.88 ** dy),
                                         0.25, 7.0))

        def on_key_press(self, symbol, modifiers):
            K = pyglet.window.key
            if symbol == K.V:
                self.go_preset(self.preset + 1)
            elif symbol == K.B:
                self.go_preset(self.preset - 1)
            elif symbol == K.R:
                self.go_preset(self.preset)
            elif symbol == K.ESCAPE:
                self.close()

        def on_resize(self, width, height):
            super().on_resize(width, height)
            r.cam_node.camera.aspectRatio = width / max(height, 1)
            self._camera_node.camera.aspectRatio = width / max(height, 1)

        # -- live data from the simulation ---------------------------------
        def _pump(self):
            got = None
            try:
                while conn.poll():
                    msg = conn.recv()
                    if msg is None:
                        self.close()
                        return
                    got = msg
            except (EOFError, OSError):
                self.close()
                return
            if got is None:
                return
            Ws, troc, frame = got
            for node, b, l, local in r.nodes:
                r.scene.set_pose(node, Ws[idx[(b, l)]] @ local)
            for arm, (n1, n2) in getattr(r, "_trocars", {}).items():
                T = troc.get(arm)
                if T is not None:
                    r.scene.set_pose(n1, T)
                    r.scene.set_pose(n2, T)
            r._pose_twin()
            if frame is not None:
                r._screen_t = 0.0
                r.set_screen_image(frame)
                r._update_screens()
            try:
                conn.send("ack")
            except (EOFError, OSError):
                pass

        def _render(self):
            self._pump()
            # glide the camera towards its goal (smooth zoom / presets)
            now = time.perf_counter()
            a = 1.0 - np.exp(-(now - self.last_t) * 12.0)
            self.last_t = now
            c, g = self.cur, self.goal
            c[0] += (g[0] - c[0]) * a
            dyaw = (g[1] - c[1] + 180.0) % 360.0 - 180.0
            c[1] += dyaw * a
            c[2] += (g[2] - c[2]) * a
            c[3] = c[3] + (g[3] - c[3]) * a
            self._camera_node.matrix = view_matrix(*c)
            from pyrender.constants import RenderFlags
            flags = RenderFlags.NONE
            if r.q["shadows"]:
                flags |= RenderFlags.SHADOWS_SPOT
            self._renderer.render(self.scene, flags)
            self.frames += 1
            if now - self.fps_t > 1.0:
                self.fps = self.frames / (now - self.fps_t)
                self.frames, self.fps_t = 0, now
                self._title()

    conn.send(("ready", keys))
    try:
        LiveViewer()                     # blocks until the window closes
    except Exception as e:                                  # noqa: BLE001
        import traceback
        try:
            conn.send(("error", f"{type(e).__name__}: {e}",
                       traceback.format_exc()))
        except Exception:                                   # noqa: BLE001
            pass


class ORWindow3D:
    """Main-program side: start/stop the window and feed it poses."""

    def __init__(self):
        self._proc = None
        self._conn = None
        self._keys = None
        self._inflight = 0
        self._last = 0.0
        self._ft = 0.0
        self.error = None

    @property
    def open(self):
        return self._proc is not None and self._proc.is_alive()

    def start(self, preset=0, timeout=90.0):
        import multiprocessing as mp
        if self.open:
            return True
        ctx = mp.get_context("spawn")
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(target=_viewer_main, args=(child, preset),
                                 name="or-window", daemon=True)
        self._proc.start()
        t0 = time.perf_counter()
        while not self._conn.poll(0.2):
            if not self._proc.is_alive() or \
                    time.perf_counter() - t0 > timeout:
                self.stop()
                self.error = "3D window did not start"
                return False
        msg = self._conn.recv()
        if msg[0] != "ready":
            self.error = msg[1]
            print(f"[or-window] could not open: {msg[1]}")
            if len(msg) > 2:
                print(msg[2])
            self.stop()
            return False
        self._keys = msg[1]
        self._inflight = 0
        print("[or-window] 3D window open: drag = orbit, right-drag = pan, "
              "wheel = zoom, v/b = angles, r = reset, Esc = close")
        return True

    def stop(self):
        if self._proc is not None:
            try:
                self._conn.send(None)
            except Exception:                               # noqa: BLE001
                pass
            self._proc.join(timeout=2.0)
            if self._proc.is_alive():
                self._proc.terminate()
        self._proc = None

    def toggle(self, preset=0):
        if self.open:
            self.stop()
            return False
        return self.start(preset)

    def push(self, sim, frame=None):
        """Send the current arm poses (never blocks; ~30 Hz)."""
        if not self.open:
            return
        try:
            while self._conn.poll():
                if self._conn.recv() == "ack":
                    self._inflight = max(0, self._inflight - 1)
        except (EOFError, OSError):
            return
        now = time.perf_counter()
        if self._inflight >= 2 or now - self._last < 1.0 / 30.0:
            return
        self._last = now
        from or_render import world_poses
        Ws, troc = world_poses(sim, self._keys)
        if frame is not None and now - self._ft >= 0.33:
            from or_render import small_pair
            self._ft = now
            frame = small_pair(frame)
        else:
            frame = None
        try:
            self._conn.send((Ws, troc, frame))
            self._inflight += 1
        except (EOFError, OSError):
            pass
