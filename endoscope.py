"""
Virtual endoscope — the piece that makes replay-based servoing legitimate.

The problem: if you feed a recorded video straight to the controller, moving the
camera arm changes nothing about what the camera sees. The loop is open, the
error never converges, and "instrument in view %" is decided by the cameraman
who shot the video rather than by your robot. Any tracking metric you report
would be meaningless.

The fix: treat the full recorded frame as the WORLD, and render the endoscope
view as a steerable crop of it. The camera arm's RCM state decides where the
crop sits and how tight it is:

    yaw, pitch  ->  where the scope is aimed on the tissue plane -> crop centre
    insertion   ->  how close the scope is                       -> crop zoom

Now moving the arm genuinely changes the image, the error converges, and
"instrument in view" is a real measurement of your controller.

Run YOLO once on the full frame, then map the detections into view coordinates
here. One inference serves both the world model and the endoscope view.
"""
import cv2
import numpy as np

import config as cfg


class VirtualEndoscope:
    def __init__(self, frame_w, frame_h):
        self.W, self.H = frame_w, frame_h
        # The view MUST share the video's aspect ratio. A 5:4 view on 16:9
        # footage made the crop 1014 px tall in a 1080 px frame, leaving only
        # +/-33 px of vertical travel: the scope could barely tilt up or down,
        # so any instrument off the horizontal centreline carried an error the
        # servo could never remove.
        self.out_w = cfg.VIEW_SIZE[0]
        self.out_h = max(1, int(round(self.out_w * frame_h / frame_w)))

        # Pan scale MUST be derived from the actual video resolution. A
        # hardcoded px-per-metre silently collapses the reachable pan range on
        # a large clip: the crop clamps to the frame edge and the scope freezes.
        max_gaze_m = cfg.TISSUE_DEPTH * np.tan(cfg.MAX_TILT)
        pan_px = cfg.PAN_FRAC * self.W
        self.px_per_m = pan_px / max(max_gaze_m, 1e-6)
        print(f"[scope] pan +/-{pan_px:.0f} px over {self.W} px frame "
              f"({self.px_per_m:.0f} px/m)")

    # ------------------------------------------------------------------ optics
    def view_params(self, state):
        """Returns (centre_u, centre_v, half_w, half_h) in full-frame pixels."""
        if cfg.CAMERA_MODE == "stable":
            # The camera arm is fixed: the endoscope sees the whole recorded
            # frame, always.
            return self.W / 2.0, self.H / 2.0, self.W / 2.0, self.H / 2.0
        # Offset of the scope's gaze on the tissue plane, in metres.
        # Linear in tan(tilt), so image error maps cleanly onto tilt rates.
        gx, gy = state.gaze(cfg.TISSUE_DEPTH)
        u = self.W / 2.0 + self.px_per_m * gx
        v = self.H / 2.0 + self.px_per_m * gy

        scale = self._scale(state)

        half_w = 0.5 * scale * self.W
        half_h = half_w * (self.out_h / self.out_w)

        # No clamping. Clamping the crop inside the frame forces a choice
        # between a wide view and a usable pan range: at ZOOM_WIDE=0.9 the
        # centre could only travel 864..1056 px in a 1920 px frame. Instead the
        # crop may overhang the frame and is padded black on render, the way a
        # real endoscope shows vignette at the edge of its field.
        return u, v, half_w, half_h

    # ------------------------------------------------------------------ render
    def _scale(self, state):
        """Crop width as a fraction of the frame."""
        if not cfg.ZOOM_ENABLED:
            return cfg.FIXED_VIEW_FRAC
        lo, hi = cfg.INSERTION_LIMITS
        f = (state.insertion - lo) / max(hi - lo, 1e-6)
        return cfg.ZOOM_WIDE + (cfg.ZOOM_TIGHT - cfg.ZOOM_WIDE) * f

    def clamp_state(self, state):
        if cfg.CAMERA_MODE == "stable":
            return state
        """
        Limit the scope's tilt so the view keeps most of the frame in it.

        This clamps the RCM STATE, not just the rendered crop. Clamping only at
        render time lets the servo keep integrating tilt against a wall it
        cannot observe — classic integral windup — and the scope then takes a
        long time to come back when the target returns. Clamping the state
        means the controller always knows where the limit is.
        """
        half_w = 0.5 * self._scale(state) * self.W
        half_h = half_w * (self.out_h / self.out_w)

        ov_w, ov_h = cfg.MAX_OVERHANG * 2 * half_w, cfg.MAX_OVERHANG * 2 * half_h
        u_min, u_max = half_w - ov_w, self.W - half_w + ov_w
        v_min, v_max = half_h - ov_h, self.H - half_h + ov_h
        if u_min > u_max:
            u_min = u_max = self.W / 2.0
        if v_min > v_max:
            v_min = v_max = self.H / 2.0

        k = self.px_per_m * cfg.TISSUE_DEPTH
        state.tilt_x = float(np.clip(state.tilt_x,
                                     np.arctan((u_min - self.W / 2.0) / k),
                                     np.arctan((u_max - self.W / 2.0) / k)))
        state.tilt_y = float(np.clip(state.tilt_y,
                                     np.arctan((v_min - self.H / 2.0) / k),
                                     np.arctan((v_max - self.H / 2.0) / k)))
        return state

    def project(self, obs, state):
        """
        Returns a ViewObservation: the cropped endoscope image plus the
        instrument and critical structures expressed in view coordinates.
        """
        u, v, hw, hh = self.view_params(state)
        x0, y0 = int(round(u - hw)), int(round(v - hh))
        x1, y1 = int(round(u + hw)), int(round(v + hh))

        # Pad rather than clamp: the crop may legitimately overhang the frame.
        cw, ch = max(x1 - x0, 1), max(y1 - y0, 1)
        canvas = np.zeros((ch, cw, 3), np.uint8)
        sx0, sy0 = max(x0, 0), max(y0, 0)
        sx1, sy1 = min(x1, obs.frame.shape[1]), min(y1, obs.frame.shape[0])
        if sx1 > sx0 and sy1 > sy0:
            canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = \
                obs.frame[sy0:sy1, sx0:sx1]
        view_img = cv2.resize(canvas, (self.out_w, self.out_h))

        sx = self.out_w / max(x1 - x0, 1)
        sy = self.out_h / max(y1 - y0, 1)

        def to_view(pt):
            return ((pt[0] - x0) * sx, (pt[1] - y0) * sy)

        def inside(p):
            return 0 <= p[0] < self.out_w and 0 <= p[1] < self.out_h

        vo = ViewObservation(image=view_img, source=obs,
                             crop=(x0, y0, x1, y1), scale=(sx, sy))

        from perception import aim_point
        aim = aim_point(obs)
        if obs.instrument is not None and aim is not None:
            # The servo centres the AIM point: the tool tip (the working end),
            # or the midpoint of both tool tips when two are in view.
            p = to_view(aim)
            vo.instrument_visible = inside(p)
            vo.instrument_centroid = p
            # Mask area grows with zoom: area scales with the linear factors.
            vo.instrument_area_frac = float(
                np.clip(obs.instrument.area_frac * sx * sy, 0.0, 1.0))
            vo.instrument_stale = obs.instrument.stale_frames

        for t in (obs.instruments or ([obs.instrument] if obs.instrument
                                      else [])):
            vo.instruments_view.append(
                (t.track_id, to_view(t.centroid),
                 to_view(t.tip) if t.tip and t.tip_visible else None,
                 t is obs.instrument,
                 to_view(t.back) if t.back else None))

        for name, det in obs.critical.items():
            if det is None or det.contour is None:
                continue
            c = det.contour.reshape(-1, 2).astype(np.float32)
            c[:, 0] = (c[:, 0] - x0) * sx
            c[:, 1] = (c[:, 1] - y0) * sy
            vo.critical_view[name] = c.astype(np.int32).reshape(-1, 1, 2)

        return vo

    def draw_frustum(self, world_img, state, colour=(90, 200, 255)):
        """Overlay the current endoscope footprint on the world frame."""
        u, v, hw, hh = self.view_params(state)
        img = world_img.copy()
        cv2.rectangle(img, (int(u - hw), int(v - hh)),
                      (int(u + hw), int(v + hh)), colour, 2)
        cv2.drawMarker(img, (int(u), int(v)), colour, cv2.MARKER_CROSS, 18, 1)
        return img


class ViewObservation:
    def __init__(self, image, source, crop, scale):
        self.image = image
        self.source = source                 # the underlying full-frame obs
        self.crop = crop
        self.scale = scale
        self.instrument_visible = False
        self.instrument_centroid = None
        self.instrument_area_frac = 0.0
        self.instrument_stale = 0
        self.instruments_view = []   # (id, centroid, tip, primary, back)
        self.critical_view = {}

    @property
    def shape(self):
        return self.image.shape
