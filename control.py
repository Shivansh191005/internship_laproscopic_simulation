"""
Controllers.

CameraServo  image-based visual servoing. Drives the instrument mask centroid
             to the image centre using yaw/pitch about the trocar, and uses
             mask area as a depth proxy for insertion. No depth estimation
             required, which is why this fits in a week.

SafetyLayer  virtual fixtures around Ureter / Obturator Nerve / iliac vessels.
             Measures pixel distance from the instrument tip to each critical
             mask and applies a repulsive term to the suction arm's command.
"""
import time

import numpy as np
import cv2

import config as cfg
from perception import aim_point


class CameraServo:
    def __init__(self):
        self.last_error = (0.0, 0.0)
        self.in_view = False
        self._prev_world = None      # instrument world position, previous frame
        self._vel = np.zeros(2)      # smoothed world-frame velocity, px/frame
        self._cmd = np.zeros(2)      # low-passed tilt command

    def _track_velocity(self, obs):
        """Smoothed instrument velocity in the world frame, for feedforward."""
        aim = aim_point(obs)
        if aim is None:
            self._prev_world = None
            self._vel *= 0.8
            return self._vel
        p = np.array(aim, dtype=float)
        if self._prev_world is not None:
            a = cfg.FF_SMOOTH
            self._vel = a * self._vel + (1 - a) * (p - self._prev_world)
        self._prev_world = p
        return self._vel

    def update(self, view, cmd=None):
        """
        Takes a ViewObservation from the virtual endoscope, not the raw frame.
        Servoing on the view is what closes the loop: the command changes the
        crop, which changes the error on the next iteration.
        Returns (command dict, telemetry dict).
        """
        if view is None:
            self.in_view = False
            return {}, {"in_view": False, "err_x": 0.0, "err_y": 0.0,
                        "mode": "idle"}

        # HOLD: the surgeon has asked the scope to stay put. Tracking still
        # runs and the HUD still reports error, but no motion is commanded.
        if cmd is not None and cmd["mode"] == "HOLD":
            self.in_view = view.instrument_visible
            return {}, {"in_view": view.instrument_visible, "mode": "hold",
                        "err_x": 0.0, "err_y": 0.0}

        # REACQUISITION. The instrument is detected in the world frame but has
        # fallen outside the endoscope view. Returning an empty command here
        # deadlocks the arm: no error signal means no motion means it can never
        # find the instrument again. Instead, steer on the world-frame bearing
        # at reduced gain until it re-enters the view, then hand back to IBVS.
        if not view.instrument_visible:
            self.in_view = False
            obs = view.source
            if obs is None or obs.instrument is None:
                return {}, {"in_view": False, "err_x": 0.0, "err_y": 0.0,
                            "mode": "lost"}

            # Error must be measured against the CURRENT CROP CENTRE, not the
            # frame centre. Frame-centre error is independent of where the
            # scope is pointing, so it is an open-loop ramp: the tilt runs to
            # its cone limit and parks there. Crop-centre error shrinks as the
            # scope comes around, so this actually converges.
            x0, y0, x1, y1 = view.crop
            ux, uy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            hw, hh = max((x1 - x0) / 2.0, 1.0), max((y1 - y0) / 2.0, 1.0)
            ix, iy = aim_point(obs)
            ex = float(np.clip((ix - ux) / hw, -2.0, 2.0))
            ey = float(np.clip((iy - uy) / hh, -2.0, 2.0))

            # No sign flag here. The mapping is fixed by geometry and verified:
            # crop centre u increases monotonically with tilt_x, v with tilt_y.
            return ({"tilt_x_rate": cfg.K_REACQUIRE * ex,
                     "tilt_y_rate": cfg.K_REACQUIRE * ey,
                     "insert_rate": (-cfg.MAX_INSERT_RATE
                                     if cfg.ZOOM_ENABLED else 0.0)},
                    {"in_view": False, "err_x": ex, "err_y": ey,
                     "mode": "reacquire"})

        h, w = view.shape[:2]
        cx, cy = view.instrument_centroid

        # Normalised image error, [-1, 1] with 0 at centre.
        ex = (cx - w / 2.0) / (w / 2.0)
        ey = (cy - h / 2.0) / (h / 2.0)
        self.last_error = (ex, ey)
        self.in_view = True

        # Operator bias: a spoken "left"/"right" shifts the target away from
        # dead centre without leaving FOLLOW mode.
        if cmd is not None:
            ex -= cmd["bias"][0]
            ey -= cmd["bias"][1]

        ex_d = 0.0 if abs(ex) < cfg.IBVS_DEADBAND else ex
        ey_d = 0.0 if abs(ey) < cfg.IBVS_DEADBAND else ey

        # Image error maps directly onto the two tilt axes. Because gaze
        # offset is linear in tan(tilt), this is a well-conditioned loop
        # everywhere in the cone. Flip SIGN_X / SIGN_Y in config if it
        # servos the wrong way for your trocar placement.
        tx_rate = cfg.SIGN_X * cfg.K_TILT_X * ex_d
        ty_rate = cfg.SIGN_Y * cfg.K_TILT_Y * ey_d

        # Feedforward. Proportional control alone always trails a moving
        # target; adding the instrument's own velocity lets the scope lead it.
        vx, vy = self._track_velocity(view.source)
        W = view.source.frame.shape[1] if view.source is not None else w
        H = view.source.frame.shape[0] if view.source is not None else h
        ffx = np.clip(cfg.K_FEEDFORWARD * (vx / (W / 2.0)) * 30.0,
                      -cfg.FF_CLAMP, cfg.FF_CLAMP)
        ffy = np.clip(cfg.K_FEEDFORWARD * (vy / (H / 2.0)) * 30.0,
                      -cfg.FF_CLAMP, cfg.FF_CLAMP)
        tx_rate += cfg.SIGN_X * float(ffx)
        ty_rate += cfg.SIGN_Y * float(ffy)

        # Low-pass the outgoing command. Without this every mask flicker
        # reaches the joints directly and the whole rig shivers.
        b = cfg.CMD_SMOOTH
        self._cmd = b * self._cmd + (1 - b) * np.array([tx_rate, ty_rate])
        tx_rate, ty_rate = float(self._cmd[0]), float(self._cmd[1])

        # Mask area in the view as a depth proxy: bigger mask means the tool
        # fills more of the frame, so the endoscope retracts (zooms out).
        target_area = cfg.TARGET_AREA_FRAC + (cmd["zoom_bias"] if cmd else 0.0)
        area_err = target_area - view.instrument_area_frac
        if abs(area_err) < cfg.AREA_DEADBAND:
            area_err = 0.0
        insert_rate = cfg.K_INSERT * area_err if cfg.ZOOM_ENABLED else 0.0

        # Trust a held (occluded) estimate less the longer it has been stale.
        if view.instrument_stale > 0:
            decay = max(0.0, 1.0 - view.instrument_stale /
                        float(cfg.OCCLUSION_HOLD_FRAMES))
            tx_rate *= decay
            ty_rate *= decay
            insert_rate *= decay

        cmd = {"tilt_x_rate": tx_rate, "tilt_y_rate": ty_rate,
               "insert_rate": insert_rate}
        tel = {"in_view": True, "mode": "track", "err_x": ex, "err_y": ey,
               "area_frac": view.instrument_area_frac,
               "stale": view.instrument_stale}
        return cmd, tel


class SafetyLayer:
    """
    A safety system that switches off when it cannot see is worse than none.
    Occlusion holding lives in perception._Smoother; this layer keeps enforcing
    the constraint on held estimates and only flags them as lower confidence.
    """

    def __init__(self):
        self.events = 0
        self.predictive_events = 0
        self.fixture_target = None
        self.fixture_dist = None
        self.min_dist_seen = float("inf")
        self.min_ttc_seen = float("inf")
        self._last_level = "none"
        self._prev = {}          # name -> (distance px, wall time)
        self._closing = {}       # name -> smoothed closing speed, px/s
        self._last_index = -1

    @staticmethod
    def _dist_px(point, contour):
        """Pixel distance from a point to a mask contour. 0 if inside."""
        d = cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])),
                                 True)
        return max(0.0, -d)

    def evaluate(self, obs):
        """Returns (status dict, per-structure distances)."""
        if obs is None or obs.instrument is None:
            return {"level": "none", "closest": None, "dist": None}, {}

        # Check EVERY tool tip, not only the primary one: the surgeon's other
        # hand near the ureter is just as dangerous. Per structure, keep the
        # nearest tool.
        tools = obs.instruments or [obs.instrument]
        dists, by_tool = {}, {}
        for name, det in obs.critical.items():
            if det is None or det.contour is None:
                continue
            for t in tools:
                d_t = self._dist_px(t.work_point, det.contour)
                if name not in dists or d_t < dists[name]:
                    dists[name] = d_t
                    by_tool[name] = t.track_id

        if not dists:
            return {"level": "none", "closest": None, "dist": None}, {}

        closest = min(dists, key=dists.get)
        d = dists[closest]
        self.min_dist_seen = min(self.min_dist_seen, d)

        W = obs.frame.shape[1]
        warn_px = cfg.WARN_DIST_FRAC * W
        danger_px = cfg.DANGER_DIST_FRAC * W

        # Closing speed, updated only when a NEW perception frame arrives. The
        # control loop runs faster than inference; re-reading the same frame
        # would register zero motion and drag the estimate toward zero.
        now = time.perf_counter()
        if obs.index != self._last_index:
            for name, dist in dists.items():
                if name in self._prev:
                    d0, t0 = self._prev[name]
                    v = (d0 - dist) / max(now - t0, 1e-3)   # + means closing
                    self._closing[name] = (cfg.TTC_SMOOTH * self._closing.get(name, 0.0)
                                           + (1 - cfg.TTC_SMOOTH) * v)
                self._prev[name] = (dist, now)
            self._last_index = obs.index

        closing = self._closing.get(closest, 0.0)
        ttc = d / closing if closing > cfg.TTC_MIN_SPEED_PX else float("inf")
        if ttc < float("inf"):
            self.min_ttc_seen = min(self.min_ttc_seen, ttc)

        predictive = False
        if d <= danger_px:
            level = "danger"
        elif d <= warn_px:
            level = "warn"
        elif ttc < cfg.TTC_WARN_S and d <= cfg.PREDICT_RANGE_FRAC * W:
            # Not close yet, but closing fast enough to get there within the
            # warning horizon. Distance-only thresholds react too late to a
            # fast approach; this is the difference between warning a surgeon
            # before contact and warning them at it.
            level = "warn"
            predictive = True
        else:
            level = "clear"

        # Count EVENTS (entries into a state), not frames spent in it.
        if level == "danger" and self._last_level != "danger":
            self.events += 1
        if predictive and self._last_level == "clear":
            self.predictive_events += 1
        self._last_level = level

        return {"level": level, "closest": closest, "dist": d,
                "tool": by_tool.get(closest),
                "warn_px": warn_px, "danger_px": danger_px,
                "ttc": ttc, "closing": closing, "predictive": predictive}, dists

    def filter(self, cmd, obs, suction_state, px_per_m):
        """
        Repulsive virtual fixture on the SUCTION ARM, measured from the
        suction arm's OWN tip.

        The robot can only be responsible for the tool it holds. An earlier
        version took its trigger distance from the surgeon's instrument and
        applied the push to the suction arm - so the robot twitched whenever
        the surgeon neared the ureter, while its own cannula could rest on the
        ureter unnoticed. Now: the surgeon's proximity is an ADVISORY alert
        (evaluate), and this fixture guards the robot's own tip.

        Strength scales from 0 at the warn distance to 1 at danger; the push is
        directed away from the nearest point of the nearest structure.
        """
        self.fixture_target = None
        self.fixture_dist = None
        if obs is None or not obs.critical or px_per_m is None:
            return cmd, 0.0
        H, W = obs.frame.shape[:2]
        gx, gy = suction_state.gaze(cfg.TISSUE_DEPTH)
        sp = np.array([W / 2.0 + px_per_m * gx, H / 2.0 + px_per_m * gy])
        warn_px = cfg.WARN_DIST_FRAC * W
        danger_px = cfg.DANGER_DIST_FRAC * W

        best = None
        for name, det in obs.critical.items():
            if det is None or det.contour is None:
                continue
            pts = det.contour.reshape(-1, 2).astype(float)
            if len(pts) < 2:
                continue
            a, b = pts, np.roll(pts, -1, axis=0)
            ab = b - a
            t = np.clip(np.einsum("ij,ij->i", sp - a, ab)
                        / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-9), 0, 1)
            proj = a + t[:, None] * ab
            dist = np.linalg.norm(sp - proj, axis=1)
            k = int(np.argmin(dist))
            inside = cv2.pointPolygonTest(
                det.contour, (float(sp[0]), float(sp[1])), False) >= 0
            d = 0.0 if inside else float(dist[k])
            away = sp - (pts.mean(axis=0) if inside else proj[k])
            if best is None or d < best[0]:
                best = (d, name, away)
        if best is None:
            return cmd, 0.0

        d, name, away = best
        self.fixture_target, self.fixture_dist = name, d
        strength = float(np.clip((warn_px - d) / max(warn_px - danger_px, 1e-6),
                                 0.0, 1.0))
        if strength <= 0.0:
            return cmd, 0.0
        n = away / max(np.linalg.norm(away), 1e-9)
        out = dict(cmd)
        # +x in the image is +tilt_x (crop centre u rises with tilt_x).
        out["tilt_x_rate"] = out.get("tilt_x_rate", 0.0) + \
            cfg.REPULSION_GAIN * strength * float(n[0])
        out["tilt_y_rate"] = out.get("tilt_y_rate", 0.0) + \
            cfg.REPULSION_GAIN * strength * float(n[1])
        if d <= danger_px:
            out["insert_rate"] = min(out.get("insert_rate", 0.0),
                                     -cfg.MAX_INSERT_RATE * strength)
        return out, strength


class SuctionPilot:
    """
    Suction-irrigator arm.

    A real assistant keeps the suction near the working field but offset from
    the surgeon's instrument, so it is ready without obstructing the view or
    colliding with the working tool. That is what this does: it servos its own
    gaze toward the instrument tip plus a lateral offset.

    The earlier sine-wave motion was placeholder only — it moved the arm but
    meant nothing, and read as the arm behaving randomly.

    The virtual fixture in SafetyLayer.filter still applies on top of this, so
    the arm is pushed away from critical structures regardless of its target.
    """

    def __init__(self):
        self._cmd = np.zeros(2)

    def update(self, dt, cmd=None, obs=None, state=None, px_per_m=None):
        mode = cmd["suction"] if cmd else "OFF"
        if mode == "PARK":
            self._cmd *= 0.5
            return {"tilt_x_rate": 0.0, "tilt_y_rate": 0.0,
                    "insert_rate": -cfg.MAX_INSERT_RATE}
        if mode != "ON":
            self._cmd *= 0.5
            return {"tilt_x_rate": 0.0, "tilt_y_rate": 0.0,
                    "insert_rate": 0.0}
        if obs is None or obs.instrument is None or state is None \
                or px_per_m is None:
            return {"tilt_x_rate": 0.0, "tilt_y_rate": 0.0,
                    "insert_rate": 0.0}

        H, W = obs.frame.shape[:2]
        # Stand off from the instrument tip rather than sitting on it.
        tip = obs.instrument.work_point
        tgt_x = tip[0] + cfg.SUCTION_OFFSET[0] * W
        tgt_y = tip[1] + cfg.SUCTION_OFFSET[1] * H

        # Where this arm is currently aimed, in the same pixel frame.
        gx, gy = state.gaze(cfg.TISSUE_DEPTH)
        ux = W / 2.0 + px_per_m * gx
        uy = H / 2.0 + px_per_m * gy

        ex = float(np.clip((tgt_x - ux) / (W / 2.0), -1.5, 1.5))
        ey = float(np.clip((tgt_y - uy) / (H / 2.0), -1.5, 1.5))
        if abs(ex) < cfg.SUCTION_DEADBAND:
            ex = 0.0
        if abs(ey) < cfg.SUCTION_DEADBAND:
            ey = 0.0

        raw = np.array([cfg.K_SUCTION * ex, cfg.K_SUCTION * ey])
        b = cfg.CMD_SMOOTH
        self._cmd = b * self._cmd + (1 - b) * raw
        return {"tilt_x_rate": float(self._cmd[0]),
                "tilt_y_rate": float(self._cmd[1]),
                "insert_rate": 0.0}


def finger_ik(base, target, axis, side):
    """
    One 2-link finger from `base` (end of the cannula) to `target`.

    Planar 2-link inverse kinematics in the plane that contains the cannula
    axis and the target; the elbow bends OUTWARD (`side` = +1 / -1 picks which
    way), so the two fingers open like a pair of hands rather than crossing.
    Returns (elbow, end, reached). If the target is out of reach the finger
    points straight at it and stops at full length.
    """
    L1, L2 = cfg.FINGER_LINKS
    base, target = np.asarray(base, float), np.asarray(target, float)
    v = target - base
    d = float(np.linalg.norm(v))
    if d < 1e-6:
        return base + axis * L1, base + axis * L1, True
    u = v / d
    # bend direction: perpendicular to u, in the plane of (axis, u)
    perp = np.cross(np.cross(u, axis), u)
    if np.linalg.norm(perp) < 1e-6:
        perp = np.cross(u, [0.0, 0.0, 1.0])
    perp = perp / max(np.linalg.norm(perp), 1e-9) * side
    reached = abs(L1 - L2) <= d <= L1 + L2
    dd = float(np.clip(d, abs(L1 - L2) + 1e-6, L1 + L2 - 1e-6))
    a = (L1 * L1 - L2 * L2 + dd * dd) / (2 * dd)
    h = float(np.sqrt(max(L1 * L1 - a * a, 0.0)))
    elbow = base + u * a + perp * h
    end = base + u * dd
    return elbow, end, reached


class InstrumentArm:
    """
    Arm 2: ONE arm, ONE port, carrying a two-finger head. Both tools seen in
    the video are this arm's end effectors: finger 1 = Tool 1, finger 2 =
    Tool 2.

    The camera (arm 1) is fixed, so each tool tip in the video is a known
    ray; it is turned into a 3D point on the tissue. The arm points its
    cannula at the MIDPOINT of the two tool tips, stopping cfg.FINGER_SPLIT
    above it; from there each finger reaches its own tool tip (finger_ik).
    With one tool in view, that finger reaches it and the other folds back.
    The cannula always stays inside the patient (insertion clamped >= min).

    Commands:  FOLLOW (default) - follow the tools
               HOLD             - stay exactly where it is
    """

    def __init__(self, camera_model):
        self.cam = camera_model
        self.targets = [None, None]     # 3D tip targets for finger 1 / 2
        self.target = None              # cannula end target
        self.mode = "waiting"
        self.base = None
        self.ends = [None, None]
        self.elbows = [None, None]
        self.reached = [False, False]

    def _tool_targets(self, obs):
        out = [None, None]
        tools = (obs.instruments or ([obs.instrument] if obs.instrument
                                     else [])) if obs is not None else []
        for t in tools:
            k = (t.track_id or 1) - 1
            if 0 <= k < 2 and t.stale_frames <= cfg.INSTRUMENT_HOLD_FRAMES:
                px, py = t.work_point
                out[k] = self.cam.pixel_to_tissue(
                    px, py, cfg.TISSUE_Z + cfg.INSTRUMENT_HOVER)
        return out

    def update(self, dt, cmd, obs, state):
        zero = {"tilt_x_rate": 0.0, "tilt_y_rate": 0.0, "insert_rate": 0.0}
        hold = cmd is not None and (cmd.get("instrument") == "HOLD")
        if not hold:
            tg = self._tool_targets(obs)
            if any(t is not None for t in tg):
                self.targets = tg
        if all(t is None for t in self.targets):
            self.mode = "waiting"
            return zero
        live = [t for t in self.targets if t is not None]
        M = np.mean(live, axis=0)
        piv = np.asarray(state.pivot, float)
        up = (piv - M) / max(np.linalg.norm(piv - M), 1e-9)
        S = M + up * cfg.FINGER_SPLIT
        self.target = S
        if hold:
            self.mode = "hold"
            return zero
        from camera_model import rcm_for_tip
        tx, ty, ins = rcm_for_tip(piv, S)
        if np.linalg.norm(state.pose()[0] - S) < cfg.INSTRUMENT_DEADBAND:
            self.mode = "on target"
            return zero
        self.mode = "following"
        k = cfg.K_INSTRUMENT
        return {"tilt_x_rate": k * float(tx - state.tilt_x),
                "tilt_y_rate": k * float(ty - state.tilt_y),
                "insert_rate": k * float(ins - state.insertion)}

    def solve_fingers(self, state):
        """Finger poses for the arm's CURRENT cannula pose."""
        tip, _, d = state.pose()
        self.base = np.asarray(tip, float)
        for i, side in enumerate((+1.0, -1.0)):
            T = self.targets[i]
            if T is None:
                # folded: tucked along the cannula, slightly opened
                rest = self.base + d * 0.6 * sum(cfg.FINGER_LINKS) + \
                    np.cross(d, [0, 0, 1.0]) * 0.01 * side
                self.elbows[i], self.ends[i], _ = finger_ik(self.base, rest,
                                                            d, side)
                self.reached[i] = False
            else:
                e, end, ok = finger_ik(self.base, T, d, side)
                self.elbows[i], self.ends[i], self.reached[i] = e, end, ok
        return self.base, self.ends, self.elbows

    def finger_errors_mm(self):
        """Per finger: distance jaw -> its tool tip (None if no tool)."""
        out = []
        for T, end in zip(self.targets, self.ends):
            out.append(None if T is None or end is None
                       else 1000 * float(np.linalg.norm(end - T)))
        return out

    def tip_pixel(self, state):
        return self.cam.world_to_pixel(state.pose()[0])


def _nearest_structure_px(pt, obs):
    """(distance px, name) from an image point to the nearest critical mask."""
    best = (float("inf"), None)
    for name, det in (obs.critical.items() if obs is not None else []):
        if det is None or det.contour is None:
            continue
        d = cv2.pointPolygonTest(det.contour, (float(pt[0]), float(pt[1])),
                                 True)
        d = max(0.0, -d)
        if d < best[0]:
            best = (d, name)
    return best


def guard_instrument(cmd, state, dt, obs, camera_model, arm_clearance,
                     camera_tip):
    """
    Safety for arm 2. It may keep following the surgeon's tool, but a step is
    REFUSED (the arm simply stops, still inside the patient) when it would:

      * bring its tip closer to a critical structure while already inside the
        danger margin (virtual fixture), or
      * bring it closer to the camera arm while the arms are already closer
        than cfg.ARM_MIN_CLEARANCE.

    The old guard RETRACTED the suction arm instead. It fired constantly, so
    that arm spent its whole time pulling out and never visibly worked.
    Returns (cmd, reason or None).
    """
    from rcm import RCMState
    nxt = RCMState(state.pivot, state.tilt_x, state.tilt_y, state.insertion)
    nxt.integrate(cmd, dt)
    tip0, tip1 = state.pose()[0], nxt.pose()[0]
    zero = {"tilt_x_rate": 0.0, "tilt_y_rate": 0.0, "insert_rate": 0.0}

    if arm_clearance is not None and arm_clearance < cfg.ARM_MIN_CLEARANCE \
            and camera_tip is not None:
        if np.linalg.norm(tip1 - camera_tip) < np.linalg.norm(tip0 - camera_tip):
            return zero, "too close to camera arm"

    if cfg.INSTRUMENT_FIXTURE and obs is not None and obs.critical and \
            camera_model is not None:
        p0 = camera_model.world_to_pixel(tip0)
        p1 = camera_model.world_to_pixel(tip1)
        if p0 is not None and p1 is not None:
            d0, name = _nearest_structure_px(p0, obs)
            d1, _ = _nearest_structure_px(p1, obs)
            danger = cfg.DANGER_DIST_FRAC * obs.frame.shape[1]
            if d1 < danger and d1 < d0:
                return zero, f"stopped at safety margin ({name})"
    return cmd, None
