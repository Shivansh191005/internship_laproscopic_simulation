"""
Remote Centre of Motion kinematics.

The key idea: don't ask an IK solver to *respect* the trocar constraint as a
null-space task. Instead, parameterise the instrument BY the trocar, then place
the arm to match. The pivot becomes an input rather than something the solver
can drift away from, so it is exact by construction.

An instrument through a trocar has exactly 4 DoF:
    yaw       rotation about the world Z axis
    pitch     tilt away from straight-down (0 = pointing into the pelvis)
    insertion depth of the tip below the pivot
    roll      spin about the shaft (not used by the controllers here)
"""
import numpy as np

import config as cfg


def direction(tilt_x: float, tilt_y: float) -> np.ndarray:
    """
    Unit vector pointing from the trocar INTO the body.

    Parameterised by two ORTHOGONAL tilt angles rather than (yaw, pitch).
    Polar (yaw, pitch) is singular when the scope points straight down: at
    pitch = 0 the yaw axis does nothing, so a visual servo loop cannot correct
    horizontal error and pitch runs to its limit instead. Two tilts have no
    such singularity near vertical, which is where a laparoscope lives.
    """
    v = np.array([np.tan(tilt_x), np.tan(tilt_y), -1.0])
    return v / np.linalg.norm(v)


def gaze_offset(tilt_x: float, tilt_y: float, depth: float) -> np.ndarray:
    """
    Where the instrument axis meets a plane `depth` below the trocar,
    as an offset from the trocar. Exactly linear in tan(tilt) — this is what
    makes image error map cleanly onto tilt rates.
    """
    return depth * np.array([np.tan(tilt_x), np.tan(tilt_y)])


def outside_length(insertion):
    """
    Length of instrument shaft outside the body.

    The instrument is RIGID and of fixed length, so pushing it deeper must move
    the arm's grip point toward the trocar by the same amount. An earlier
    version held the grip point fixed while only the tip moved, which made
    insertion a ghost degree of freedom: "suction park" retracted the tip and
    the arm never moved.
    """
    return max(cfg.INSTRUMENT_LENGTH - insertion, cfg.MIN_OUTSIDE)


def tool_pose_from_rcm(pivot, tilt_x, tilt_y, insertion, shaft=None):
    """
    Returns (tip, shaft_base, d).
        tip         instrument tip, inside the body
        shaft_base  where the arm's end effector grips the shaft, outside
        d           unit direction into the body
    """
    d = direction(tilt_x, tilt_y)
    out = outside_length(insertion) if shaft is None else shaft
    tip = np.asarray(pivot) + insertion * d
    shaft_base = np.asarray(pivot) - out * d
    return tip, shaft_base, d


def mat_to_quat(R: np.ndarray):
    """Rotation matrix -> PyBullet quaternion order [x, y, z, w]."""
    t = np.trace(R)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return [float(x), float(y), float(z), float(w)]


def quat_from_dir(d: np.ndarray, roll: float = 0.0):
    """
    Orientation whose local +Z axis aligns with d (the KUKA tool axis),
    with an optional roll about that axis.
    """
    z = np.asarray(d, dtype=float)
    z = z / np.linalg.norm(z)
    ref = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(ref, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    if abs(roll) > 1e-9:
        c, s = np.cos(roll), np.sin(roll)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        R = R @ Rz
    return mat_to_quat(R)


class RCMState:
    """Integrates velocity commands into RCM joint values, with clamping."""

    def __init__(self, pivot, tilt_x=0.0, tilt_y=0.0, insertion=0.12, roll=0.0):
        self.pivot = np.asarray(pivot, dtype=float)
        self.tilt_x = float(tilt_x)
        self.tilt_y = float(tilt_y)
        self.insertion = float(insertion)
        self.roll = float(roll)

    def integrate(self, cmd, dt: float):
        """cmd keys: tilt_x_rate, tilt_y_rate, insert_rate."""
        rx = float(np.clip(cmd.get("tilt_x_rate", 0.0),
                           -cfg.MAX_TILT_RATE, cfg.MAX_TILT_RATE))
        ry = float(np.clip(cmd.get("tilt_y_rate", 0.0),
                           -cfg.MAX_TILT_RATE, cfg.MAX_TILT_RATE))
        ir = float(np.clip(cmd.get("insert_rate", 0.0),
                           -cfg.MAX_INSERT_RATE, cfg.MAX_INSERT_RATE))

        self.tilt_x += rx * dt
        self.tilt_y += ry * dt
        self._clamp_cone()
        self.insertion = float(np.clip(self.insertion + ir * dt,
                                       *cfg.INSERTION_LIMITS))

    def _clamp_cone(self):
        """Clamp the tilt pair to a circular cone, not a square box."""
        tmax = np.tan(cfg.MAX_TILT)
        tx, ty = np.tan(self.tilt_x), np.tan(self.tilt_y)
        r = np.hypot(tx, ty)
        if r > tmax:
            tx, ty = tx * tmax / r, ty * tmax / r
            self.tilt_x, self.tilt_y = float(np.arctan(tx)), float(np.arctan(ty))

    def pose(self):
        return tool_pose_from_rcm(self.pivot, self.tilt_x, self.tilt_y,
                                  self.insertion)

    def gaze(self, depth=None):
        d = cfg.TISSUE_DEPTH if depth is None else depth
        return gaze_offset(self.tilt_x, self.tilt_y, d)

    @property
    def polar(self):
        """(yaw, pitch) for reporting — clinicians think in these terms."""
        tx, ty = np.tan(self.tilt_x), np.tan(self.tilt_y)
        return float(np.arctan2(ty, tx)), float(np.arctan(np.hypot(tx, ty)))

    def as_dict(self):
        yaw, pitch = self.polar
        return {"tilt_x": self.tilt_x, "tilt_y": self.tilt_y,
                "insertion": self.insertion, "roll": self.roll,
                "yaw": yaw, "pitch": pitch}


def _wrap(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def pivot_error(tip: np.ndarray, shaft_base: np.ndarray,
                pivot: np.ndarray) -> float:
    """
    Perpendicular distance from the true pivot to the instrument shaft line.
    Should be ~0 by construction — log it anyway as a sanity metric for your
    report, since it catches IK failures that silently move the shaft.
    """
    tip = np.asarray(tip)
    shaft_base = np.asarray(shaft_base)
    pivot = np.asarray(pivot)
    v = tip - shaft_base
    n = np.linalg.norm(v)
    if n < 1e-9:
        return 0.0
    return float(np.linalg.norm(np.cross(v / n, pivot - shaft_base)))
