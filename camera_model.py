"""
The fixed endoscope (arm 1) and the link between the video and the world.

Arm 1 holds the endoscope + light inside the patient and does NOT move: the
recorded video is exactly what this fixed camera sees. That makes the video a
measurement in a known camera frame, so a pixel can be turned into a 3D point
on the tissue (and a 3D point back into a pixel):

    pixel  --(camera ray)-->  point on the tissue plane  -->  arm 2 target

Image orientation follows the usual laparoscopy convention for a surgeon at
the patient's feet looking toward the head-end monitors: image UP = toward the
patient's head (+x), image RIGHT = the surgeon's right (-y).
"""
import numpy as np

import config as cfg
from rcm import RCMState


def working_point():
    """Centre of the operative field on the tissue plane, between the ports."""
    c, s = cfg.TROCAR_CAMERA, cfg.TROCAR_INSTRUMENT
    return np.array([(c[0] + s[0]) / 2 + cfg.FIELD_OFFSET[0],
                     (c[1] + s[1]) / 2 + cfg.FIELD_OFFSET[1],
                     cfg.TISSUE_Z])


def fixed_camera_state():
    """RCM state of arm 1: aimed at the working point, fixed insertion."""
    v = working_point() - np.asarray(cfg.TROCAR_CAMERA, float)
    tx = float(np.arctan2(v[0], -v[2]))
    ty = float(np.arctan2(v[1], -v[2]))
    return RCMState(cfg.TROCAR_CAMERA, tilt_x=tx, tilt_y=ty,
                    insertion=cfg.CAMERA_INSERTION)


class FixedCamera:
    """Pinhole model of the fixed endoscope, in frame pixels."""

    def __init__(self, frame_w, frame_h, state=None):
        self.W, self.H = float(frame_w), float(frame_h)
        self.state = state or fixed_camera_state()
        tip, _, d = self.state.pose()
        self.pos = np.asarray(tip, float)              # lens position
        self.f = np.asarray(d, float)                  # optical axis
        up_hint = np.array([1.0, 0.0, 0.0])            # patient's head
        r = np.cross(self.f, up_hint)
        self.r = r / np.linalg.norm(r)                 # image right
        u = np.cross(self.r, self.f)
        self.u = u / np.linalg.norm(u)                 # image up
        self.fx = (self.W / 2.0) / np.tan(np.radians(cfg.CAMERA_HFOV / 2.0))

    def ray(self, px, py):
        d = self.f + ((px - self.W / 2) / self.fx) * self.r \
            - ((py - self.H / 2) / self.fx) * self.u
        return d / np.linalg.norm(d)

    def pixel_to_tissue(self, px, py, z=None):
        """Where the ray through pixel (px, py) meets the tissue plane."""
        z = cfg.TISSUE_Z if z is None else z
        d = self.ray(px, py)
        if abs(d[2]) < 1e-6:
            return None
        t = (z - self.pos[2]) / d[2]
        if t <= 0:
            return None
        return self.pos + t * d

    def world_to_pixel(self, p):
        v = np.asarray(p, float) - self.pos
        zc = v @ self.f
        if zc <= 1e-6:
            return None
        return (self.W / 2 + self.fx * (v @ self.r) / zc,
                self.H / 2 - self.fx * (v @ self.u) / zc)

    def metres_per_pixel(self):
        """Scale on the tissue plane at the image centre."""
        a = self.pixel_to_tissue(self.W / 2, self.H / 2)
        b = self.pixel_to_tissue(self.W / 2 + 100, self.H / 2)
        return float(np.linalg.norm(b - a) / 100.0)


def rcm_for_tip(pivot, target):
    """
    Tilts + insertion that put an instrument through `pivot` with its tip at
    `target` (insertion clamped to the allowed range, so the tool always stays
    inside the patient).
    """
    v = np.asarray(target, float) - np.asarray(pivot, float)
    tx = float(np.arctan2(v[0], -v[2]))
    ty = float(np.arctan2(v[1], -v[2]))
    ins = float(np.clip(np.linalg.norm(v), *cfg.INSERTION_LIMITS))
    return tx, ty, ins
