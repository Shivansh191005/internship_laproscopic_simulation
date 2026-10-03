"""
Put the tool line ON the instrument.

The segmentation mask only says roughly where a tool is: it spills onto
tissue, misses glare-white or blood-covered parts of the shaft, and its outline
is coarse. A line fitted to that mask inherits all of it.

But a laparoscopic instrument is a straight rigid bar, and in the video it has
two long, sharp, parallel edges (metal against tissue). Those edges are in the
PIXELS, not in the mask. So:

  1. use the mask only as a search area (slightly enlarged),
  2. inside it, keep the pixels that LOOK like metal (grey, silver, black)
     and fit the tool's axis to those - tissue the mask spilled onto is
     ignored,
  3. look for the shaft's two long parallel edges in the frame and, if a
     clear pair is found, centre the line exactly between them,
  4. the line runs along that axis to the tip.

If the picture does not show clear shaft edges (motion blur, smoke, tool
almost hidden), the old mask-based line is kept, so this can only help.
"""
import cv2
import numpy as np

import config as cfg

try:
    _LSD = cv2.createLineSegmentDetector()
except Exception:                                           # noqa: BLE001
    _LSD = None


def _segments(gray):
    """Straight edge segments as (N, 4) array x1 y1 x2 y2."""
    if _LSD is not None:
        try:
            lines = _LSD.detect(gray)[0]
            if lines is None:
                return np.zeros((0, 4), np.float32)
            return lines.reshape(-1, 4).astype(np.float32)
        except Exception:                                   # noqa: BLE001
            pass
    edges = cv2.Canny(gray, 40, 120)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30,
                            minLineLength=20, maxLineGap=6)
    if lines is None:
        return np.zeros((0, 4), np.float32)
    return lines.reshape(-1, 4).astype(np.float32)


def _ang(d):
    """Undirected line angle in [0, pi)."""
    return float(np.arctan2(d[1], d[0]) % np.pi)


def _ang_diff(a, b):
    d = abs(a - b) % np.pi
    return min(d, np.pi - d)


def metal_mask(bgr):
    """
    Pixels that look like instrument metal: grey/silver (low saturation) or
    dark/black. Measured on this video: catches ~68% of tool pixels and only
    ~8% of tissue pixels (tissue is saturated red / pink / yellow).
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    S, V = hsv[..., 1].astype(np.int16), hsv[..., 2].astype(np.int16)
    m = ((S < cfg.METAL_S_MAX) & (V < cfg.METAL_V_MAX)) | (S < cfg.METAL_S_GREY)
    return m.astype(np.uint8)


def _pca(p):
    mean = p.mean(axis=0)
    _, sv, vt = np.linalg.svd(p - mean, full_matrices=False)
    return mean, vt[0], sv


def _single_tool_run(frame, pts, along, lo, hi):
    """
    Trim the line to ONE tool when the mask runs on into another one.

    When two tools touch end to end (a black grasper's jaws against a silver
    suction tube), the model often colours both as one mask and the metal
    pixels run on continuously, so the line overshoots onto the second tool.
    Different instruments differ in brightness (black vs. silver), so walk
    along the line in short steps, split it where the brightness jumps, and
    keep the longest run.

    OFF by default (cfg.AXIS_RUN_SPLIT): on this video it also cut the
    bright silver jaws off black-shafted graspers, moving the tip back onto
    the shaft.
    """
    n_bins = cfg.AXIS_RUN_BINS
    if not cfg.AXIS_RUN_SPLIT or hi - lo < 10 or len(pts) < n_bins * 3:
        return lo, hi
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    xs = np.clip(pts[:, 0].astype(int), 0, W - 1)
    ys = np.clip(pts[:, 1].astype(int), 0, H - 1)
    v = gray[ys, xs].astype(np.float32)
    edges = np.linspace(lo, hi, n_bins + 1)
    idx = np.clip(np.digitize(along, edges) - 1, 0, n_bins - 1)
    med = np.full(n_bins, np.nan, np.float32)
    for b in range(n_bins):
        sel = v[idx == b]
        if len(sel) >= 3:
            med[b] = np.median(sel)
    good = ~np.isnan(med)
    if good.sum() < 4:
        return lo, hi
    # fill gaps, smooth lightly, find big jumps between neighbouring bins
    xb = np.arange(n_bins)
    med = np.interp(xb, xb[good], med[good])
    med = np.convolve(np.pad(med, 1, mode="edge"), np.ones(3) / 3, "valid")
    cuts = [0] + [b for b in range(1, n_bins)
                  if abs(med[b] - med[b - 1]) > cfg.AXIS_RUN_JUMP
                  or abs(np.mean(med[max(0, b - 3):b]) -
                         np.mean(med[b:b + 3])) > cfg.AXIS_RUN_JUMP] + [n_bins]
    runs = [(cuts[k], cuts[k + 1]) for k in range(len(cuts) - 1)
            if cuts[k + 1] > cuts[k]]
    # merge tiny fragments into neighbours by keeping only the longest run
    a, b = max(runs, key=lambda r: r[1] - r[0])
    if (b - a) < 0.35 * n_bins:
        return lo, hi                     # no clear single tool: keep all
    return float(edges[a]), float(edges[b])


def refine(frame, contour, tip_hint, field=None):
    """
    Returns (tip, back, quality) with the line on the instrument, or None
    when the picture gives no usable evidence (then the old line is kept).

        tip      end-effector end of the centre line (frame pixels)
        back     a point further up the shaft on the same line
        quality  0..1, how much evidence supported the line

    Two kinds of evidence from the actual video frame:
      * colour: metal pixels inside (and just around) the mask. The axis is
        fitted to THOSE, so mask spill onto tissue no longer drags the line
        off the tool.
      * edges: the shaft's two long parallel edges. When a clear pair is
        found, the line is centred exactly between them.
    """
    H, W = frame.shape[:2]
    pts = contour.reshape(-1, 2).astype(np.float32)
    if len(pts) < 5:
        return None
    mean0, u0, _ = _pca(pts)
    area = abs(cv2.contourArea(pts))
    proj = (pts - mean0) @ u0
    length = float(proj.max() - proj.min())
    if length < 0.05 * W:
        return None
    width0 = max(area / max(length, 1.0), 4.0)

    s = cfg.AXIS_WORK_SCALE
    x, y, bw, bh = cv2.boundingRect(pts.astype(np.int32))
    m = int(cfg.AXIS_MARGIN_FRAC * W)
    x0, y0 = max(x - m, 0), max(y - m, 0)
    x1, y1 = min(x + bw + m, W), min(y + bh + m, H)
    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    small = cv2.resize(roi, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    off0 = np.array([x0, y0], np.float32)

    def to_frame(xy):
        return xy / s + off0

    mask = np.zeros(small.shape[:2], np.uint8)
    cv2.fillPoly(mask, [((pts - off0) * s).astype(np.int32)], 1)
    grow = max(2, int(0.3 * width0 * s))
    near = cv2.dilate(mask, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8))

    # ---- 1. colour: metal pixels in/around the mask --------------------
    metal = metal_mask(small) & near
    metal = cv2.morphologyEx(metal, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n_l, lab, stats, _ = cv2.connectedComponentsWithStats(metal)
    use = np.zeros_like(metal)
    for li in range(1, n_l):
        comp = lab == li
        if stats[li, cv2.CC_STAT_AREA] >= 0.05 * mask.sum() and \
                (comp & (mask > 0)).sum() >= 0.5 * comp.sum():
            use |= comp.astype(np.uint8)
    ys, xs = np.nonzero(use)
    metal_frac = len(xs) / max(mask.sum(), 1)
    if metal_frac >= cfg.AXIS_MIN_METAL_FRAC and len(xs) > 30:
        mp = to_frame(np.column_stack([xs, ys]).astype(np.float32))
        centre, u, sv = _pca(mp)
        evidence = mp
        q_col = float(np.clip(metal_frac, 0, 1))
    else:
        ys, xs = np.nonzero(mask)
        if len(xs) < 30:
            return None
        mp = to_frame(np.column_stack([xs, ys]).astype(np.float32))
        centre, u, sv = _pca(mp)
        evidence = mp
        q_col = 0.0
    if u @ u0 < 0:
        u = -u
    nrm = np.array([-u[1], u[0]], np.float32)
    across = (evidence - centre) @ nrm
    half_w = max(float(np.percentile(np.abs(across), 80)), 2.0)

    # ---- 2. edges: centre the line between the shaft's two edges -------
    q_edge = 0.0
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    seg = _segments(gray[::2, ::2] if cfg.AXIS_EDGE_HALF else gray)
    if cfg.AXIS_EDGE_HALF and len(seg):
        seg = seg * 2.0
    if len(seg):
        p, q = to_frame(seg[:, :2]), to_frame(seg[:, 2:])
        d = q - p
        L = np.hypot(d[:, 0], d[:, 1])
        mid = (p + q) / 2
        a = _ang(u)
        angs = np.arctan2(d[:, 1], d[:, 0]) % np.pi
        dang = np.abs(angs - a) % np.pi
        dang = np.minimum(dang, np.pi - dang)
        ok = (L >= cfg.AXIS_MIN_SEG_FRAC * length) & \
            (dang < np.radians(cfg.AXIS_PARALLEL_DEG))
        ok &= np.abs((mid - centre) @ nrm) < 2.5 * half_w
        if ok.sum() >= 2:
            off = (mid[ok] - centre) @ nrm
            Lk = L[ok]
            best = None
            for i in range(len(off)):
                for j in range(len(off)):
                    gap = off[j] - off[i]
                    if not (cfg.AXIS_WIDTH_RANGE[0] * 2 * half_w <= gap <=
                            cfg.AXIS_WIDTH_RANGE[1] * 2 * half_w):
                        continue
                    if off[i] > 0.5 * half_w or off[j] < -0.5 * half_w:
                        continue        # the pair must straddle the metal
                    wi = Lk[np.abs(off - off[i]) < 0.15 * gap].sum()
                    wj = Lk[np.abs(off - off[j]) < 0.15 * gap].sum()
                    sc = min(wi, wj)
                    if best is None or sc > best[0]:
                        best = (sc, off[i], off[j])
            if best is not None and best[0] >= cfg.AXIS_MIN_SEG_FRAC * length:
                sc, oa, ob = best
                centre = centre + nrm * (oa + ob) / 2.0
                half_w = (ob - oa) / 2.0
                q_edge = float(np.clip(sc / (0.5 * length), 0, 1))

    quality = max(q_col, q_edge)
    if quality < cfg.AXIS_MIN_QUALITY:
        return None

    # ---- 3. extent along the shaft and the tip end ---------------------
    band = np.abs((evidence - centre) @ nrm) <= 1.5 * half_w
    along = (evidence[band] - centre) @ u if band.sum() > 10 else \
        (evidence - centre) @ u
    lo, hi = float(np.percentile(along, 1)), float(np.percentile(along, 99))
    lo, hi = _single_tool_run(frame, evidence[band] if band.sum() > 10
                              else evidence, along, lo, hi)
    end_lo, end_hi = centre + u * lo, centre + u * hi
    # Which end is the tip? The tool came INTO the picture at one end, so
    # carrying on along the line past that end hits the edge of the picture
    # (the round scope edge) almost at once; past the tip there is still
    # plenty of picture. Decided on THIS cleaned-up line, because the mask's
    # own guess fails when the mask spills onto a neighbouring tool.
    from perception import edge_distance
    d_lo = edge_distance(end_lo, frame.shape, field, (-u[0], -u[1]))
    d_hi = edge_distance(end_hi, frame.shape, field, (u[0], u[1]))
    th = np.asarray(tip_hint, np.float32)
    if abs(d_lo - d_hi) > cfg.TIP_AMBIGUOUS_FRAC * W:
        lo_is_tip = d_lo > d_hi
    else:   # no clear answer from the picture: keep the previous guess
        lo_is_tip = np.hypot(*(end_lo - th)) < np.hypot(*(end_hi - th))
    tip, entry = (end_lo, end_hi) if lo_is_tip else (end_hi, end_lo)
    back = tip + (entry - tip) * cfg.DISTAL_AXIS_FRAC
    return (float(tip[0]), float(tip[1])), (float(back[0]), float(back[1])), \
        quality
