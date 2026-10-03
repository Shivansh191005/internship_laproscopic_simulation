"""
The surgeon's monitor tower.

Left screen  : clean endoscope feed, exactly what a surgeon looks at today.
Right screen : the same feed with YOLO instance segmentation overlaid, plus the
               safety state.

Keeping them side by side rather than overlaying everything on one image is
deliberate. Surgeons reject augmentation that obscures the operative field, so
the reference view stays untouched and the interpretation sits next to it.
"""
import cv2
import numpy as np

import config as cfg

BEZEL = (24, 24, 28)
PANEL = (16, 17, 20)
LEVEL_COLOUR = {"clear": (110, 220, 120), "warn": (60, 200, 255),
                "danger": (70, 70, 255), "none": (150, 150, 150)}
CLASS_COLOUR = {
    "Ureter": (90, 230, 255),
    "Obturator Nerve": (200, 160, 255),
    "External Iliac Artery": (90, 90, 245),
    "External Iliac Vein": (235, 160, 80),
    "Uterus": (170, 200, 140),
    "Ovary": (140, 220, 200),
    "Uterine Artery": (120, 120, 245),
}


# Shown on the AI-overlay screen; main.py sets it from the loaded weights.
MODEL_LABEL = "YOLO26-seg"
LAST_OVERLAY = None


# Re-used image buffers. Allocating fresh full-size arrays every frame
# (np.full / hstack / vstack) cost ~10-20 ms per console frame, mostly in
# page faults; the console layout never changes, so buffers are kept.
_BUFS = {}


def _buf(key, shape, fill=None):
    b = _BUFS.get(key)
    new = b is None or b.shape != shape
    if new:
        b = np.empty(shape, np.uint8)
        _BUFS[key] = b
        if fill is not None:              # filled template, copied later
            t = np.empty(shape, np.uint8)
            t[:] = fill
            _BUFS[(key, "tpl")] = t
    if fill is not None:
        np.copyto(b, _BUFS[(key, "tpl")])   # memcpy: far faster than b[:]=c
    return b, new


def _screen(img, label, accent):
    """Wrap an image in a monitor bezel with a label strip."""
    h, w = img.shape[:2]
    pad, strip = 10, 26
    out, new = _buf(("screen", label), (h + 2 * pad + strip, w + 2 * pad, 3))
    if new:                       # bezel + label are static: draw once
        out[:] = BEZEL
        cv2.putText(out, label, (pad + 4, pad + h + strip - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, accent, 1, cv2.LINE_AA)
    out[pad:pad + h, pad:pad + w] = img
    cv2.rectangle(out, (pad, pad), (pad + w, pad + h), accent, 1)
    return out


# One colour per tool (BGR): yellow, magenta, cyan.
TOOL_COLOUR = [(0, 230, 255), (255, 80, 255), (255, 220, 60)]


def draw_tool(img, tid, ctr, tip, primary, back=None, contour=None):
    """
    One tool: its mask outline, the axis of its distal shaft (drawn ON the
    instrument, from the tip back along the jaws/shaft) and the end effector
    (tip) with its label. The primary tool - the one the surgeon is actively
    working with - gets a thicker line and a filled tip.
    """
    col = TOOL_COLOUR[(max(tid, 1) - 1) % len(TOOL_COLOUR)]
    if contour is not None:
        cv2.drawContours(img, [contour], -1, col, 1, cv2.LINE_AA)
    th = 3
    if tip is not None:
        t = (int(tip[0]), int(tip[1]))
        start = back if back is not None else ctr
        s = (int(start[0]), int(start[1]))
        cv2.line(img, s, t, (0, 0, 0), th + 2, cv2.LINE_AA)
        cv2.line(img, s, t, col, th, cv2.LINE_AA)
        cv2.circle(img, t, 10, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.circle(img, t, 10, col, 2, cv2.LINE_AA)
        cv2.circle(img, t, 4, col, -1, cv2.LINE_AA)
        lab, at = f"TOOL {tid}", t
    else:
        c = (int(ctr[0]), int(ctr[1]))
        lab, at = f"TOOL {tid}  tip out of view", c
    cv2.putText(img, lab, (at[0] + 12, at[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, lab, (at[0] + 12, at[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)


def draw_aim(img, p, colour=(90, 200, 255)):
    """Where the camera arm is aiming (tip, or midpoint of both tips)."""
    cv2.drawMarker(img, (int(p[0]), int(p[1])), colour,
                   cv2.MARKER_TILTED_CROSS, 14, 2, cv2.LINE_AA)


def draw_overlay(view, status, obs):
    """Right screen: segmentation masks + safety state on the endoscope view."""
    img = view.image.copy()
    h, w = img.shape[:2]
    tint = np.zeros_like(img)

    for name, cnt in view.critical_view.items():
        colour = CLASS_COLOUR.get(name, (180, 180, 180))
        cv2.drawContours(tint, [cnt], -1, colour, -1)
        cv2.drawContours(img, [cnt], -1, colour, 2)
        pt = cnt.reshape(-1, 2)[0]
        cv2.putText(img, name, (int(pt[0]), max(int(pt[1]) - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)
    img = cv2.addWeighted(img, 1.0, tint, 0.22, 0)

    for tid, ctr, tip, primary, back in view.instruments_view:
        draw_tool(img, tid, ctr, tip, primary, back)
    if view.instrument_visible:
        draw_aim(img, view.instrument_centroid)
        cv2.drawMarker(img, (w // 2, h // 2), (100, 100, 100),
                       cv2.MARKER_CROSS, 22, 1)

    lvl = status.get("level", "none")
    if lvl in ("warn", "danger"):
        col = LEVEL_COLOUR[lvl]
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), col, 6)
        txt = f"{lvl.upper()}  {status.get('closest','')}  {status.get('dist',0):.0f}px"
        if status.get("predictive"):
            txt = f"APPROACHING  {status.get('closest','')}  TTC {status.get('ttc',0):.1f}s"
        (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        cv2.rectangle(img, (w // 2 - tw // 2 - 8, 8),
                      (w // 2 + tw // 2 + 8, 36), (0, 0, 0), -1)
        cv2.putText(img, txt, (w // 2 - tw // 2, 29),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2, cv2.LINE_AA)
    return img


def _card(w, h, title, lines, accent, img=None):
    """One status tile: coloured title, then up to three detail lines
    (drawn straight into `img`, a slice of the panel, when given)."""
    if img is None:
        img = np.empty((h, w, 3), np.uint8)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), PANEL, -1)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (40, 42, 48), 1)
    cv2.rectangle(img, (0, 0), (3, h - 1), accent, -1)
    cv2.putText(img, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.56, accent,
                2, cv2.LINE_AA)
    for i, (txt, col) in enumerate(lines[:3]):
        cv2.putText(img, txt, (12, 46 + i * 19), cv2.FONT_HERSHEY_SIMPLEX,
                    0.44, col, 1, cv2.LINE_AA)
    return img


def status_panel(width, tel, status, cmd, sim_diag, fps, infer_ms,
                 voice_status, extra):
    """
    Grid of status tiles that reflows to any width, so the console never
    grows wider than the two monitors (the old fixed-position bar forced a
    wide window that ran off the screen).
    """
    grey, dim = (215, 215, 220), (130, 130, 138)
    lvl = status.get("level", "none")
    mode_col = (120, 230, 140) if cmd["mode"] == "FOLLOW" else (80, 190, 255)
    ttc = status.get("ttc", float("inf"))
    fresh = cmd.get("age", 99) < 2.5
    clr = extra.get("arm_clearance")
    clr_col = ((70, 70, 255) if clr is not None and clr < cfg.ARM_MIN_CLEARANCE
               else (120, 230, 140))

    cards = [
        (extra.get("cam_title") or
         (f"CAMERA ARM  STABLE" if cfg.CAMERA_MODE == "stable"
          else f"CAMERA ARM  {cmd['mode']}"), [
            ("endoscope + light, inside patient", dim),
            (extra.get("cam_activity", "-"), grey),
            (f"RCM dev {sim_diag['camera']['pivot_error_m'] * 1000:.1f} mm"
             + ("" if cfg.CAMERA_MODE == "stable" else
                f"   err {np.hypot(tel.get('err_x', 0), tel.get('err_y', 0)):.2f}"),
             grey)],
         (80, 190, 255)),
        (f"INSTRUMENT ARM  {cmd.get('instrument', 'FOLLOW')}", [
            (f"follows: {extra.get('aim', '-')}", dim),
            (extra.get("suc_activity", "-"),
             (70, 70, 255) if extra.get("arm_guard") else grey),
            (("fingers " + " / ".join(
                "-" if e is None else f"{e:.1f}" for e in
                (tel.get("finger_mm") or [None, None])) + " mm")
             + f"   RCM dev {sim_diag['suction']['pivot_error_m'] * 1000:.1f} mm",
             grey)],
         (255, 190, 110)),
        (f"SAFETY  {lvl.upper()}", [
            (f"{status.get('closest') or '-'}"
             + (f"  (tool {status['tool']})" if status.get("tool") else ""),
             grey),
            (f"{status['dist']:.0f} px" if status.get("dist") is not None
             else "no structure", grey),
            ((f"TTC {ttc:.1f} s" + ("  PREDICTED" if status.get("predictive")
                                     else "")) if ttc < 99 else "not closing",
             (60, 200, 255) if status.get("predictive") else dim)],
         LEVEL_COLOUR[lvl]),
        ("ARMS", [
            (f"clearance {clr * 100:.1f} cm" if clr is not None else
             "clearance -", clr_col),
            (f"view: {extra.get('view_name', '-')}", dim)],
         clr_col),
        ("PERFORMANCE", [
            (f"inference {infer_ms:5.1f} ms", grey),
            (f"loop {fps:5.1f} fps", grey)],
         (200, 200, 205)),
        ("LAST COMMAND", [
            (cmd.get("last") or "-", (120, 240, 160) if fresh else dim),
            (f"voice: {voice_status}", dim)],
         (150, 150, 160)),
    ]
    cw, ch = 230, 104
    cols = max(1, width // cw)
    cw = width // cols
    rows = (len(cards) + cols - 1) // cols
    panel, _ = _buf("status", (rows * ch, width, 3), PANEL)
    for i, (title, lines, accent) in enumerate(cards):
        r, c = divmod(i, cols)
        _card(cw, ch, title, lines, accent,
              panel[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw])
    return panel


_RESIZED = {}


_LIGHT_LUT = {}


def _light(img, gain):
    """Scope light up / down: brighten or dim the video (gamma-shaped, so
    highlights do not burn out), applied BEFORE the AI masks are drawn."""
    if abs(gain - 1.0) < 1e-3:
        return img
    lut = _LIGHT_LUT.get(gain)
    if lut is None:
        x = np.arange(256) / 255.0
        lut = (np.clip(x ** (1.0 / gain), 0, 1) * 255).astype(np.uint8)
        _LIGHT_LUT[gain] = lut
    return cv2.LUT(img, lut)


def draw_full(obs, status, view, with_overlay, crop=None, light=1.0):
    """
    Full recorded frame, never cropped or zoomed - so no black borders and a
    steady picture. The scope's framing is drawn as a rectangle so the robot's
    aim is still visible. The servo loop is unaffected: it runs on the virtual
    endoscope regardless of what the monitors display.
    """
    H, W = obs.frame.shape[:2]
    # Preserve the video's aspect ratio. Forcing 16:9 footage into the 5:4
    # scope panel stretches the anatomy, which matters when a surgeon is
    # judging tissue planes by eye.
    ow = cfg.VIEW_SIZE[0]
    oh = max(1, int(round(ow * H / W)))
    # Voice zoom / pan: show only the window `crop` = (x0, y0, x1, y1) as
    # fractions of the frame. Everything drawn below is mapped into it, and
    # the warning banner is drawn AFTER, so it is never cropped away.
    if crop is None:
        crop = (0.0, 0.0, 1.0, 1.0)
    X0, Y0 = int(round(crop[0] * W)), int(round(crop[1] * H))
    X1, Y1 = int(round(crop[2] * W)), int(round(crop[3] * H))
    X1, Y1 = max(X1, X0 + 2), max(Y1, Y0 + 2)
    sx, sy = ow / (X1 - X0), oh / (Y1 - Y0)
    # the clean and the overlay screen show the same frame: resize it once
    key = (id(obs.frame), ow, oh, X0, Y0, X1, Y1)
    base = _RESIZED.get(key)
    if base is None:
        if len(_RESIZED) >= 4:            # full view + voice view, 2 frames
            _RESIZED.clear()
        base = cv2.resize(obs.frame[Y0:Y1, X0:X1], (ow, oh),
                          interpolation=cv2.INTER_AREA)
        _RESIZED[key] = base
    img = _light(base, light)
    if img is base:
        img = img.copy()

    if with_overlay:
        tint = np.zeros_like(img)
        for name, det in obs.critical.items():
            if det is None or det.contour is None:
                continue
            colour = CLASS_COLOUR.get(name, (180, 180, 180))
            c = det.contour.reshape(-1, 2).astype(np.float32)
            c[:, 0] = (c[:, 0] - X0) * sx
            c[:, 1] = (c[:, 1] - Y0) * sy
            c = c.astype(np.int32).reshape(-1, 1, 2)
            cv2.drawContours(tint, [c], -1, colour, -1)
            cv2.drawContours(img, [c], -1, colour, 2)
            pt = c.reshape(-1, 2)[0]
            cv2.putText(img, name, (int(pt[0]), max(int(pt[1]) - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, colour, 1, cv2.LINE_AA)
        img = cv2.addWeighted(img, 1.0, tint, 0.20, 0)

        tools = obs.instruments or ([obs.instrument] if obs.instrument
                                    else [])
        for t in tools:
            ctr = ((t.centroid[0] - X0) * sx, (t.centroid[1] - Y0) * sy)
            tp = (((t.tip[0] - X0) * sx, (t.tip[1] - Y0) * sy)
                  if t.tip and t.tip_visible else None)
            bk = (((t.back[0] - X0) * sx, (t.back[1] - Y0) * sy)
                  if t.back else None)
            cnt = None
            if t.contour is not None:
                cc = t.contour.reshape(-1, 2).astype(np.float32)
                cc[:, 0] = (cc[:, 0] - X0) * sx
                cc[:, 1] = (cc[:, 1] - Y0) * sy
                cnt = cc.astype(np.int32).reshape(-1, 1, 2)
            draw_tool(img, t.track_id or 1, ctr, tp, t is obs.instrument,
                      bk, cnt)
        from perception import aim_point
        aim = aim_point(obs)
        if aim is not None and cfg.CAMERA_MODE != "stable":
            draw_aim(img, ((aim[0] - X0) * sx, (aim[1] - Y0) * sy))

    # what the robot's scope is framing (only when the camera arm moves)
    if cfg.CAMERA_MODE != "stable":
        x0, y0, x1, y1 = view.crop
        cv2.rectangle(img, (int((x0 - X0) * sx), int((y0 - Y0) * sy)),
                      (int((x1 - X0) * sx), int((y1 - Y0) * sy)),
                      (90, 200, 255), 2)
    if with_overlay:
        lvl = status.get("level", "none")
        if lvl in ("warn", "danger"):
            col = LEVEL_COLOUR[lvl]
            h, w = img.shape[:2]
            cv2.rectangle(img, (0, 0), (w - 1, h - 1), col, 5)
            txt = (f"{lvl.upper()}  {status.get('closest','')}  "
                   f"{status.get('dist',0):.0f}px")
            if status.get("predictive"):
                txt = (f"APPROACHING  {status.get('closest','')}  "
                       f"TTC {status.get('ttc',0):.1f}s")
            (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
            cv2.rectangle(img, (w // 2 - tw // 2 - 8, 6),
                          (w // 2 + tw // 2 + 8, 34), (0, 0, 0), -1)
            cv2.putText(img, txt, (w // 2 - tw // 2, 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, col, 2, cv2.LINE_AA)
    return img


def compose(view, obs, status, tel, cmd, sim_diag, fps, infer_ms,
            voice_status, sim_img=None, extra=None):
    """
    Assemble the surgeon console:

        [ endoscope  |  AI overlay ]
        [ OR view    |  status tiles ]     <- OR view only with --show-sim

    The OR view sits UNDER the monitors instead of beside them, so the console
    is never wider than the two screens.
    """
    extra = extra or {}
    if cfg.DISPLAY_MODE == "full":
        crop = extra.get("view_crop")
        light = extra.get("light", 1.0)
        # Left screen: always the untouched camera feed (full field).
        # Right screen: the AI overlay, with the voice zoom / pan / light.
        clean = draw_full(obs, status, view, with_overlay=False)
        overlay = draw_full(obs, status, view, with_overlay=True, crop=crop,
                            light=light)
        if crop is not None and (crop[2] - crop[0]) < 0.999:
            z = 1.0 / max(crop[2] - crop[0], 1e-3)
            for im in (overlay,):
                cv2.putText(im, f"ZOOM {z:.1f}x", (10, im.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3,
                            cv2.LINE_AA)
                cv2.putText(im, f"ZOOM {z:.1f}x", (10, im.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 220, 255),
                            1, cv2.LINE_AA)
    else:
        clean = view.image.copy()
        overlay = draw_overlay(view, status, obs)

    global LAST_OVERLAY
    LAST_OVERLAY = overlay          # the OR tower's second monitor shows it
    tag = "  [full field]" if cfg.DISPLAY_MODE == "full" else ""
    rtag = ""
    vc = extra.get("view_crop")
    if vc is not None and (vc[2] - vc[0]) < 0.999:
        rtag = "  [voice view - AI still checks the full field]"
    left = _screen(clean, "ENDOSCOPE  (camera feed)" + tag, (150, 190, 160))
    right = _screen(overlay, f"AI OVERLAY  ({MODEL_LABEL})" + rtag,
                    (120, 200, 235))
    W = left.shape[1] + right.shape[1]
    Ht = max(left.shape[0], right.shape[0])

    if sim_img is not None:
        thumb_w = W // 3
        th = int(sim_img.shape[0] * thumb_w / sim_img.shape[1])
        thumb = _screen(cv2.resize(sim_img, (thumb_w - 20, th)),
                        "OPERATING ROOM  (PyBullet)", (200, 200, 205))
        panel = status_panel(W - thumb.shape[1], tel, status, cmd, sim_diag,
                             fps, infer_ms, voice_status, extra)
        h = max(thumb.shape[0], panel.shape[0])
        out, _ = _buf("console", (Ht + h, W, 3), PANEL)
        out[Ht:Ht + thumb.shape[0], :thumb.shape[1]] = thumb
        out[Ht:Ht + panel.shape[0], thumb.shape[1]:thumb.shape[1] +
            panel.shape[1]] = panel
    else:
        bottom = status_panel(W, tel, status, cmd, sim_diag, fps, infer_ms,
                              voice_status, extra)
        out, _ = _buf("console", (Ht + bottom.shape[0], W, 3), PANEL)
        out[Ht:, :bottom.shape[1]] = bottom
    out[:left.shape[0], :left.shape[1]] = left
    out[:right.shape[0], left.shape[1]:] = right

    s = min(cfg.MAX_DISPLAY_WIDTH / out.shape[1],
            cfg.MAX_DISPLAY_HEIGHT / out.shape[0], 1.0)
    if s < 1.0:
        out = cv2.resize(out, (int(out.shape[1] * s), int(out.shape[0] * s)),
                         interpolation=cv2.INTER_AREA)
    return out


def _pad_h(img, h):
    if img.shape[0] >= h:
        return img
    pad = np.full((h - img.shape[0], img.shape[1], 3), PANEL, np.uint8)
    return np.vstack([img, pad])


def _screen_fit(img, h):
    if img.shape[0] == h:
        return img
    return cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))
