"""
Self-test: checks every function of the system against a known correct answer.

    python selftest.py                     all sections that can run
    python selftest.py --no-sim            skip the PyBullet section
    python selftest.py --frames 300        sample more video for perception

Sections
    A  Kinematics      RCM, rigid instrument, reach, tilt limits
    B  Control/safety  arm 2 follows the video tool, hold, always inside,
                       camera model, safety levels, TTC, guards, (follow-mode
                       camera servo),
                       occlusion hold, virtual fixture, voice grammar
    C  Perception      your model on your video: detection rate, confidence,
                       stability, latency - and whether a high tracking error
                       is a DETECTION problem or a GEOMETRY limit
    D  Simulation      PyBullet: IK accuracy, physical RCM compliance of the
                       simulated arms, arm clearance, depth change moves arm
    E  Closed loop     whole robot on a two-tool scene with known truth

Results print as a table and are saved to runs/selftest.md.
"""
import argparse
import os
import sys
import time

import numpy as np

import config as cfg

RESULTS = []


def record(section, name, status, measured, criterion, note=""):
    RESULTS.append((section, name, status, measured, criterion, note))
    tag = {"PASS": "PASS", "WARN": "WARN", "FAIL": "FAIL", "SKIP": "SKIP"}[status]
    print(f"  [{tag}] {name:42s} {measured:>22s}   ({criterion})"
          + (f"  - {note}" if note else ""))


def grade(value, pass_if, warn_if):
    if pass_if(value):
        return "PASS"
    return "WARN" if warn_if(value) else "FAIL"


class FakeClock:
    """Deterministic time for the safety layer's closing-speed estimate."""
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


# ============================================================ A. kinematics
def section_a():
    print("\nA. KINEMATICS")
    from rcm import (RCMState, tool_pose_from_rcm, pivot_error, direction,
                     outside_length)

    worst = 0.0
    for tx in np.linspace(-cfg.MAX_TILT, cfg.MAX_TILT, 15):
        for ty in np.linspace(-cfg.MAX_TILT, cfg.MAX_TILT, 15):
            for ins in np.linspace(*cfg.INSERTION_LIMITS, 5):
                tip, sb, _ = tool_pose_from_rcm(cfg.TROCAR_CAMERA, tx, ty, ins)
                worst = max(worst, pivot_error(tip, sb, cfg.TROCAR_CAMERA))
    record("A", "RCM pivot (commanded pose)", grade(worst, lambda v: v < 1e-9,
           lambda v: v < 1e-6), f"{worst:.1e} m", "< 1e-9 m")

    lens, grips = [], []
    for ins in np.linspace(*cfg.INSERTION_LIMITS, 6):
        tip, sb, _ = tool_pose_from_rcm(cfg.TROCAR_SUCTION, 0.2, 0.1, ins)
        lens.append(np.linalg.norm(tip - sb))
        grips.append(sb[2])
    spread = max(lens) - min(lens)
    record("A", "Instrument is rigid (constant length)",
           grade(spread, lambda v: v < 1e-9, lambda v: v < 1e-4),
           f"{spread:.1e} m", "length change < 1e-9 m")
    moved = max(grips) - min(grips)
    record("A", "Insertion moves the arm's grip point",
           grade(moved, lambda v: v > 0.10, lambda v: v > 0.02),
           f"{moved * 100:.1f} cm", "> 10 cm over insertion range")

    s = RCMState(cfg.TROCAR_CAMERA)
    for _ in range(500):
        s.integrate({"tilt_x_rate": 9, "tilt_y_rate": 9}, 1 / 30)
    _, pitch = s.polar
    record("A", "Tilt cone limit enforced",
           grade(abs(pitch - cfg.MAX_TILT), lambda v: v < 1e-6,
                 lambda v: v < 0.01),
           f"{np.degrees(pitch):.1f} deg", f"= {np.degrees(cfg.MAX_TILT):.0f} deg")

    # Reach is checked by solving the real IK over the whole tilt cone x
    # insertion range (scene.check_reachability), not by a distance-from-
    # shoulder rule of thumb, which passed while a third of the cone was
    # unreachable.
    try:
        import scene
        scene.check_reachability(verbose=False)
        for name in ("camera", "instrument"):
            f = scene.REACH_FRACTION.get(name, 0.0)
            record("A", f"{name} arm reaches its workspace (real IK)",
                   grade(f, lambda v: v >= 0.95,
                         lambda v: v >= cfg.MIN_REACHABLE_FRAC),
                   f"{100 * f:.1f}%", ">= 95% of cone x insertion",
                   "the rest is refused at the cone edge, never forced")
    except Exception as e:                                  # noqa: BLE001
        record("A", "Reach check (needs pybullet)", "SKIP", "n/a", str(e))


# ====================================================== B. control & safety
def _obs(W, H, inst=None, structures=None, index=0):
    from perception import Observation, Detection
    o = Observation(frame=np.zeros((H, W, 3), np.uint8), index=index)
    if inst is not None:
        o.instrument = Detection(cfg.CLASS_INSTRUMENT, tuple(inst), 0.03,
                                 contour=None, tip=tuple(inst))
    for name, cnt in (structures or {}).items():
        c = np.asarray(cnt).reshape(-1, 2)
        o.critical[name] = Detection(name, tuple(c.mean(axis=0)), 0.02,
                                     contour=np.asarray(cnt, np.int32))
    return o


def _square(cx, cy, r):
    return np.array([[[cx - r, cy - r]], [[cx + r, cy - r]],
                     [[cx + r, cy + r]], [[cx - r, cy + r]]], np.int32)


def section_b():
    print("\nB. CONTROL AND SAFETY")
    import control
    from control import (CameraServo, SafetyLayer, InstrumentArm,
                         guard_instrument)
    from endoscope import VirtualEndoscope
    from rcm import RCMState
    from commands import CommandBus, match_phrase
    from perception import _Smoother, Detection
    from camera_model import FixedCamera, fixed_camera_state, rcm_for_tip, \
        working_point

    W, H = 1920, 1080
    dt = 1 / 30
    # The camera-servo checks below test the OPTIONAL "follow" camera mode
    # (CAMERA_MODE = "follow"); the default is a fixed camera.
    mode_saved = cfg.CAMERA_MODE
    cfg.CAMERA_MODE = "follow"
    scope = VirtualEndoscope(W, H)

    # --- servo converges on a static, reachable target
    tgt = (W / 2 + 0.12 * W, H / 2 - 0.08 * H)
    cam, servo, bus = RCMState(cfg.TROCAR_CAMERA, insertion=0.13), CameraServo(), CommandBus()
    errs = []
    for i in range(300):
        o = _obs(W, H, tgt, index=i)
        c, tel = servo.update(scope.project(o, cam), bus.snapshot())
        cam.integrate(c, dt)
        scope.clamp_state(cam)
        errs.append(np.hypot(tel["err_x"], tel["err_y"]))
    record("B", "[follow mode] Camera servo centres a target",
           grade(errs[-1], lambda v: v < 0.10, lambda v: v < 0.20),
           f"{errs[0]:.2f} -> {errs[-1]:.3f}", "final error < 0.10")

    # --- reacquire: instrument starts outside the view
    far = (W / 2 + 0.36 * W, H / 2)
    cam, servo = RCMState(cfg.TROCAR_CAMERA, insertion=0.13), CameraServo()
    first_mode, found = None, None
    for i in range(300):
        o = _obs(W, H, far, index=i)
        c, tel = servo.update(scope.project(o, cam), bus.snapshot())
        first_mode = first_mode or tel["mode"]
        cam.integrate(c, dt)
        scope.clamp_state(cam)
        if tel.get("in_view") and found is None:
            found = i
    ok = first_mode == "reacquire" and found is not None
    record("B", "[follow mode] Reacquires a lost target",
           "PASS" if ok and found < 90 else ("WARN" if ok else "FAIL"),
           f"{found * dt:.2f} s" if found is not None else "never",
           "back in view < 3 s")

    # --- HOLD freezes the scope
    cam, servo, hb = RCMState(cfg.TROCAR_CAMERA, insertion=0.13), CameraServo(), CommandBus()
    hb.push("HOLD", "test")
    before = (cam.tilt_x, cam.tilt_y)
    for i in range(60):
        c, _ = servo.update(scope.project(_obs(W, H, tgt, index=i), cam),
                            hb.snapshot())
        cam.integrate(c, dt)
    moved = np.hypot(cam.tilt_x - before[0], cam.tilt_y - before[1])
    record("B", "[follow mode] HOLD freezes the camera arm",
           grade(moved, lambda v: v < 1e-9, lambda v: v < 1e-3),
           f"{np.degrees(moved):.4f} deg", "no motion")
    cfg.CAMERA_MODE = mode_saved

    # --- arm 2 (one port, two fingers) puts each finger on its tool tip
    camm = FixedCamera(W, H, fixed_camera_state())
    arm2, ab = InstrumentArm(camm), CommandBus()
    ins = RCMState(cfg.TROCAR_INSTRUMENT,
                   *rcm_for_tip(cfg.TROCAR_INSTRUMENT, working_point()))
    px1 = (W / 2 + 0.15 * W, H / 2 + 0.10 * H)
    px2 = (W / 2 - 0.12 * W, H / 2 - 0.05 * H)
    o2 = _obs(W, H, px1)
    from perception import Detection as _D
    t1 = o2.instrument
    t1.track_id = 1
    t2 = _D(cfg.CLASS_INSTRUMENT, px2, 0.03, contour=None, tip=px2)
    t2.track_id = 2
    o2.instruments = [t1, t2]
    for i in range(240):
        o2.index = i
        ins.integrate(arm2.update(dt, ab.snapshot(), o2, ins), dt)
    arm2.solve_fingers(ins)
    fe = arm2.finger_errors_mm()
    record("B", "Both fingers reach their tool tips",
           grade(max(fe), lambda v: v < 2.0, lambda v: v < 5.0),
           f"finger 1 {fe[0]:.2f} / finger 2 {fe[1]:.2f} mm", "< 2 mm each")

    # --- arm 2 HOLD, and it never leaves the patient
    ab.push("TOOL_HOLD", "test")
    before = (ins.tilt_x, ins.tilt_y, ins.insertion)
    for i in range(60):
        ins.integrate(arm2.update(dt, ab.snapshot(),
                                  _obs(W, H, (100, 100), index=i), ins), dt)
    moved = max(abs(a - b) for a, b in zip(before,
                (ins.tilt_x, ins.tilt_y, ins.insertion)))
    record("B", "'tool hold' keeps arm 2 still",
           grade(moved, lambda v: v < 1e-9, lambda v: v < 1e-4),
           f"{moved:.2e}", "no motion")
    deep = RCMState(cfg.TROCAR_INSTRUMENT, insertion=0.10)
    for _ in range(300):
        deep.integrate({"insert_rate": -1.0}, dt)
    record("B", "Tools always stay inside the patient",
           grade(deep.insertion, lambda v: v >= cfg.INSERTION_LIMITS[0] - 1e-9,
                 lambda v: False),
           f"min depth {deep.insertion * 100:.1f} cm",
           f">= {cfg.INSERTION_LIMITS[0] * 100:.0f} cm even when pulled out")

    # --- fixed camera model is self-consistent (pixel -> 3D -> pixel)
    worst = 0.0
    for u, v in [(200, 150), (960, 540), (1700, 900), (1500, 200)]:
        P = camm.pixel_to_tissue(u, v)
        q = camm.world_to_pixel(P)
        worst = max(worst, np.hypot(q[0] - u, q[1] - v))
    record("B", "Camera model: pixel -> tissue -> pixel",
           grade(worst, lambda v: v < 0.01, lambda v: v < 1.0),
           f"{worst:.2e} px", "round trip exact")

    # --- safety levels at known distances
    ur = {"Ureter": _square(1000, 540, 60)}
    warn_px, danger_px = cfg.WARN_DIST_FRAC * W, cfg.DANGER_DIST_FRAC * W
    cases = [("clear", 1060 + warn_px + 60), ("warn", 1060 + (warn_px + danger_px) / 2),
             ("danger", 1060 + danger_px / 2), ("danger", 1000)]
    got = []
    for expect, x in cases:
        st, _ = SafetyLayer().evaluate(_obs(W, H, (x, 540), ur))
        got.append(st["level"] == expect)
    record("B", "Safety levels at known distances",
           "PASS" if all(got) else "FAIL", f"{sum(got)}/{len(got)} correct",
           "clear / warn / danger / inside")

    # --- predictive time-to-contact
    clock = FakeClock()
    real = control.time.perf_counter
    control.time.perf_counter = clock
    try:
        def approach(speed):
            s, x = SafetyLayer(), 1060 + 0.21 * W
            for i in range(60):
                clock.t += dt
                st, _ = s.evaluate(_obs(W, H, (x, 540), ur, index=i))
                if st.get("predictive"):
                    return st["dist"]
                if st["level"] != "clear":
                    return None
                x -= speed * dt
            return None
        fast, slow = approach(450.0), approach(60.0)
    finally:
        control.time.perf_counter = real
    ok = fast is not None and slow is None
    record("B", "Predictive warning: fast yes, slow no",
           "PASS" if ok else "FAIL",
           f"fast@{fast:.0f}px" if fast else "fast: none",
           f"warns before {warn_px:.0f} px")

    # --- occlusion hold
    sm = _Smoother()
    sm.update("Ureter", Detection("Ureter", (900, 500), 0.02,
                                  contour=_square(900, 500, 40)))
    held = all(sm.update("Ureter", None) is not None
               for _ in range(cfg.OCCLUSION_HOLD_FRAMES))
    expired = sm.update("Ureter", None) is None
    record("B", "Occluded structure held, then expires",
           "PASS" if held and expired else "FAIL",
           f"held {cfg.OCCLUSION_HOLD_FRAMES} frames", "then released")

    # --- arm 2 safety: optional fixture, and the camera-arm guard
    saved_fx = cfg.INSTRUMENT_FIXTURE
    cfg.INSTRUMENT_FIXTURE = True
    st2 = RCMState(cfg.TROCAR_INSTRUMENT,
                   *rcm_for_tip(cfg.TROCAR_INSTRUMENT, working_point()))
    tp = camm.world_to_pixel(st2.pose()[0])
    near = {"Ureter": _square(tp[0] + 0.03 * W, tp[1], 40)}
    toward = rcm_for_tip(cfg.TROCAR_INSTRUMENT, camm.pixel_to_tissue(
        tp[0] + 0.03 * W, tp[1]))
    cmd_in = {"tilt_x_rate": 3 * (toward[0] - st2.tilt_x),
              "tilt_y_rate": 3 * (toward[1] - st2.tilt_y), "insert_rate": 0.0}
    _, why = guard_instrument(cmd_in, st2, dt, _obs(W, H, None, near), camm,
                              1.0, None)
    away = {k: -v for k, v in cmd_in.items()}
    _, why2 = guard_instrument(away, st2, dt, _obs(W, H, None, near), camm,
                               1.0, None)
    cfg.INSTRUMENT_FIXTURE = saved_fx
    record("B", "Fixture (optional) stops arm 2 at the ureter",
           "PASS" if why and not why2 else "FAIL",
           f"toward: {'stopped' if why else 'moved'}, away: "
           f"{'stopped' if why2 else 'moved'}", "stop toward, free away")
    cam_tip = fixed_camera_state().pose()[0]
    to_cam = rcm_for_tip(cfg.TROCAR_INSTRUMENT, cam_tip)
    c_in = {"tilt_x_rate": 3 * (to_cam[0] - st2.tilt_x),
            "tilt_y_rate": 3 * (to_cam[1] - st2.tilt_y), "insert_rate": 0.0}
    _, w1 = guard_instrument(c_in, st2, dt, None, camm, 0.01, cam_tip)
    _, w2 = guard_instrument({k: -v for k, v in c_in.items()}, st2, dt, None,
                             camm, 0.01, cam_tip)
    record("B", "Arms too close: arm 2 stops, never retracts",
           "PASS" if w1 and not w2 else "FAIL",
           f"toward camera: {'stopped' if w1 else 'moved'}, away: "
           f"{'stopped' if w2 else 'moved'}", "stop toward, free away")

    # --- voice grammar (wake word "robot"; "stop" works on its own)
    w = cfg.VOICE_WAKE_WORD
    tests = {f"{w} zoom in": "ZOOM_IN", f"{w} zoom out": "ZOOM_OUT",
             f"{w} left": "PAN_LEFT", f"{w} up": "PAN_UP",
             f"{w} centre": "CENTER", f"{w} follow": "CAM_FOLLOW",
             f"{w} freeze": "CAM_HOLD", f"{w} reset": "HOME",
             f"{w} tool follow": "TOOL_FOLLOW", f"{w} tool hold": "TOOL_HOLD",
             f"{w} picture": "SNAPSHOT", f"{w} light up": "LIGHT_UP",
             f"{w} light down": "LIGHT_DOWN", f"{w} mark": "MARK",
             "stop": "STOP", "hello there": None,
             "zoom in": None if w else "ZOOM_IN",       # no wake word
             f"the {w} is fine": None}
    right = sum(match_phrase(k) == v for k, v in tests.items())
    record("B", "Voice phrases map to commands",
           "PASS" if right == len(tests) else "FAIL",
           f"{right}/{len(tests)}", "wake word + rejects non-commands")

    # --- voice camera: zoom / pan / follow / home move the view AND the arm
    from camview import CameraVoice
    camm2 = FixedCamera(W, H, fixed_camera_state())
    cv_ = CameraVoice(camm2, fixed_camera_state())
    arm_s = fixed_camera_state()
    home_ins = arm_s.insertion
    tgt_px = (W * 0.70, H * 0.35)
    ob = _obs(W, H, tgt_px)
    cv_.command("ZOOM_IN", ob)
    cv_.command("ZOOM_IN", ob)
    cv_.command("PAN_LEFT", ob)
    for i in range(90):
        cv_.update(dt, ob)
        cv_.drive_arm(arm_s, dt)
    x0, y0, x1, y1 = cv_.crop()
    zoom_ok = abs((1 / (x1 - x0)) - 1.25 ** 2) < 0.05
    pan_ok = (x0 + x1) / 2 < 0.5
    lo, hi = cfg.INSERTION_LIMITS
    arm_ok = arm_s.insertion > home_ins + 0.005 and lo <= arm_s.insertion <= hi
    record("B", "Voice zoom / pan move the view and the camera arm",
           "PASS" if zoom_ok and pan_ok and arm_ok else "FAIL",
           f"zoom {1 / (x1 - x0):.2f}x, centre x {(x0 + x1) / 2:.2f}, "
           f"scope +{(arm_s.insertion - home_ins) * 100:.1f} cm",
           "1.56x, moved left, deeper but in limits")
    cv_.command("CAM_FOLLOW", ob)
    for i in range(240):
        cv_.update(dt, ob)
    cx = (cv_.crop()[0] + cv_.crop()[2]) / 2 * W
    cy = (cv_.crop()[1] + cv_.crop()[3]) / 2 * H
    err = np.hypot(cx - tgt_px[0], cy - tgt_px[1]) / W
    record("B", "Voice 'follow' keeps the tool in the middle",
           grade(err, lambda v: v < 0.03, lambda v: v < 0.08),
           f"{err * 100:.1f} % of width off-centre", "< 3 %")
    cv_.command("HOME", ob)
    for i in range(150):
        cv_.update(dt, ob)
        cv_.drive_arm(arm_s, dt)
    back = abs(arm_s.insertion - home_ins) < 1e-3 and cv_.is_home()
    record("B", "Voice 'home' returns view and camera arm",
           "PASS" if back else "FAIL",
           f"insertion {arm_s.insertion * 100:.1f} cm (home "
           f"{home_ins * 100:.1f})", "back at start")
    import display as _disp
    img0 = np.full((40, 40, 3), 100, np.uint8)
    cv_.command("LIGHT_UP", ob)
    cv_.command("LIGHT_UP", ob)
    up = _disp._light(img0, cv_.light).mean()
    for _ in range(6):
        cv_.command("LIGHT_DOWN", ob)
    dn = _disp._light(img0, cv_.light).mean()
    record("B", "Voice 'light up / down' brightens / dims the picture",
           "PASS" if up > 110 and dn < 90 and cv_.light >= 0.5 else "FAIL",
           f"grey 100 -> {up:.0f} (up) / {dn:.0f} (down)", "brighter, darker")
    sb = CommandBus()
    sb.push("STOP", "test")
    snap = sb.snapshot()
    record("B", "'stop' freezes both arms",
           "PASS" if snap["mode"] == "HOLD" and snap["instrument"] == "HOLD"
           else "FAIL", f"camera {snap['mode']}, arm 2 {snap['instrument']}",
           "both HOLD")


# ======================================================== C. perception
def section_c(args):
    print("\nC. PERCEPTION (your model on your video)")
    model_path = args.model or cfg.MODEL_PATH
    video_path = args.video or cfg.VIDEO_PATH
    if not (os.path.isfile(model_path) and os.path.isfile(video_path)):
        record("C", "Model + video present", "SKIP", "not found",
               f"{model_path}, {video_path}")
        return
    try:
        import cv2
        from ultralytics import YOLO
        from perception import instrument_tip, precision_kwargs, estimate_field
        from endoscope import VirtualEndoscope
        from rcm import RCMState
    except ImportError as e:
        record("C", "Imports", "SKIP", "missing", str(e))
        return

    model = YOLO(model_path)
    names = {i: n.lower() for i, n in model.names.items()}
    expected = [cfg.CLASS_INSTRUMENT] + cfg.CLASS_CRITICAL + cfg.CLASS_SOFT
    missing = [c for c in expected if c.lower() not in names.values()]
    record("C", "Model has all 8 expected classes",
           "PASS" if not missing else "FAIL",
           f"{len(expected) - len(missing)}/{len(expected)}",
           "names match config", ", ".join(missing))
    inst_id = next((i for i, n in names.items()
                    if n == cfg.CLASS_INSTRUMENT.lower()), None)
    if inst_id is None:
        return

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 10000
    step = max(1, total // args.frames)
    half = bool(cfg.HALF) and str(cfg.DEVICE) != "cpu"

    frames = hits = 0
    confs, areas, tips, lat, per_class = [], [], [], [], {}
    W = H = None
    idx = 0
    prec = precision_kwargs(half)
    field = None
    print(f"     sampling {args.frames} of {total} frames on device "
          f"{cfg.DEVICE} (no window; add --show to watch detections)")
    while frames < args.frames:
        # grab() skips a frame WITHOUT decoding it. Decoding every frame of a
        # 2 GB 1080p clip just to throw most away took minutes and looked
        # like a hang.
        if idx % step:
            if not cap.grab():
                break
            idx += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if frames and frames % 25 == 0:
            print(f"     ... {frames}/{args.frames} frames", flush=True)
        H, W = frame.shape[:2]
        t0 = time.perf_counter()
        r = model.predict(frame, imgsz=cfg.IMGSZ, conf=cfg.CONF,
                          device=cfg.DEVICE, verbose=False,
                          **prec)[0]
        lat.append((time.perf_counter() - t0) * 1000)
        frames += 1
        
        if args.show: cv2.imshow("Perception Test", r.plot())
        if args.show: cv2.waitKey(1)
        
        if r.boxes is None or len(r.boxes.cls) == 0:
            tips.append(None)
            continue
        cls = r.boxes.cls.cpu().numpy().astype(int)
        cf = r.boxes.conf.cpu().numpy()
        for c in set(cls.tolist()):
            per_class[names[c]] = per_class.get(names[c], 0) + 1
        sel = np.where(cls == inst_id)[0]
        if sel.size == 0 or r.masks is None:
            tips.append(None)
            continue
        best = sel[np.argmax(cf[sel])]
        hits += 1
        confs.append(float(cf[best]))
        poly = np.asarray(r.masks.xy[best], np.float32)
        areas.append(abs(cv2.contourArea(poly)) / (W * H) if len(poly) > 2 else 0)
        if field is None:
            field = estimate_field(frame)
        tips.append(instrument_tip(poly.astype(np.int32).reshape(-1, 1, 2),
                                   frame.shape, field)
                    if len(poly) > 2 else None)
    cap.release()
    cv2.destroyAllWindows()
    if frames == 0:
        record("C", "Video readable", "FAIL", "0 frames", video_path)
        return

    rate = hits / frames
    record("C", "Instrument detection rate",
           grade(rate, lambda v: v >= 0.85, lambda v: v >= 0.60),
           f"{100 * rate:.0f}% of {frames}", ">= 85%")
    if confs:
        mc = float(np.mean(confs))
        record("C", "Instrument confidence (mean)",
               grade(mc, lambda v: v >= 0.60, lambda v: v >= 0.40),
               f"{mc:.2f}", ">= 0.60")
    if len(areas) > 2:
        a = np.array(areas)
        flick = float(np.median(np.abs(np.diff(a)) / np.maximum(a[:-1], 1e-6)))
        record("C", "Mask area stability (frame-to-frame)",
               grade(flick, lambda v: v < 0.15, lambda v: v < 0.35),
               f"{100 * flick:.0f}% change", "< 15%",
               "high = flickering mask -> zoom/tip noise")
    jumps = [np.hypot(a[0] - b[0], a[1] - b[1])
             for a, b in zip(tips[:-1], tips[1:]) if a and b]
    if jumps:
        mj = float(np.median(jumps)) / W
        record("C", "Tip jump between samples",
               grade(mj, lambda v: v < 0.03, lambda v: v < 0.08),
               f"{100 * mj:.1f}% of width", "< 3%",
               "includes real motion; big = tip flips ends")
    lat = np.array(lat[3:] if len(lat) > 5 else lat)
    p95 = float(np.percentile(lat, 95))
    record("C", "Inference latency (95th pct)",
           grade(p95, lambda v: v <= 60, lambda v: v <= 120),
           f"{np.mean(lat):.0f} / {p95:.0f} ms", "p95 <= 60 ms")

    # Is a high tracking error a detection problem, or a geometry limit?
    scope = VirtualEndoscope(W, H)
    st = RCMState(cfg.TROCAR_CAMERA, insertion=0.13)
    half_w = 0.5 * scope._scale(st) * W
    half_h = half_w * (scope.out_h / scope.out_w)
    ov_w, ov_h = cfg.MAX_OVERHANG * 2 * half_w, cfg.MAX_OVERHANG * 2 * half_h
    pan = cfg.PAN_FRAC * W
    u_lo, u_hi = max(half_w - ov_w, W / 2 - pan), min(W - half_w + ov_w, W / 2 + pan)
    v_lo, v_hi = max(half_h - ov_h, H / 2 - pan), min(H - half_h + ov_h, H / 2 + pan)
    floor = []
    for t in tips:
        if t is None:
            continue
        ex = max(u_lo - t[0], 0, t[0] - u_hi) / half_w
        ey = max(v_lo - t[1], 0, t[1] - v_hi) / half_h
        floor.append(np.hypot(ex, ey))
    if floor:
        f = np.array(floor)
        cent = float(np.mean(f < 1e-9))
        record("C", "Instrument positions the scope CAN centre",
               grade(cent, lambda v: v >= 0.8, lambda v: v >= 0.5),
               f"{100 * cent:.0f}%", ">= 80%",
               f"unavoidable error floor mean {f.mean():.2f}")
    print("     classes seen (frames):",
          ", ".join(f"{k} {v}" for k, v in sorted(per_class.items())))


# ======================================================== D. simulation
def section_d():
    print("\nD. PYBULLET SIMULATION")
    try:
        import pybullet as p
        from scene import Scene
        from rcm import RCMState
    except Exception as e:                                  # noqa: BLE001
        record("D", "PyBullet available", "SKIP", "not importable", str(e))
        return

    sc = Scene(gui=False)

    def settle(name, st, n=160):
        diag = None
        for _ in range(n):
            diag = sc.drive(name, st)
            sc.step(4)
        return diag

    for name, pivot in [("camera", cfg.TROCAR_CAMERA),
                        ("instrument", cfg.TROCAR_INSTRUMENT)]:
        res, dev = [], []
        for tx, ty in [(0, 0), (0.35, 0), (-0.35, 0.2), (0, -0.4), (0.3, 0.3)]:
            dg = settle(name, RCMState(pivot, tilt_x=tx, tilt_y=ty, insertion=0.13))
            res.append(dg["ik_residual_m"])
            dev.append(dg["pivot_error_m"])
        r, d = max(res) * 1000, max(dev) * 1000
        record("D", f"{name} arm reaches commanded grip point",
               grade(r, lambda v: v < 5, lambda v: v < 15), f"{r:.1f} mm",
               "IK error < 5 mm")
        record("D", f"{name} arm holds the trocar (physical RCM)",
               grade(d, lambda v: v < 3, lambda v: v < 10), f"{d:.1f} mm",
               "tool axis within 3 mm of port")

    # Same start poses as main.py: fixed camera, instrument at the centre of
    # the operative field.
    from camera_model import fixed_camera_state, rcm_for_tip, working_point
    settle("camera", fixed_camera_state(), 120)
    settle("instrument", RCMState(cfg.TROCAR_INSTRUMENT, *rcm_for_tip(
        cfg.TROCAR_INSTRUMENT, working_point())), 120)
    c = sc.arm_clearance()
    record("D", "Arms clear of each other at the start pose",
           grade(c, lambda v: v > cfg.ARM_MIN_CLEARANCE,
                 lambda v: v > 0.0), f"{c * 100:.1f} cm",
           f"> {cfg.ARM_MIN_CLEARANCE * 100:.0f} cm")

    z_in = sc.flange_rcm_deviation("instrument")[0][2]
    settle("instrument", RCMState(cfg.TROCAR_INSTRUMENT,
                                  insertion=cfg.INSERTION_LIMITS[0]), 160)
    z_out = sc.flange_rcm_deviation("instrument")[0][2]
    record("D", "Changing tool depth really moves the arm",
           grade(z_out - z_in, lambda v: v > 0.04, lambda v: v > 0.01),
           f"+{(z_out - z_in) * 100:.1f} cm", "flange rises > 4 cm")
    sc.disconnect()


# ======================================================== E. accuracy
def section_e(seconds=40.0):
    """
    End-to-end accuracy of the whole robot, in closed loop, on the two-tool
    synthetic scene (ground truth known exactly): perception -> tracker ->
    arm 2 follows the active tool through the FIXED camera -> RCM -> IK ->
    simulated arms. Same control-loop order as main.py, no threads.
    """
    print("\nE. CLOSED-LOOP ACCURACY (two-tool synthetic scene, "
          f"{seconds:.0f} s)")
    try:
        from scene import Scene
    except Exception as e:                                  # noqa: BLE001
        record("E", "PyBullet available", "SKIP", "not importable", str(e))
        return
    from perception import DemoPerception
    from control import SafetyLayer, InstrumentArm, guard_instrument
    from rcm import RCMState
    from commands import CommandBus
    from camera_model import (FixedCamera, fixed_camera_state, rcm_for_tip,
                              working_point)
    import cv2

    src = DemoPerception(w=1280, h=720)
    W, H = src.w, src.h
    sc = Scene(gui=False)
    cam = fixed_camera_state()
    ins = RCMState(cfg.TROCAR_INSTRUMENT,
                   *rcm_for_tip(cfg.TROCAR_INSTRUMENT, working_point()))
    sc.place("camera", cam)
    sc.place("instrument", ins)
    camm = FixedCamera(W, H, cam)
    arm2, safety, bus = InstrumentArm(camm), SafetyLayer(), CommandBus()
    dt = 1.0 / cfg.CONTROL_HZ
    sub = max(1, cfg.SIM_HZ // cfg.CONTROL_HZ)
    cam0 = (cam.tilt_x, cam.tilt_y, cam.insertion)

    settle = int(2.0 / dt)
    tip_err, both, follow, rcm_c, rcm_s, grip_s = [], [], [], [], [], []
    clear, stops = [], []
    lvl_ok = lvl_n = 0
    id_of, swaps = {}, 0
    clearance = cfg.ARM_CLEAR_QUERY
    for i in range(int(seconds / dt)):
        tt = i * dt
        obs = src.make(tt, i)
        truth = src.true_tips(tt)
        cmd = bus.snapshot()
        status, _ = safety.evaluate(obs)
        c2 = arm2.update(dt, cmd, obs, ins)
        c2, why = guard_instrument(c2, ins, dt, obs, camm, clearance,
                                   cam.pose()[0])
        ins.integrate(c2, dt)
        dc = sc.drive("camera", cam)
        ds = sc.drive("instrument", ins)
        sc.step(sub)
        clearance = sc.arm_clearance()
        if i < settle:
            continue
        both.append(len(obs.instruments) >= 2)
        for k, (gx, gy) in enumerate(truth):
            if not obs.instruments:
                continue
            near = min(obs.instruments,
                       key=lambda d: np.hypot(d.tip[0] - gx, d.tip[1] - gy))
            tip_err.append(np.hypot(near.tip[0] - gx, near.tip[1] - gy) / W)
            if k not in id_of:
                id_of[k] = near.track_id
            elif id_of[k] != near.track_id:
                swaps += 1
                id_of[k] = near.track_id
        # each finger jaw vs. the TRUE tip of the tool it follows (mm)
        arm2.solve_fingers(ins)
        for k in range(2):
            T = arm2.targets[k]
            if T is None or arm2.ends[k] is None:
                continue
            tpx = camm.world_to_pixel(T)
            gt = min(truth, key=lambda q: np.hypot(q[0] - tpx[0],
                                                   q[1] - tpx[1]))
            G = camm.pixel_to_tissue(*gt, cfg.TISSUE_Z + cfg.INSTRUMENT_HOVER)
            follow.append(1000 * np.linalg.norm(arm2.ends[k] - G))
        rcm_c.append(dc["pivot_error_m"] * 1000)
        rcm_s.append(ds["pivot_error_m"] * 1000)
        grip_s.append(ds["ik_residual_m"] * 1000)
        clear.append(clearance)
        stops.append(why is not None)
        ur = obs.critical.get("Ureter")
        if ur is not None:
            dtrue = min(max(0.0, -cv2.pointPolygonTest(
                ur.contour, (float(x), float(y)), True)) for x, y in truth)
            want = ("danger" if dtrue <= cfg.DANGER_DIST_FRAC * W else
                    "warn" if dtrue <= cfg.WARN_DIST_FRAC * W else "clear")
            got = status["level"]
            lvl_n += 1
            lvl_ok += int(got == want or (want == "clear" and got == "warn"
                                          and status.get("predictive")))
    sc.disconnect()
    cam_moved = max(abs(a - b) for a, b in zip(
        cam0, (cam.tilt_x, cam.tilt_y, cam.insertion)))

    def p95(a):
        return float(np.percentile(a, 95)) if len(a) else float("nan")

    record("E", "Both tools tracked", grade(np.mean(both), lambda v: v >= 0.98,
                                           lambda v: v >= 0.9),
           f"{100 * np.mean(both):.1f}% of frames", ">= 98%")
    record("E", "Tool-tip localisation error (p95)",
           grade(p95(tip_err), lambda v: v < 0.03, lambda v: v < 0.06),
           f"{100 * p95(tip_err):.2f}% of width", "< 3% (smoothing lag incl.)")
    record("E", "Tool identity swaps", grade(swaps, lambda v: v == 0,
                                            lambda v: v <= 2),
           f"{swaps}", "0 (Tool 1 stays Tool 1)")
    record("E", "Camera arm stays fixed",
           grade(cam_moved, lambda v: v < 1e-9, lambda v: v < 1e-4),
           f"{cam_moved:.1e}", "no motion")
    record("E", "Fingers follow the true tool tips (median / p95)",
           grade(float(np.median(follow)), lambda v: v < 5, lambda v: v < 10),
           f"{np.median(follow):.1f} / {p95(follow):.1f} mm",
           "median < 5 mm (moving tool)")
    for name, rc in (("camera", rcm_c), ("instrument", rcm_s)):
        record("E", f"{name} arm RCM deviation (p95 / max)",
               grade(p95(rc), lambda v: v < 2, lambda v: v < 5),
               f"{p95(rc):.2f} / {max(rc):.2f} mm", "p95 < 2 mm at the port")
    record("E", "instrument arm grip tracking (p95)",
           grade(p95(grip_s), lambda v: v < 5, lambda v: v < 15),
           f"{p95(grip_s):.2f} mm", "p95 < 5 mm (includes motion lag)")
    record("E", "Arm-to-arm clearance (min)",
           grade(min(clear), lambda v: v > cfg.ARM_MIN_CLEARANCE,
                 lambda v: v > 0.0),
           f"{100 * min(clear):.1f} cm  stopped {100 * np.mean(stops):.0f}%",
           f"> {cfg.ARM_MIN_CLEARANCE * 100:.0f} cm")
    if lvl_n:
        record("E", "Safety warning level matches ground truth",
               grade(lvl_ok / lvl_n, lambda v: v >= 0.95, lambda v: v >= 0.85),
               f"{100 * lvl_ok / lvl_n:.1f}%", ">= 95% (any tool, either hand)")


# ================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video")
    ap.add_argument("--model")
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--no-sim", action="store_true")
    ap.add_argument("--accuracy-seconds", type=float, default=40.0,
                    help="length of the closed-loop accuracy run (section E)")
    ap.add_argument("--show", action="store_true",
                    help="show YOLO detections while section C runs")
    args = ap.parse_args()

    section_a()
    section_b()
    section_c(args)
    if not args.no_sim:
        section_d()
        section_e(args.accuracy_seconds)

    n = {k: sum(1 for r in RESULTS if r[2] == k)
         for k in ("PASS", "WARN", "FAIL", "SKIP")}
    print(f"\nRESULT: {n['PASS']} pass, {n['WARN']} warn, {n['FAIL']} fail, "
          f"{n['SKIP']} skipped")

    os.makedirs("runs", exist_ok=True)
    with open(os.path.join("runs", "selftest.md"), "w") as f:
        f.write("# Self-test\n\n| Section | Check | Result | Measured | "
                "Criterion | Note |\n|---|---|---|---|---|---|\n")
        for sec, name, st, m, c, note in RESULTS:
            f.write(f"| {sec} | {name} | {st} | {m} | {c} | {note} |\n")
    print("saved runs/selftest.md")
    sys.exit(1 if n["FAIL"] else 0)


if __name__ == "__main__":
    main()
