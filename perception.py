"""
Perception runs on its own thread with a single-slot queue that drops stale
frames. If you run inference inline with the sim loop, the simulation stutters
down to inference speed and everything feels broken.

Two backends:
    YoloPerception    real YOLO26-seg over a laparoscopic video
    DemoPerception    synthetic moving target, no model or video needed —
                      use this on Day 1 and 2 to test kinematics
"""
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

import config as cfg


@dataclass
class Detection:
    name: str
    centroid: tuple          # (x, y) in pixels
    area_frac: float         # mask area / frame area
    contour: np.ndarray = None
    tip: tuple = None        # distal end of the instrument, not the centroid
    stale_frames: int = 0    # >0 means this is a held, occluded estimate
    track_id: int = 0        # instruments only: stable "Tool n" label
    activity: float = 0.0    # instruments only: smoothed tip speed, px/frame
    conf: float = 0.0
    ends: tuple = None       # both extremes of the tool's principal axis
    tip_margin: float = 1e9  # how clearly one end is the tip (px)
    back: tuple = None       # point on the shaft behind the tip (drawn axis)
    axis_quality: float = 0.0  # >0: line snapped to the tool's real pixels
    tip_visible: bool = True # False: both ends touch the picture edge, i.e.
                             # the shaft crosses the view and the jaws are
                             # out of frame - no end effector is drawn

    @property
    def work_point(self):
        """Tip when it is in view, otherwise the mask centroid."""
        if self.tip is not None and self.tip_visible:
            return self.tip
        return self.centroid


@dataclass
class Observation:
    frame: np.ndarray
    instrument: Detection = None
    critical: dict = field(default_factory=dict)   # name -> Detection
    infer_ms: float = 0.0
    index: int = 0
    # Every tracked instrument (up to cfg.MAX_INSTRUMENTS). `instrument`
    # above is the PRIMARY one - the tool the surgeon is actively working
    # with - kept as a single field so the controllers stay simple.
    instruments: list = field(default_factory=list)
    camera_target: tuple = None   # where the camera arm aims (full-frame px)


def aim_point(obs):
    """The point the camera arm should centre, in full-frame pixels."""
    if obs is None:
        return None
    if obs.camera_target is not None:
        return obs.camera_target
    if obs.instrument is not None:
        return obs.instrument.work_point
    return None


def precision_kwargs(half):
    """
    FP16 flag for model.predict. Ultralytics >= 8.4 replaced half=True with
    quantize=16 and prints a deprecation warning on EVERY frame for the old
    argument, which floods the terminal.
    """
    if not half:
        return {}
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT
        if "quantize" in DEFAULT_CFG_DICT:
            return {"quantize": 16}
    except Exception:                                       # noqa: BLE001
        pass
    return {"half": True}


class _Smoother:
    """Exponential smoothing on centroids AND area, plus occlusion hold."""

    def __init__(self):
        self.last = {}
        self.missing = {}

    def update(self, name, det):
        a = cfg.CENTROID_SMOOTH
        if det is not None:
            prev = self.last.get(name)
            if prev is not None:
                cx = a * prev.centroid[0] + (1 - a) * det.centroid[0]
                cy = a * prev.centroid[1] + (1 - a) * det.centroid[1]
                det.centroid = (cx, cy)
                # Area MUST be smoothed too. It drives the zoom axis, and raw
                # per-frame mask area swings wildly with segmentation flicker —
                # unsmoothed it pumps the insertion joint and the view breathes.
                b = cfg.AREA_SMOOTH
                det.area_frac = b * prev.area_frac + (1 - b) * det.area_frac
                if prev.tip is not None and det.tip is not None:
                    det.tip = (a * prev.tip[0] + (1 - a) * det.tip[0],
                               a * prev.tip[1] + (1 - a) * det.tip[1])
            det.stale_frames = 0
            self.last[name] = det
            self.missing[name] = 0
            return det

        # Occluded. Hold the last estimate rather than dropping the constraint.
        n = self.missing.get(name, 0) + 1
        self.missing[name] = n
        if n <= cfg.OCCLUSION_HOLD_FRAMES and name in self.last:
            held = self.last[name]
            held.stale_frames = n
            return held
        self.last.pop(name, None)
        return None


def split_tools(mask, max_tools=2, rng=None):
    """
    Split one instrument mask that actually contains several tools.

    The model was trained on labels made from SEMANTIC masks: every
    instrument pixel had the same colour and each connected patch became one
    "instance". Two tools that touch or cross in a training frame were
    therefore labelled as ONE object, and the model reproduces that - a
    single mask spanning both tools, whose axis and tip land between them.

    Laparoscopic instruments are straight rigid shafts. Fit the dominant
    straight band through the mask (RANSAC on its pixels, band width from the
    mask's own thickness), remove it, and look for a second band. If a second
    shaft is found at a clearly different angle, pixels are assigned to the
    nearer line. Returns a list of binary masks (one per tool).
    """
    ys, xs = np.nonzero(mask)
    n = len(xs)
    if n < 200 or max_tools < 2:
        return [mask]
    rng = rng or np.random.default_rng(0)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    # shaft half-width from the distance transform (thickness of the mask)
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    half = max(2.0, float(np.percentile(dt[mask > 0], 90)))
    tol = 1.3 * half

    def ransac(p, iters=120):
        best, best_in = None, None
        m = len(p)
        for _ in range(iters):
            i, j = rng.integers(0, m, 2)
            d = p[j] - p[i]
            L = float(np.hypot(*d))
            if L < 4 * half:
                continue
            nrm = np.array([-d[1], d[0]]) / L
            inl = np.abs((p - p[i]) @ nrm) < tol
            if best_in is None or inl.sum() > best_in.sum():
                best, best_in = (p[i], d / L, nrm), inl
        return best, best_in

    line1, in1 = ransac(pts)
    if line1 is None:
        return [mask]
    rest = pts[~in1]
    if len(rest) < cfg.SPLIT_MIN_FRAC * n:
        return [mask]
    line2, in2 = ransac(rest)
    if line2 is None or in2.sum() < cfg.SPLIT_MIN_FRAC * n:
        return [mask]
    ang = np.degrees(np.arccos(min(1.0, abs(float(line1[1] @ line2[1])))))
    if ang < cfg.SPLIT_MIN_ANGLE:
        return [mask]                    # parallel bands: one fat tool
    d1 = np.abs((pts - line1[0]) @ line1[2])
    d2 = np.abs((pts - line2[0]) @ line2[2])
    lab = (d2 < d1)
    out = []
    for k in (False, True):
        m = np.zeros_like(mask)
        sel = pts[lab == k].astype(int)
        m[sel[:, 1], sel[:, 0]] = 1
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        if m.sum() >= cfg.SPLIT_MIN_FRAC * n:
            out.append(m)
    return out or [mask]


class InstrumentTracker:
    """
    Tracks EVERY instrument in view, not just the largest mask.

    The earlier pipeline kept only the largest instrument instance per frame,
    so with two tools in the field the second was never drawn, never checked
    against the ureter and never considered by the camera. Here each
    detection is matched to an existing track by nearest centroid (a two-tool
    scene needs nothing fancier than greedy matching), so "Tool 1" stays
    Tool 1 from frame to frame; each track is smoothed and occlusion-held on
    its own.

    Primary tool = the one whose tip has been moving most (the surgeon's
    working hand), with hysteresis so it does not flicker between tools.
    The camera aims at the midpoint of all tool tips (cfg.CAMERA_AIM), which
    is how a human camera assistant frames a two-handed task.
    """

    def __init__(self):
        self.tracks = {}         # id -> Detection (smoothed)
        self.missing = {}
        self.primary_id = None

    def _new_id(self):
        # Re-use the lowest free number so labels stay "Tool 1"/"Tool 2".
        i = 1
        while i in self.tracks:
            i += 1
        return i

    def update(self, dets, shape):
        h, w = shape[:2]
        gate = cfg.TRACK_GATE_FRAC * w
        a, b = cfg.CENTROID_SMOOTH, cfg.AREA_SMOOTH
        free = dict(self.tracks)
        matched = {}
        # greedy nearest-tip assignment
        pairs = []
        for k, d in enumerate(dets):
            # Match on the mask CENTROID: it moves smoothly, whereas the tip
            # estimate can hop to the other end of the shaft in one frame.
            dp = np.array(d.centroid)
            for tid, t in free.items():
                tp = np.array(t.centroid)
                pairs.append((float(np.linalg.norm(dp - tp)), k, tid))
        pairs.sort()
        used_d, used_t = set(), set()
        for dist, k, tid in pairs:
            if k in used_d or tid in used_t or dist > gate:
                continue
            used_d.add(k)
            used_t.add(tid)
            matched[tid] = dets[k]

        for tid, d in matched.items():
            prev = self.tracks[tid]
            if d.ends is not None and prev.tip is not None and \
                    d.tip_margin < cfg.TIP_AMBIGUOUS_FRAC * w:
                # Both ends about equally far from the picture edge (tool
                # lying across the field): don't let the tip jump to the
                # other end of the shaft - keep the end nearest the last tip.
                d.tip = min(d.ends, key=lambda e: np.hypot(
                    e[0] - prev.tip[0], e[1] - prev.tip[1]))
            new_tip = d.tip
            old_tip = prev.tip
            d.centroid = (a * prev.centroid[0] + (1 - a) * d.centroid[0],
                          a * prev.centroid[1] + (1 - a) * d.centroid[1])
            d.area_frac = b * prev.area_frac + (1 - b) * d.area_frac
            if old_tip is not None and new_tip is not None and np.hypot(
                    new_tip[0] - old_tip[0], new_tip[1] - old_tip[1]) > \
                    cfg.TIP_SNAP_FRAC * w:
                # A jump this big in one frame is not motion: the tip estimate
                # moved to the other end of the shaft (or the mask changed
                # shape). Snap instead of easing, or the drawn tip slides
                # along the shaft for a second.
                speed = 0.0
            elif old_tip is not None and new_tip is not None:
                d.tip = (a * old_tip[0] + (1 - a) * new_tip[0],
                         a * old_tip[1] + (1 - a) * new_tip[1])
                speed = float(np.hypot(d.tip[0] - old_tip[0],
                                       d.tip[1] - old_tip[1]))
            else:
                speed = 0.0
            if d.back is not None and new_tip is not None and d.tip is not None:
                # keep the drawn axis attached to the smoothed tip
                d.back = (d.back[0] + d.tip[0] - new_tip[0],
                          d.back[1] + d.tip[1] - new_tip[1])
            d.activity = 0.9 * prev.activity + 0.1 * speed
            d.track_id = tid
            d.stale_frames = 0
            self.tracks[tid] = d
            self.missing[tid] = 0

        for k, d in enumerate(dets):
            if k in used_d:
                continue
            if len(self.tracks) >= cfg.MAX_INSTRUMENTS:
                break
            tid = self._new_id()
            d.track_id = tid
            self.tracks[tid] = d
            self.missing[tid] = 0

        for tid in list(self.tracks):
            if tid in matched or tid not in free:   # seen, or created now
                continue
            n = self.missing.get(tid, 0) + 1
            self.missing[tid] = n
            if n > cfg.INSTRUMENT_HOLD_FRAMES:
                self.tracks.pop(tid)
                self.missing.pop(tid)
            else:
                self.tracks[tid].stale_frames = n

        # primary = most active tool, switch only with a clear margin
        live = [t for t in self.tracks.values()]
        if not live:
            self.primary_id = None
            return [], None, None
        best = max(live, key=lambda t: (t.activity, t.area_frac))
        cur = self.tracks.get(self.primary_id)
        if cur is None or best.activity > cfg.PRIMARY_SWITCH_RATIO * \
                max(cur.activity, 0.5):
            self.primary_id = best.track_id
        primary = self.tracks[self.primary_id]

        tools = sorted(live, key=lambda t: t.track_id)
        fresh = [t for t in tools if t.stale_frames <= 3]
        if cfg.CAMERA_AIM == "midpoint" and len(fresh) >= 2:
            pts = np.array([t.work_point for t in fresh], float)
            target = tuple(pts.mean(axis=0))
        else:
            target = primary.work_point
        return tools, primary, target


def model_label(model_path):
    """
    Human name of the network inside the weights, e.g. 'YOLO26s-seg', read
    from the checkpoint itself (so the console never shows a wrong name).
    A TensorRT engine has no architecture info: its source .pt is read.
    """
    import re
    src = str(model_path)
    trt = src.endswith(".engine")
    if trt:
        src = cfg.PT_WEIGHTS
    name = None
    try:
        import torch
        ck = torch.load(src, map_location="cpu", weights_only=False)
        m = ck.get("model") or ck.get("ema")
        y = getattr(m, "yaml", {}) or {}
        name = y.get("yaml_file") or (ck.get("train_args") or {}).get("model")
    except Exception:                                       # noqa: BLE001
        pass
    if not name:
        return "YOLO-seg" + ("  TensorRT" if trt else "")
    base = re.sub(r"\.(ya?ml|pt)$", "", str(name).split("/")[-1])
    m = re.match(r"yolo(v?)(\d+)([nslmx]?)(-seg)?", base, re.I)
    if m:
        ver = ("v" if m.group(1) else "") + m.group(2)
        base = f"YOLO{ver}{m.group(3)}{m.group(4) or ''}"
    return base + ("  TensorRT" if trt else "")


class _BaseSource(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self._slot = None
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self.fps = 0.0
        self.finished = False           # set when a non-looping video ends
        self.error = None               # set if the worker thread crashed
        # Controlled degradation, for robustness experiments (--drop/--jitter).
        self.drop_prob = 0.0
        self.jitter_px = 0.0
        self._rng = np.random.default_rng(0)
        self.tracker = InstrumentTracker()

    def _degrade(self, raw):
        """
        Simulate worse perception: randomly drop instrument detections and
        jitter their positions. Applied BEFORE smoothing, like a real missed or
        noisy detection. Seeded, so experiment runs are repeatable.
        """
        kept = []
        for inst in raw.get("instruments", []):
            if self.drop_prob > 0 and self._rng.random() < self.drop_prob:
                continue
            if self.jitter_px > 0:
                j = self._rng.normal(0, self.jitter_px, 2)
                inst.centroid = (inst.centroid[0] + j[0],
                                 inst.centroid[1] + j[1])
                if inst.tip is not None:
                    inst.tip = (inst.tip[0] + j[0], inst.tip[1] + j[1])
            kept.append(inst)
        raw["instruments"] = kept
        return raw

    def _apply_tracker(self, obs, raw):
        tools, primary, target = self.tracker.update(
            raw.get("instruments", []), obs.frame.shape)
        obs.instruments = tools
        obs.instrument = primary
        obs.camera_target = target

    def _publish(self, obs):
        with self._lock:
            self._slot = obs      # single slot: newest wins, old frame dropped

    def latest(self):
        with self._lock:
            return self._slot

    def stop(self):
        self._stop_evt.set()


class YoloPerception(_BaseSource):
    def __init__(self, model_path=None, video_path=None, loop=True):
        model_path = model_path or cfg.MODEL_PATH
        video_path = video_path or cfg.VIDEO_PATH
        super().__init__()
        from ultralytics import YOLO          # imported lazily
        self.model = YOLO(model_path, task="segment")
        self.names = {i: n for i, n in self.model.names.items()}
        self.model_label = model_label(model_path)
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        self.loop = loop
        self.smoother = _Smoother()
        # FP16 roughly halves inference time on an RTX GPU; CPU cannot use it.
        self.half = bool(cfg.HALF) and str(cfg.DEVICE) != "cpu"
        # A TensorRT engine already has its precision baked in.
        self._prec = ({} if str(model_path).endswith(".engine")
                      else precision_kwargs(self.half))
        self.field = None             # endoscope's circular field of view
        self._resolve_classes()
        # predict at the lowest per-class threshold, then filter per class
        per = {n.lower(): v for n, v in getattr(cfg, "CONF_PER_CLASS",
                                                 {}).items()}
        self.conf_by_id = {i: per.get(n.lower(), cfg.CONF)
                           for i, n in self.names.items()}
        self.min_conf = min(self.conf_by_id.values())
        self._warm_up()

    def _warm_up(self):
        """Run the model twice on a blank frame before the video starts:
        the first CUDA call allocates memory and picks kernels (cuDNN
        benchmark), which otherwise shows up as a stutter in the first
        second of the video."""
        if str(cfg.DEVICE) == "cpu":
            return
        try:
            import torch
            torch.backends.cudnn.benchmark = True    # fixed input size
            blank = np.zeros((720, 1280, 3), np.uint8)
            for _ in range(2):
                self.model.predict(blank, imgsz=cfg.IMGSZ, conf=0.9,
                                   device=cfg.DEVICE, **self._prec,
                                   retina_masks=False, verbose=False)
        except Exception:                                   # noqa: BLE001
            pass

    def _resolve_classes(self):
        """Map config class names to model indices, case-insensitively."""
        lower = {n.lower(): i for i, n in self.names.items()}
        self.idx_instrument = lower.get(cfg.CLASS_INSTRUMENT.lower())
        self.idx_critical = {}
        for name in cfg.CLASS_CRITICAL:
            i = lower.get(name.lower())
            if i is None:
                print(f"  [warn] class '{name}' not in model.names "
                      f"-> {list(self.names.values())}")
            else:
                self.idx_critical[i] = name
        if self.idx_instrument is None:
            raise RuntimeError(
                f"Instrument class '{cfg.CLASS_INSTRUMENT}' not found. "
                f"Model classes: {list(self.names.values())}")

    def run(self):
        # A crash in this thread used to be silent: main.py then waited
        # forever for the first frame and the program looked frozen. Record
        # the error so main.py can print it and exit.
        try:
            self._run()
        except Exception as e:                              # noqa: BLE001
            import traceback
            traceback.print_exc()
            self.error = e
            self.finished = True

    def _run(self):
        idx = 0
        bad_reads = 0
        # Play the video in REAL TIME. Without this the thread decodes and
        # runs YOLO as fast as the GPU allows (~45 frames/s on an RTX 4050),
        # keeping the GPU at 100 % non-stop: a laptop then heats up within a
        # minute or two and lowers its clocks, and everything slows down.
        vid_fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
        if not (5.0 <= vid_fps <= 120.0):
            vid_fps = 30.0
        period = 1.0 / vid_fps if getattr(cfg, "VIDEO_REALTIME", True) else 0
        self.video_fps = vid_fps
        next_t = time.perf_counter()
        # Decode the next frame on a helper thread WHILE the GPU runs YOLO on
        # the current one (decoding a 1080p frame costs ~10-25 ms of CPU;
        # done in series it adds straight onto every frame).
        import queue
        q = queue.Queue(maxsize=2)

        def reader():
            while not self._stop_evt.is_set():
                ok, fr = self.cap.read()
                if not ok and self.loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, fr = self.cap.read()
                q.put((ok, fr))
                if not ok:
                    break
        threading.Thread(target=reader, daemon=True, name="video").start()
        while not self._stop_evt.is_set():
            if period:
                now = time.perf_counter()
                if next_t > now:
                    time.sleep(next_t - now)
                    next_t += period
                else:                     # running behind: don't try to
                    next_t = now + period  # catch up with a burst
            try:
                ok, frame = q.get(timeout=1.0)
            except queue.Empty:
                continue
            if not ok:
                if idx == 0:
                    # The file opens but no frame decodes (codec missing,
                    # truncated file): say so instead of hanging.
                    raise RuntimeError(
                        f"Video opened but no frame could be decoded. "
                        f"Check the codec of {cfg.VIDEO_PATH}; try "
                        f"re-encoding it to mp4 (see README) and pass "
                        f"--video.")
                self.finished = True
                break
            bad_reads = 0

            if idx % 60 == 0 or self.field is None and idx % 10 == 0:
                f = estimate_field(frame)
                if f is not None:
                    self.field = f

            t0 = time.perf_counter()
            res = self.model.predict(frame, imgsz=cfg.IMGSZ,
                                     conf=self.min_conf,
                                     device=cfg.DEVICE, **self._prec,
                                     retina_masks=False, verbose=False)[0]
            infer_ms = (time.perf_counter() - t0) * 1000.0

            obs = Observation(frame=frame, infer_ms=infer_ms, index=idx)
            raw = self._degrade(self._extract(res, frame.shape, frame))

            self._apply_tracker(obs, raw)
            for name in cfg.CLASS_CRITICAL:
                d = self.smoother.update(name, raw.get(name))
                if d is not None:
                    obs.critical[name] = d

            self._publish(obs)
            self.fps = 1000.0 / max(infer_ms, 1e-3)
            idx += 1

    def _components(self, res, k, shape, min_area, split=False):
        """
        Outline(s) of instance k in ORIGINAL image pixels, one per separate
        blob of its mask.

        NOT res.masks.xy: Ultralytics builds that by JOINING every blob of an
        instance into one polygon (masks2segments strategy="all"). When the
        model's mask for one tool also covers a second tool or a patch of
        glare, the joined polygon spans both and the drawn axis/tip lands
        between the instruments. Contours are taken here from the raw mask at
        model resolution and mapped back through the letterbox, which costs
        well under a millisecond (no full-resolution upsampling).
        """
        m = res.masks.data[k].cpu().numpy().astype(np.uint8)
        mh, mw = m.shape[:2]
        H, W = shape[:2]
        gain = min(mh / H, mw / W)
        padx, pady = (mw - W * gain) / 2.0, (mh - H * gain) / 2.0
        parts = [m]
        if split and cfg.SPLIT_MERGED_TOOLS:
            parts = []
            n_l, lab = cv2.connectedComponents(m)
            for li in range(1, n_l):
                parts += split_tools((lab == li).astype(np.uint8))
        cnts = []
        for part in parts:
            cs, _ = cv2.findContours(part, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
            cnts += list(cs)
        out = []
        for c in cnts:
            pts = c.reshape(-1, 2).astype(np.float32)
            pts[:, 0] = (pts[:, 0] - padx) / gain
            pts[:, 1] = (pts[:, 1] - pady) / gain
            if len(pts) < 3:
                continue
            area = float(abs(cv2.contourArea(pts)))
            if area >= min_area:
                out.append((area, pts))
        return out

    def _extract(self, res, shape, frame=None):
        """
        Critical structures: largest blob per class.
        Instruments: EVERY separate blob above a minimum size (up to
        cfg.MAX_INSTRUMENTS), so two tools merged into one instance by the
        model come out as two tools again; duplicates and non-elongated blobs
        are dropped.
        """
        out = {"instruments": []}
        if res.masks is None or res.boxes is None:
            return out
        h, w = shape[:2]
        frame_area = float(h * w)
        classes = res.boxes.cls.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()
        n = min(len(classes), len(res.masks.data))

        best = {}
        insts = []
        conf_by_id = getattr(self, "conf_by_id", {})
        for k in range(n):
            cls_id = classes[k]
            if confs[k] < conf_by_id.get(cls_id, 0.0):
                continue
            if cls_id == self.idx_instrument:
                for area, poly in self._components(
                        res, k, shape, cfg.MIN_INSTRUMENT_AREA_FRAC * frame_area,
                        split=True):
                    insts.append((area, poly, float(confs[k])))
            elif cls_id in self.idx_critical:
                for area, poly in self._components(res, k, shape, 50):
                    if cls_id not in best or area > best[cls_id][0]:
                        best[cls_id] = (area, poly)

        for cls_id, (area, poly) in best.items():
            M = cv2.moments(poly)
            if M["m00"] == 0:
                continue
            name = self.idx_critical[cls_id]
            out[name] = Detection(
                name=name, centroid=(M["m10"] / M["m00"], M["m01"] / M["m00"]),
                area_frac=area / frame_area,
                contour=poly.astype(np.int32).reshape(-1, 1, 2))

        insts.sort(key=lambda x: -x[0])
        kept = []
        for area, poly, conf in insts:
            M = cv2.moments(poly)
            if M["m00"] == 0:
                continue
            c = (M["m10"] / M["m00"], M["m01"] / M["m00"])
            contour = poly.astype(np.int32).reshape(-1, 1, 2)
            # duplicate: its centroid falls inside a larger kept tool's mask
            if any(cv2.pointPolygonTest(k.contour, c, True) > -0.01 * w
                   for k in kept):
                continue
            # Laparoscopic tools are long and thin. A blob that is not
            # (glare, a fold of the specimen bag) is not a tool.
            q = poly - poly.mean(axis=0)
            sv = np.linalg.svd(q, compute_uv=False)
            if len(sv) > 1 and sv[0] < cfg.MIN_TOOL_ELONGATION * max(sv[1], 1e-6):
                continue
            det = Detection(name=cfg.CLASS_INSTRUMENT, centroid=c,
                            area_frac=area / frame_area, contour=contour,
                            conf=conf)
            det.ends = instrument_ends(contour)
            det.tip, m_entry, m_inner = tip_and_margins(contour, shape,
                                                        self.field)
            det.tip_margin = abs(m_inner - m_entry)
            # The tip end runs straight into the picture edge: the jaws are
            # outside the view.
            det.tip_visible = m_inner > cfg.TIP_EDGE_FRAC * w
            det.back = distal_axis(contour, det.tip)
            if cfg.AXIS_SNAP and det.tip_visible and frame is not None:
                # Put the line on the metal: re-fit it to the tool's actual
                # pixels and shaft edges in the video frame (tool_axis.py).
                from tool_axis import refine
                ref = refine(frame, contour, det.tip, self.field)
                if ref is not None:
                    det.tip, det.back, det.axis_quality = ref
            kept.append(det)
            if len(kept) >= cfg.MAX_INSTRUMENTS:
                break
        out["instruments"] = kept
        return out


class DemoPerception(_BaseSource):
    """
    No model, no video. Draws a synthetic instrument orbiting a fake 'ureter'
    band so you can verify RCM, IK, IBVS and the safety layer end to end before
    your weights are anywhere near the loop.
    """

    def __init__(self, w=960, h=540, hz=30):
        super().__init__()
        self.w, self.h, self.hz = w, h, hz
        self.smoother = _Smoother()

    def _ureter(self):
        return np.array([[[int(0.22 * self.w), int(0.30 * self.h)]],
                         [[int(0.78 * self.w), int(0.42 * self.h)]],
                         [[int(0.78 * self.w), int(0.50 * self.h)]],
                         [[int(0.22 * self.w), int(0.38 * self.h)]]],
                        dtype=np.int32)

    def true_tips(self, t):
        """Ground-truth tool tips at time t (used by the accuracy test)."""
        return [
            (self.w * (0.55 + 0.22 * np.sin(t * 0.6)),
             self.h * (0.48 + 0.20 * np.sin(t * 0.41 + 1.1))),
            (self.w * (0.38 + 0.05 * np.sin(t * 0.25)),
             self.h * (0.55 + 0.04 * np.sin(t * 0.3))),
        ]

    def make(self, t, idx):
        """
        One synthetic frame and its observation at time t. Deterministic, so
        selftest.py can run the whole closed loop on it without threads.

        Two laparoscopic tools enter from the lower corners: a shaft polygon
        from the border to a moving tip. Tool A (right hand) works actively;
        tool B (left hand) mostly holds/retracts, so the primary-tool logic
        has something to decide.
        """
        ureter = self._ureter()
        frame = np.full((self.h, self.w, 3), (32, 24, 28), np.uint8)
        cv2.drawContours(frame, [ureter], -1, (90, 180, 230), -1)
        entries = [(self.w * 1.02, self.h * 0.95),
                   (-0.02 * self.w, self.h * 0.92)]
        dets = []
        for (tx, ty), (ex, ey), col in zip(self.true_tips(t), entries,
                                           [(225, 225, 230), (150, 150, 160)]):
            poly = _shaft_polygon((ex, ey), (tx, ty), 0.022 * self.w)
            cv2.fillPoly(frame, [poly], col)
            area = float(abs(cv2.contourArea(poly.astype(np.float32))))
            M = cv2.moments(poly.astype(np.float32))
            det = Detection(name=cfg.CLASS_INSTRUMENT,
                            centroid=(M["m10"] / M["m00"], M["m01"] / M["m00"]),
                            area_frac=area / float(self.w * self.h),
                            contour=poly, conf=0.9)
            det.tip = instrument_tip(poly, frame.shape)
            det.back = distal_axis(poly, det.tip)
            dets.append(det)

        ur = Detection(name="Ureter", centroid=(self.w * 0.5, self.h * 0.4),
                       area_frac=0.08, contour=ureter)
        obs = Observation(frame=frame, index=idx, infer_ms=1000.0 / self.hz)
        raw = self._degrade({"instruments": dets})
        self._apply_tracker(obs, raw)
        obs.critical["Ureter"] = self.smoother.update("Ureter", ur)
        return obs

    def run(self):
        idx, t = 0, 0.0
        while not self._stop_evt.is_set():
            self._publish(self.make(t, idx))
            idx += 1
            t += 1.0 / self.hz
            time.sleep(1.0 / self.hz)


def estimate_field(frame):
    """
    The endoscope's circular field of view, as (cx, cy, r) in pixels, or None
    when the image fills the whole frame.

    Laparoscopic video is a bright disc on black, usually clipped at the top
    and bottom. Tools enter the picture at the EDGE OF THAT DISC, not at the
    edge of the rectangle, so the tip test below has to know where the disc
    is. A circle is fitted (least squares) to the disc boundary, ignoring the
    parts of the boundary that are just the frame edge.
    """
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (w // 4, h // 4))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(gray, 18, 255, cv2.THRESH_BINARY)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(float)
    if cv2.contourArea(c.astype(np.float32)) > 0.97 * small.shape[0] * small.shape[1]:
        return None                                   # no black surround
    hs, ws = small.shape[:2]
    keep = (c[:, 0] > 2) & (c[:, 0] < ws - 3) & (c[:, 1] > 2) & (c[:, 1] < hs - 3)
    pts = c[keep]
    if len(pts) < 20:
        return None
    A = np.column_stack([2 * pts[:, 0], 2 * pts[:, 1], np.ones(len(pts))])
    bvec = (pts ** 2).sum(axis=1)
    (cx, cy, k), *_ = np.linalg.lstsq(A, bvec, rcond=None)
    r = np.sqrt(max(k + cx * cx + cy * cy, 1.0))
    return (cx * 4.0, cy * 4.0, r * 4.0)


def instrument_ends(contour):
    """The two extreme points of a tool mask along its principal axis."""
    pts = contour.reshape(-1, 2).astype(np.float32)
    if len(pts) < 3:
        m = tuple(pts.mean(axis=0))
        return m, m
    mean = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - mean, full_matrices=False)
    proj = (pts - mean) @ vt[0]
    return tuple(pts[proj.argmin()]), tuple(pts[proj.argmax()])


def edge_distance(p, shape, field=None, direction=None):
    """
    Distance from p to the edge of the VISIBLE picture (frame rectangle and,
    if given, the endoscope's circular field).

    With `direction`, the distance is measured ALONG that ray (outward along
    the tool's own axis) rather than to the nearest edge. That is the right
    question for "which end did this tool come in from": a shaft entering from
    the left whose far end happens to sit near the TOP of the frame is still
    far from where it entered.
    """
    h, w = shape[:2]
    px, py = float(p[0]), float(p[1])
    if direction is None:
        d = min(px, py, w - px, h - py)
        if field is not None:
            cx, cy, r = field
            d = min(d, r - float(np.hypot(px - cx, py - cy)))
        return d
    ux, uy = direction
    ts = []
    if ux > 1e-9:
        ts.append((w - px) / ux)
    if ux < -1e-9:
        ts.append(-px / ux)
    if uy > 1e-9:
        ts.append((h - py) / uy)
    if uy < -1e-9:
        ts.append(-py / uy)
    t = min(ts) if ts else 0.0
    if field is not None:
        cx, cy, r = field
        fx, fy = px - cx, py - cy
        bq = ux * fx + uy * fy
        disc = bq * bq - (fx * fx + fy * fy - r * r)
        tc = -bq + np.sqrt(disc) if disc > 0 else 0.0
        t = min(t, tc)
    return max(float(t), 0.0)


def tip_and_margins(contour, shape, field=None):
    """
    (tip, entry_margin, inner_margin): the tool's distal end, plus how far each
    end is from the picture edge along the shaft axis.
    """
    a, b = instrument_ends(contour)
    v = np.array(b, float) - np.array(a, float)
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return a, 0.0, 0.0
    u = v / n
    da = edge_distance(a, shape, field, (-u[0], -u[1]))   # beyond end a
    db = edge_distance(b, shape, field, (u[0], u[1]))     # beyond end b
    return (a, db, da) if da > db else (b, da, db)


def distal_axis(contour, tip, frac=None):
    """
    A point ON the tool, back along its shaft from the tip.

    The drawn line runs tip -> this point instead of tip -> centroid: the
    centroid of a bent, partly hidden or oddly segmented mask can sit off the
    instrument. Laparoscopic instruments are rigid and straight, so the
    direction is the mask's principal axis (tip toward the entry end), and
    the length is the distal `frac` of the tool. The end point is pulled onto
    the nearest mask pixel so the line never ends on tissue.
    """
    frac = cfg.DISTAL_AXIS_FRAC if frac is None else frac
    a, b = instrument_ends(contour)
    t = np.array(tip, np.float32)
    far = np.array(b if np.hypot(*(np.array(a) - t)) < np.hypot(
        *(np.array(b) - t)) else a, np.float32)
    v = far - t
    L = float(np.linalg.norm(v))
    if L < 1e-6:
        return tuple(t)
    p = t + v * frac
    if cv2.pointPolygonTest(contour, (float(p[0]), float(p[1])), False) < 0:
        pts = contour.reshape(-1, 2).astype(np.float32)
        p = pts[np.argmin(np.hypot(*(pts - p).T))]
    return (float(p[0]), float(p[1]))


def instrument_tip(contour, shape, field=None):
    """
    Distal tip (end effector) of an instrument mask.

    The centroid of a laparoscopic instrument sits halfway up the shaft, often
    hundreds of pixels from the working end. Take the principal axis of the
    mask; the tool entered the picture from one end, so continuing the axis
    past that end reaches the picture edge quickly. The tip is the OTHER end.

    "Picture edge" includes the endoscope's circular field when one is given.
    The first version measured the nearest edge of the rectangle, which
    picked the shaft's entry point as the tip whenever a tool came in through
    the round edge of the scope image, or lay along the top of the frame.
    """
    return tip_and_margins(contour, shape, field)[0]


def _shaft_polygon(entry, tip, width):
    """Quadrilateral shaft from the image border to the tip."""
    e, t = np.array(entry, float), np.array(tip, float)
    d = t - e
    n = np.array([-d[1], d[0]]) / max(np.linalg.norm(d), 1e-6) * width / 2
    pts = np.array([e + n, t + 0.6 * n, t - 0.6 * n, e - n])
    return pts.astype(np.int32).reshape(-1, 1, 2)


def _circle_contour(cx, cy, r, n=24):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], axis=1)
    return pts.astype(np.int32).reshape(-1, 1, 2)
