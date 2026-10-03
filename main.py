"""
Entry point.

    python main.py --demo                  synthetic target, no model or video
    python main.py                         YOLO26-seg over config.VIDEO_PATH
    python main.py --video data/x.avi      override the clip
    python main.py --show-sim              add the operating-room view
    python main.py --no-voice              keyboard only

Experiments (each run writes runs/metrics_<tag>.csv, then use analyze.py):
    python main.py --once --tag baseline
    python main.py --once --tag drop30  --drop 0.30
    python main.py --once --tag jit15   --jitter 15

Voice:  follow | hold | center | zoom in/out | left right up down
        tool follow / tool hold      (arm 2, the instrument)
Keys :  f h c = - a d w x 1 2 3   v cycle camera view   space pause
        s screenshot   q quit
"""
import argparse
import csv
import os
import time

import cv2
import numpy as np

import config as cfg
import display
from scene import Scene, check_reachability
from rcm import RCMState
from control import (CameraServo, SafetyLayer, InstrumentArm,
                     guard_instrument)
from camera_model import (FixedCamera, fixed_camera_state, rcm_for_tip,
                          working_point)
from endoscope import VirtualEndoscope
from commands import CommandBus

WINDOW = "surgeon console"


def build_source(args):
    if args.demo:
        from perception import DemoPerception
        print("[perception] demo mode - synthetic target, no model loaded")
        src = DemoPerception(w=1280, h=720)
    else:
        from perception import YoloPerception
        video = args.video or cfg.VIDEO_PATH
        model = args.model or cfg.MODEL_PATH
        print(f"[perception] {model} over {video}")
        src = YoloPerception(model_path=model, video_path=video,
                             loop=not args.once)
    src.drop_prob = args.drop
    src.jitter_px = args.jitter
    if args.drop or args.jitter:
        print(f"[perception] degraded: drop={args.drop:.2f} "
              f"jitter={args.jitter:.1f}px")
    return src


def describe_aim(obs):
    """Short text for the console: which fingers have a tool to follow."""
    ids = sorted({t.track_id for t in (obs.instruments if obs else [])})
    if not ids:
        return "no tool in view"
    return " + ".join(f"finger {i} -> tool {i}" for i in ids if i in (1, 2))


def describe_activity(cmd, tel, status, arm2, guard_reason, camv=None):
    """Plain-language description of what each arm is doing right now."""
    if cfg.CAMERA_MODE == "stable" and camv is not None:
        cam = (camv.status()[1], (0.15, 0.40, 0.85))
    elif cfg.CAMERA_MODE == "stable":
        cam = ("FIXED - not moving",
               (0.15, 0.40, 0.85))
    else:
        mode = tel.get("mode", "")
        if cmd["mode"] == "HOLD":
            cam = ("HOLD - keeping the view still", (0.15, 0.40, 0.85))
        elif mode == "track":
            cam = ("TRACKING the surgeon's instrument", (0.10, 0.55, 0.20))
        elif mode == "reacquire":
            cam = ("SEARCHING - instrument left the view", (0.85, 0.55, 0.05))
        else:
            cam = ("WAITING - no instrument detected", (0.55, 0.20, 0.20))

    if guard_reason:
        ins = (f"STOPPED - {guard_reason}", (0.85, 0.10, 0.10))
    elif arm2.mode == "hold":
        ins = ("HOLD - instrument kept still", (0.15, 0.40, 0.85))
    elif arm2.mode == "waiting":
        ins = ("WAITING - no tool in the video, holding position",
               (0.55, 0.40, 0.20))
    else:
        ins = ("OPERATING - fingers on the tool tips", (0.10, 0.55, 0.20))
    return cam, ins


_PANEL_MODES = ("process", "thread", "inline")


def _panel_error(where, e, tb=None):
    """Print a panel problem clearly and keep the details in a file."""
    import traceback
    msg = f"[or-panel] {where} failed: {type(e).__name__}: {e}"
    print(msg)
    try:
        os.makedirs("runs", exist_ok=True)
        with open(os.path.join("runs", "or_panel_error.txt"), "a") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")
            f.write(tb or "".join(traceback.format_exception(
                type(e), e, e.__traceback__)))
            f.write("\n")
    except Exception:                                       # noqa: BLE001
        pass


class _InlinePanel:
    """Last resort: draw in the main loop (same look, a little slower)."""
    mode = "inline"

    def __init__(self, scene):
        from or_render import ORRenderer
        self.r = ORRenderer(scene)
        self.error = None
        self.image = None
        self.frames = 0
        self.render_ms = 0.0
        self.busy = False

    cam = None
    size = None

    def set_camera(self, cam):
        self.cam = cam

    def set_size(self, size):
        self.size = size

    def submit(self, frame):
        try:
            t = time.perf_counter()
            if frame is not None:
                self.r.set_screen_image(frame)
            snap = self.r.snapshot()
            snap["cam"], snap["size"] = self.cam, self.size
            self.image = self.r.draw(snap)
            self.render_ms = (time.perf_counter() - t) * 1000
            self.frames += 1
        except Exception as e:                              # noqa: BLE001
            self.error = e

    def set_view(self, i):
        self.r.set_view(i)

    def close(self):
        self.r.close()


def start_panel(scene, modes=None):
    """
    Start the realistic OR panel. Tries, in order: its own process (fastest
    for the console), a background thread, then drawing inline. Falls back
    to the old PyBullet picture ONLY if pyrender cannot draw at all.
    """
    modes = cfg.OR_PANEL_MODES if modes is None else modes
    for mode in modes:
        try:
            if mode == "process":
                from or_render import PanelProcess
                v = PanelProcess(scene)
            elif mode == "thread":
                from or_render import PanelWorker
                v = PanelWorker(scene)
            else:
                v = _InlinePanel(scene)
            v.mode = mode
            print(f"[or-panel] realistic panel ON ({mode} mode, "
                  f"quality {cfg.OR_QUALITY})")
            return v
        except Exception as e:                              # noqa: BLE001
            _panel_error(f"{mode} mode", e)
    print("\n" + "!" * 70)
    print("[or-panel] realistic panel could not start in any mode;")
    print("[or-panel] using the old PyBullet picture. Details:")
    print("[or-panel]   runs/or_panel_error.txt   and   python check_or_panel.py")
    print("!" * 70 + "\n")
    return None


def _fine_timer(on=True):
    """Windows sleeps in 15.6 ms steps by default, so a 33 ms frame becomes
    31 or 47 ms at random (visible judder). Ask for 1 ms timer resolution
    while the program runs (the standard fix used by games / media apps)."""
    if os.name != "nt":
        return
    try:
        import ctypes
        f = ctypes.windll.winmm.timeBeginPeriod if on else \
            ctypes.windll.winmm.timeEndPeriod
        f(1)
    except Exception:                                       # noqa: BLE001
        pass


def main():
    _fine_timer(True)
    try:
        _main()
    finally:
        _fine_timer(False)


def _main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--video", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--no-voice", action="store_true")
    ap.add_argument("--show-sim", action="store_true")
    ap.add_argument("--or-window", action="store_true",
                    help="open the big zoomable operating-room window at start")
    ap.add_argument("--bullet-window", action="store_true",
                    help="also open PyBullet's own 3D window (off by default "
                         "when the realistic OR panel is shown)")
    ap.add_argument("--record", default=None, help="save the console as mp4")
    ap.add_argument("--tag", default="run", help="name for this run's CSV")
    ap.add_argument("--once", action="store_true",
                    help="play the video once, then stop (for experiments)")
    ap.add_argument("--seconds", type=float, default=0,
                    help="stop after N seconds (0 = run until q)")
    ap.add_argument("--drop", type=float, default=0.0,
                    help="probability of dropping each instrument detection")
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="std-dev of instrument position noise, px")
    args = ap.parse_args()

    print("[scene] reachability check over the full workspace")
    if not check_reachability():
        print("  fix the geometry in config.py before continuing")
        return

    # Physics runs in DIRECT mode when the realistic OR panel draws the
    # scene; PyBullet's own window only on request (or with no panel).
    use_gui = not args.headless and (args.bullet_window or not args.show_sim
                                     or cfg.OR_RENDERER != "pyrender")
    scene = Scene(gui=use_gui)
    or_view = None
    if args.show_sim and not args.headless and cfg.OR_RENDERER == "pyrender":
        or_view = start_panel(scene)
    view_idx = 0
    from orcam import OrbitCamera
    from or_viewer import ORWindow3D
    or_cam = OrbitCamera()            # camera of the small console panel
    or_win = ORWindow3D()             # big 3D window (own process)
    if args.or_window and or_view is not None:
        or_win.start(or_cam.preset)
    view_name = (cfg.OR_VIEWS[0][0] if or_view is not None
                 else scene.VIEWS[0][0])
    # Arm 1: endoscope + light, inside the patient. Fixed in "stable" mode.
    cam_state = (fixed_camera_state() if cfg.CAMERA_MODE == "stable"
                 else RCMState(cfg.TROCAR_CAMERA, insertion=0.13))
    # Arm 2: the instrument, starting at the centre of the operative field.
    inst_state = RCMState(cfg.TROCAR_INSTRUMENT,
                          *rcm_for_tip(cfg.TROCAR_INSTRUMENT, working_point()))

    scene.place("camera", cam_state)
    scene.place("instrument", inst_state)

    servo, safety = CameraServo(), SafetyLayer()
    bus = CommandBus()

    listener, voice_status = None, "disabled (--no-voice)"
    if not args.no_voice:
        from voice import VoiceListener
        listener = VoiceListener(bus)
        listener.start()
        time.sleep(0.6)
        voice_status = listener.status

    source = build_source(args)
    display.MODEL_LABEL = getattr(source, "model_label", "demo target")
    print(f"[perception] model: {display.MODEL_LABEL}  |  weights: "
          f"{cfg.MODEL_PATH}  |  device: "
          f"{'GPU (CUDA)' if cfg.DEVICE != 'cpu' else 'CPU  <-- SLOW'}")
    source.start()
    obs = None
    t_wait = time.perf_counter()
    while obs is None:
        obs = source.latest()
        if obs is None and getattr(source, "error", None) is not None:
            print(f"[perception] FAILED: {source.error}")
            scene.disconnect()
            return
        if obs is None and time.perf_counter() - t_wait > 120:
            print("[perception] no frame after 120 s - giving up. "
                  "Run  python selftest.py  to see which part fails.")
            scene.disconnect()
            return
        time.sleep(0.01)
    print(f"[perception] first frame after "
          f"{time.perf_counter() - t_wait:.1f} s - starting control loop")
    scope = VirtualEndoscope(obs.frame.shape[1], obs.frame.shape[0])
    camera_model = FixedCamera(obs.frame.shape[1], obs.frame.shape[0],
                               fixed_camera_state())
    arm2 = InstrumentArm(camera_model)
    # Voice/keyboard camera control (zoom, pan, centre, follow, home). In
    # "follow" CAMERA_MODE the old automatic camera servo is used instead.
    camv = None
    if cfg.CAMERA_MODE == "stable":
        from camview import CameraVoice
        camv = CameraVoice(camera_model, fixed_camera_state())
    snapshot_req = False
    print(f"[camera] fixed endoscope sees {camera_model.metres_per_pixel()*1000:.3f}"
          f" mm/px on the tissue ({camera_model.metres_per_pixel()*obs.frame.shape[1]*100:.0f}"
          f" cm across)")

    os.makedirs("runs", exist_ok=True)
    csv_path = os.path.join("runs", f"metrics_{args.tag}.csv")
    csv_f = open(csv_path, "w", newline="")
    writer = csv.writer(csv_f)
    writer.writerow(["t", "frame", "infer_ms", "track_mode", "cam_mode",
                     "instrument", "in_view", "err_norm", "closest", "dist_px",
                     "warn_px", "danger_px", "level", "predictive", "ttc_s",
                     "fixture", "arm_clear_m", "arm_guard",
                     "cam_pivot_err_mm", "suc_pivot_err_mm", "last_command",
                     "follow_err_px", "follow_err_mm", "arm2_mode"])

    dt = 1.0 / cfg.CONTROL_HZ
    sub = max(1, cfg.SIM_HZ // cfg.CONTROL_HZ)
    t0 = time.perf_counter()
    next_tick = t0
    n = in_view = 0
    err_sum = 0.0
    paused = False
    vw = None
    fps = 0.0
    last = time.perf_counter()
    strength = 0.0
    clearance = cfg.ARM_CLEAR_QUERY
    arm_guard = False
    sim_img = None
    window_ready = False
    cam_txt = suc_txt = "starting"
    follow_sum, follow_n = 0.0, 0
    last_index = -1

    perf = None
    if getattr(cfg, "PERF_LOG", True):
        from perfmon import PerfMonitor
        perf = PerfMonitor(args.tag)
    last_submit = 0.0
    panel_dirty = False

    print("[run] q quit | space pause | s snapshot | m mark | z STOP")
    print("[run] camera: = / - zoom | a d w x pan | c centre | f follow | "
          "h hold | g home | l / k light     instrument: 1 follow | 2 hold")
    if cfg.VOICE_WAKE_WORD:
        w = cfg.VOICE_WAKE_WORD.capitalize()
        print(f'[run] voice: "{w} zoom in", "{w} left", "{w} follow", '
              f'"{w} light up", ... and "stop" (no wake word) - see README')
    if or_view is not None:
        print("[run] operating room: o = open big 3D window (drag / wheel to "
              "orbit & zoom) | v / b = next / previous angle | r = reset")
    try:
        while True:
            if not paused:
                obs = source.latest()
            if obs is None:
                time.sleep(0.01)
                continue
            if args.once and source.finished and obs.index == last_index:
                print("[run] video finished")
                break
            if args.seconds and time.perf_counter() - t0 > args.seconds:
                print(f"[run] {args.seconds:.0f} s elapsed")
                break
            if getattr(source, "error", None) is not None:
                print(f"[perception] stopped: {source.error}")
                break
            last_index = obs.index

            cmd = bus.snapshot()
            for ev, src in bus.pop_events():
                if camv is not None:
                    camv.command(ev, obs)
                if ev == "SNAPSHOT":
                    snapshot_req = True
                elif ev == "MARK":
                    with open(os.path.join("runs", f"marks_{args.tag}.csv"),
                              "a") as mf:
                        mf.write(f"{time.strftime('%H:%M:%S')},"
                                 f"{time.perf_counter() - t0:.2f},"
                                 f"{obs.index},{src}\n")
                    print(f"[mark] {time.perf_counter() - t0:.1f} s, video "
                          f"frame {obs.index} -> runs/marks_{args.tag}.csv")

            if not paused:
                view = scope.project(obs, cam_state)
                status, dists = safety.evaluate(obs)
                if cfg.CAMERA_MODE == "stable":
                    cam_cmd = {}
                    tel = {"in_view": obs.instrument is not None,
                           "mode": "stable", "err_x": 0.0, "err_y": 0.0}
                else:
                    cam_cmd, tel = servo.update(view, cmd)

                # Arm 2 follows the active tool; safety may stop it (never
                # pulls it out of the patient).
                inst_cmd = arm2.update(dt, cmd, obs, inst_state)
                cam_tip = cam_state.pose()[0]
                inst_cmd, guard_reason = guard_instrument(
                    inst_cmd, inst_state, dt, obs, camera_model, clearance,
                    cam_tip)
                arm_guard = guard_reason is not None
                strength = 1.0 if guard_reason and "safety" in guard_reason \
                    else 0.0
                status["strength"] = strength
                status["fixture_target"] = guard_reason

                cam_state.integrate(cam_cmd, dt)
                scope.clamp_state(cam_state)
                if camv is not None:
                    # the camera arm aims at the centre of the voice-chosen
                    # view and goes deeper when zoomed in (rate-limited)
                    camv.update(dt, obs)
                    camv.drive_arm(cam_state, dt)
                inst_state.integrate(inst_cmd, dt)
                sim_diag = {"camera": scene.drive("camera", cam_state),
                            "suction": scene.drive("instrument", inst_state)}
                scene.step(sub)
                clearance = scene.arm_clearance()
                (cam_txt, cam_col), (suc_txt, suc_col) = describe_activity(
                    cmd, tel, status, arm2, guard_reason, camv)
                scene.annotate("camera", cam_txt, cam_col)
                scene.annotate("instrument", suc_txt, suc_col)

                # Two-finger head: solve both fingers for the cannula pose
                # the simulated arm now has, and draw them.
                base, ends, elbows = arm2.solve_fingers(inst_state)
                scene.set_fingers(base, ends, elbows)
                # How well the fingers reproduce the video: each jaw vs. the
                # 3D point of its tool tip (mm).
                fe = arm2.finger_errors_mm()
                fe_ok = [e for e in fe if e is not None]
                follow_mm = float(np.mean(fe_ok)) if fe_ok else None
                follow_px = None
                if follow_mm is not None:
                    follow_sum += follow_mm
                    follow_n += 1
                tel["follow_mm"] = follow_mm
                tel["finger_mm"] = fe

                n += 1
                e = float(np.hypot(tel.get("err_x", 0), tel.get("err_y", 0)))
                if tel.get("in_view"):
                    in_view += 1
                    err_sum += e
                ttc = status.get("ttc", float("inf"))
                writer.writerow([
                    f"{time.perf_counter()-t0:.3f}", obs.index,
                    f"{obs.infer_ms:.2f}", tel.get("mode", ""), cmd["mode"],
                    cmd["instrument"], int(tel.get("in_view", False)),
                    f"{e:.4f}", status.get("closest") or "",
                    "" if status.get("dist") is None else f"{status['dist']:.1f}",
                    f"{status.get('warn_px', 0):.1f}",
                    f"{status.get('danger_px', 0):.1f}",
                    status["level"], int(bool(status.get("predictive"))),
                    "" if ttc == float("inf") else f"{ttc:.3f}",
                    f"{strength:.2f}", f"{clearance:.4f}", int(arm_guard),
                    f"{sim_diag['camera']['pivot_error_m']*1000:.4f}",
                    f"{sim_diag['suction']['pivot_error_m']*1000:.4f}",
                    cmd["last"],
                    "" if follow_px is None else f"{follow_px:.1f}",
                    "" if follow_mm is None else f"{follow_mm:.2f}",
                    arm2.mode])

            now = time.perf_counter()
            if perf is not None and not paused:
                perf.tick(obs.index, obs.infer_ms,
                          getattr(or_view, "frames", 0) if or_view else 0,
                          getattr(or_view, "render_ms", 0.0) if or_view else 0)
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
            last = now

            if not args.headless:
                # The OR render is expensive and slows inference on a laptop
                # GPU; the room barely changes frame to frame, so refresh it
                # only every few frames.
                if args.show_sim and or_view is not None:
                    # Realistic panel: drawn on its own thread. Hand it the
                    # current poses (never waits) and show its newest picture.
                    if or_view.error is not None:
                        # keep the realistic look: restart it in the next,
                        # simpler mode instead of going back to PyBullet
                        _panel_error(f"{or_view.mode} mode", or_view.error)
                        nxt = _PANEL_MODES.index(or_view.mode) + 1
                        try:
                            or_view.close()
                        except Exception:                   # noqa: BLE001
                            pass
                        or_view = start_panel(scene, _PANEL_MODES[nxt:])
                        if or_view is not None:
                            or_view.set_view(view_idx)
                            or_view.set_camera(or_cam.spec())
                        sim_img = None
                    else:
                        # the big 3D window draws itself; it only needs
                        # the arm poses (tiny message, never waits)
                        or_win.push(scene, (obs.frame, display.LAST_OVERLAY))
                        if (n % cfg.OR_RENDER_EVERY == 0 or panel_dirty) and \
                                now - last_submit >= 1.0 / cfg.OR_MAX_FPS \
                                and not or_view.busy:
                            last_submit = now
                            or_view.set_camera(or_cam.spec())
                            panel_dirty = False
                            # the tower monitors show the live endoscope feed
                            or_view.submit((obs.frame, display.LAST_OVERLAY))
                        if or_view.image is not None and \
                                or_view.frames != getattr(or_view, "_shown", -1):
                            or_view._shown = or_view.frames
                            big = cv2.cvtColor(or_view.image,
                                               cv2.COLOR_RGB2BGR)
                            if big.shape[1] != cfg.OR_SIZE[0]:
                                big = cv2.resize(big, tuple(cfg.OR_SIZE),
                                                 interpolation=cv2.INTER_AREA)
                            sim_img = big
                        view_name = or_cam.name
                if args.show_sim and or_view is None and (
                        sim_img is None or n % cfg.SIM_RENDER_EVERY == 0):
                    sim_img = cv2.cvtColor(scene.render(), cv2.COLOR_RGB2BGR)
                if listener is not None:
                    voice_status = listener.status
                console = display.compose(
                    view, obs, status, tel, cmd, sim_diag, fps, obs.infer_ms,
                    voice_status, sim_img if args.show_sim else None,
                    extra={"arm_clearance": clearance, "arm_guard": arm_guard,
                           "view_name": view_name, "cam_activity": cam_txt,
                           "view_crop": camv.crop() if camv else None,
                           "light": camv.light if camv else 1.0,
                           "cam_title": (f"CAMERA ARM  {camv.status()[0]}"
                                         if camv else None),
                           "aim": describe_aim(obs),
                           "suc_activity": suc_txt})

                if not window_ready:
                    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(WINDOW, console.shape[1], console.shape[0])
                    window_ready = True

                if snapshot_req:
                    snapshot_req = False
                    sp = os.path.join("runs", f"shot_{obs.index:06d}.png")
                    cv2.imwrite(sp, console)
                    print(f"[snapshot] saved {sp}")
                if args.record:
                    if vw is None:
                        vw = cv2.VideoWriter(
                            args.record, cv2.VideoWriter_fourcc(*"mp4v"),
                            cfg.CONTROL_HZ, (console.shape[1], console.shape[0]))
                    vw.write(console)

                cv2.imshow(WINDOW, console)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
                elif k == ord(" "):
                    paused = not paused
                elif k == ord("s"):
                    snapshot_req = True
                elif k in (ord("v"), ord("b")):
                    step = 1 if k == ord("v") else -1
                    view_idx += step
                    if or_view is not None:
                        or_cam.next(step)
                        or_view.set_view(or_cam.preset)
                        panel_dirty = True          # redraw at once
                        view_name = or_cam.name
                    else:
                        view_name = scene.set_view(view_idx) if use_gui \
                            else scene.VIEWS[view_idx % len(scene.VIEWS)][0]
                elif k == ord("r") and or_view is not None:
                    or_cam.apply_preset(or_cam.preset)
                    panel_dirty = True
                elif k == ord("o") and or_view is not None:
                    # big zoomable 3D window of the operating room
                    or_win.toggle(or_cam.preset)
                elif k != 255:
                    bus.key(k)

            # Fixed-rate loop: wait for the NEXT TICK (deadline), not "dt
            # after the display finished" - that made each loop dt + work
            # long and the frame rate uneven.
            next_tick += dt
            wait = next_tick - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            elif wait < -dt:              # far behind: don't burst-catch-up
                next_tick = time.perf_counter()
    except KeyboardInterrupt:
        pass
    finally:
        source.stop()
        # Wait for the video/YOLO thread to finish its frame. Exiting while
        # it is still inside a CUDA/torch call aborted the process
        # ("terminate called without an active exception").
        try:
            source.join(timeout=3.0)
        except RuntimeError:
            pass
        if listener is not None:
            listener.stop()
        csv_f.close()
        if vw is not None:
            vw.release()
        or_win.stop()
        cv2.destroyAllWindows()
        if or_view is not None:
            or_view.close()
        scene.disconnect()

        print("\n---- summary ----")
        if n:
            print(f"instrument in view : {100*in_view/n:5.1f} %")
        if in_view and cfg.CAMERA_MODE != "stable":
            print(f"mean centroid error: {err_sum/in_view:.4f}")
        if follow_n:
            print(f"finger follow error: {follow_sum / follow_n:.2f} mm "
                  f"(mean, finger jaw vs. its tool tip in the video)")
        print(f"min distance seen  : {safety.min_dist_seen:.1f} px")
        print(f"danger events      : {safety.events}")
        print(f"predictive warnings: {safety.predictive_events}")
        print(f"voice commands     : "
              f"{sum(1 for _, _, s in bus.log if s == 'voice')}")
        print(f"metrics written to : {csv_path}")
        print(f"analyse with       : python analyze.py {csv_path}")
        if perf is not None:
            perf.report()


if __name__ == "__main__":
    main()
