"""
Central configuration. Tune everything here, not in the other modules.
Lengths are metres, angles are radians unless noted.
"""
import numpy as np

# ---------------------------------------------------------------- perception
PT_WEIGHTS = "weights/best.pt"     # your trained YOLO-seg weights
VIDEO_PATH = "data/case01.avi"     # laparoscopic video to replay
# The model was trained at imgsz 512 on 1920x1080 frames. On this video 768
# finds instruments in 82% of sampled frames vs 61% at 640 (same weights);
# the cost is ~1.3x inference time. Drop back to 640 if the loop is too slow.
IMGSZ = 768
CONF = 0.35                        # default for classes not listed below
# Per-class thresholds, tuned on the validation set to maximise F1
# (tune_and_probe.py in the training repo). Instruments peaked at 0.17:
# at a single 0.35 the model silently dropped ~1/3 of real tools.
CONF_PER_CLASS = {
    "external iliac artery": 0.32,
    "external iliac vein": 0.22,
    "obturator nerve": 0.15,       # val best was 0.09; slightly raised: few GT
    "ovary": 0.34,
    "ureter": 0.15,
    "uterine artery": 0.25,
    "uterus": 0.30,
    "instruments": 0.17,
}
# 0 = first CUDA device, "cpu" to force CPU. Falls back to CPU automatically
# when torch has no CUDA (e.g. the CPU wheel got installed by pip): asking
# Ultralytics for device=0 without CUDA raises "Invalid CUDA 'device=0'".
def _pick_device():
    try:
        import torch
        if torch.cuda.is_available():
            return 0
    except Exception:                                       # noqa: BLE001
        pass
    print("[config] CUDA not available - running YOLO on CPU (slow). "
          "Install the CUDA build of torch for the RTX 4050.")
    return "cpu"


DEVICE = _pick_device()
HALF = DEVICE != "cpu"             # FP16 inference only on the GPU

# TensorRT: OFF. The engine built on this laptop ran slower than the normal
# PyTorch weights, so best.pt (FP16 on the GPU) is used. Set True to try
# weights/best_<IMGSZ>.engine again.
USE_TENSORRT = False
_ENGINE = f"weights/best_{IMGSZ}.engine"
import os as _os
MODEL_PATH = (_ENGINE if USE_TENSORRT and DEVICE != "cpu"
              and _os.path.isfile(_ENGINE) else PT_WEIGHTS)

# Your 8 dataset classes. Names must match model.names (case-insensitive).
CLASS_INSTRUMENT = "Instruments"
CLASS_CRITICAL = [
    "Ureter",
    "Obturator Nerve",
    "External Iliac Artery",
    "External Iliac Vein",
]
CLASS_SOFT = ["Uterus", "Ovary", "Uterine Artery"]

# Exponential smoothing on mask centroids. 0 = no smoothing, 0.9 = very heavy.
# Raw per-frame masks flicker; without this the camera arm will shake.
CENTROID_SMOOTH = 0.82
# Area drives the zoom axis and is far noisier than position — smooth it hard.
AREA_SMOOTH = 0.92
# Instruments. Every tool in view is tracked (not only the largest mask).
MAX_INSTRUMENTS = 2                # the surgeon's two working tools
MIN_INSTRUMENT_AREA_FRAC = 0.002   # smaller masks are fragments / noise
TRACK_GATE_FRAC = 0.25             # max tip jump (x frame width) to keep an ID
INSTRUMENT_HOLD_FRAMES = 15        # keep a vanished tool this many frames
PRIMARY_SWITCH_RATIO = 1.6
TIP_EDGE_FRAC = 0.04               # both ends nearer the edge than this
                                   # -> the tool's tip is out of view
# Split masks in which the model fused two touching/crossing tools into one
# (its training labels did exactly that - see perception.split_tools).
SPLIT_MERGED_TOOLS = True
SPLIT_MIN_FRAC = 0.18              # 2nd shaft must hold >= this share of pixels
SPLIT_MIN_ANGLE = 12.0             # deg between the two shafts
MIN_TOOL_ELONGATION = 1.8         # length/width of a tool mask (axis ratio)
DISTAL_AXIS_FRAC = 0.35
# Snap each tool line to the shaft's real straight edges in the video frame
# (tool_axis.py). The mask is then only a search area.
AXIS_SNAP = True
AXIS_WORK_SCALE = 0.5             # colour search at half resolution (speed)
AXIS_EDGE_HALF = False            # True: edges at 1/4 res, ~2x faster but
                                  # less accurate (on-tool score 0.67 -> 0.63)
AXIS_MARGIN_FRAC = 0.03           # search this far (x width) outside the mask
AXIS_MIN_SEG_FRAC = 0.08          # ignore edge bits shorter than this x tool
AXIS_MAX_ANGLE_DEV = 25.0         # deg from the mask's own direction
AXIS_PARALLEL_DEG = 6.0           # deg: edges that belong to the shaft
AXIS_WIDTH_RANGE = (0.35, 2.2)    # edge gap vs. mask width estimate
AXIS_MIN_QUALITY = 0.15           # below this, keep the mask-based line
AXIS_RUN_SPLIT = False            # see tool_axis._single_tool_run
AXIS_RUN_BINS = 24                # steps along the line for the brightness check
AXIS_RUN_JUMP = 55.0              # grey-level jump that marks another tool
AXIS_MIN_METAL_FRAC = 0.25        # metal pixels needed to trust colour axis
METAL_S_MAX, METAL_V_MAX, METAL_S_GREY = 105, 170, 55   # HSV metal rule            # drawn axis = distal 35% of the tool
TIP_SNAP_FRAC = 0.12               # tip jump (x width) treated as a re-pick
TIP_AMBIGUOUS_FRAC = 0.06          # ends closer than this to equally "inner"
                                   # -> keep the end nearest the previous tip         # other tool must be this much more active
# What the camera arm centres:
#   "midpoint" - midpoint of all tool tips when two are visible (frames a
#                two-handed task, like a human camera assistant)
#   "primary"  - the most active tool's tip only
CAMERA_AIM = "midpoint"

# How many frames to keep trusting a critical structure after it goes occluded.
OCCLUSION_HOLD_FRAMES = 45

# ---------------------------------------------------------------- geometry
# Abdominal wall plane height, and the two trocar (pivot) points on it.
# Operating-room layout. +x is toward the patient's head.
TABLE_TOP_Z = 0.78                 # OR table height
WALL_Z = 0.95                      # abdominal wall (top of the draped torso)
TROCAR_CAMERA = np.array([-0.04, 0.06, WALL_Z])      # arm 1: endoscope+light
TROCAR_INSTRUMENT = np.array([0.12, -0.02, WALL_Z])  # arm 2: surgical tool
TROCAR_SUCTION = TROCAR_INSTRUMENT                   # old name, kept for scripts

# Assist-arm carts. Keep pivots well inside KUKA iiwa's ~0.8 m reach.
# Lithotomy setup, as used for gynaecological laparoscopy:
#   - patient supine, head at +x, legs raised and apart at -x
#   - surgeon stands BETWEEN the legs at the foot of the table, facing +x
#   - assist arms on carts either side of the table
#   - monitor tower at the head end, so the surgeon looks straight down the
#     patient at the screens
# Each cart sits level with its trocar in x and 0.58 m out to the side, on a
# 0.95 m pedestal. The old carts (0, +/-0.50, 0.72) were too close and too
# low: tilting a tool back toward its own cart put the grip point almost
# inside the shoulder, and ~1/3 of the tilt cone was physically unreachable
# (the arm then left the trocar by 50+ mm). Now ~96% is reachable and the few
# cone-edge poses left are refused cleanly (scene.Scene.drive).
CART_CAMERA = np.array([-0.04, 0.64, 0.95])   # patient's left
CART_INSTRUMENT = np.array([0.12, -0.60, 0.95])  # patient's right
CART_SUCTION = CART_INSTRUMENT                    # old name
IIWA_SHOULDER_Z = 0.34            # axis-2 height above the base
IIWA_REACH = 0.80                 # nominal reach, measured from axis 2

# Where the surgeon's own instruments enter the abdomen (x, |y|).
# These are the working ports; the robot uses TROCAR_CAMERA / TROCAR_SUCTION.
SURGEON_PORTS = (-0.10, 0.11)
SURGEON_POS = (-0.70, 0.0)         # at the foot, between the legs, facing +x
MONITOR_POS = (1.35, 0.0, 1.32)
MONITOR_YAW = np.pi                # screens face -x, toward the surgeon

# Room half-extents (metres) for the enclosing floor/walls/ceiling.
ROOM_SIZE = (3.2, 3.2, 1.6)
# Square on purpose: the GUI camera orbits, so the walls must clear the
# camera distance in EVERY direction, not just the default one. If you scroll
# the camera out past the walls you will see flat grey - zoom back in.
GUI_CAMERA_DISTANCE = 2.3

# Arm-to-arm clearance monitor
ARM_CLEAR_QUERY = 0.30        # m; distances beyond this are not computed
ARM_MIN_CLEARANCE = 0.06      # m; below this arm 2 stops moving toward arm 1

# Scope view cone drawn in the OR scene
SCOPE_HALF_FOV = 35.0         # deg; a typical 0-degree laparoscope is ~70 FOV
SCOPE_CONE_LEN = 0.06         # m

SHAFT_LENGTH = 0.12   # legacy; kept for scripts that import it
# Total rigid instrument length. The part outside the body is
# INSTRUMENT_LENGTH - insertion, so insertion now visibly moves the arm.
INSTRUMENT_LENGTH = 0.32
MIN_OUTSIDE = 0.08
EE_LINK = 6           # KUKA iiwa end-effector link index

# RCM limits. Tilt is parameterised as two orthogonal angles (tilt_x, tilt_y)
# rather than polar yaw/pitch, which is singular pointing straight down.
MAX_TILT = np.deg2rad(50)                  # half-angle of the reachable cone
# Depth below the trocar. Minimum > 0: both tools ALWAYS stay inside the
# patient (no "park" any more). Max = tool length - minimum outside length.
INSERTION_LIMITS = (0.06, 0.24)
ROLL_LIMITS = (-np.pi, np.pi)              # unused by control, kept for completeness

# Velocity clamps. A twitchy camera is the #1 complaint about camera-holder
# robots — keep these low and raise them only if tracking feels sluggish.
MAX_TILT_RATE = np.deg2rad(45)      # rad/s, applies to both tilt axes
# 28 deg/s panned the view at only ~270 px/s on a 1920 px clip, while an
# instrument crossing the frame moves ~1300 px/s. The servo saturated and the
# error parked near 1.0 forever. Lower this only if the view looks frantic.
MAX_INSERT_RATE = 0.05              # m/s (arm 2 follows tool depth too)

# ---------------------------------------------------------------- arm roles
# Arm 1 holds the endoscope + light, inside the patient, STILL: the video is
# what this fixed camera sees. Arm 2 holds the surgical instrument and moves
# its tip to wherever the active tool's tip is in the video.
#   "stable"  - camera fixed (default, the new behaviour)
#   "follow"  - old behaviour: camera arm servoes to keep the tools centred
CAMERA_MODE = "stable"
CAMERA_INSERTION = 0.06           # m inside the abdominal wall
CAMERA_HFOV = 70.0                # deg, a typical 0-degree laparoscope
TISSUE_Z = WALL_Z - 0.16          # height of the tissue the tools work on
FIELD_OFFSET = (0.0, 0.0)         # shift the camera's aim point (m, x/y)
INSTRUMENT_HOVER = 0.0            # m above the tissue for the finger tips
# Arm 2's head: one cannula, two articulated fingers (finger 1 -> Tool 1,
# finger 2 -> Tool 2 in the video). Reach = sum of the two links.
FINGER_LINKS = (0.055, 0.050)     # m, proximal / distal link
FINGER_SPLIT = 0.035              # m: fingers leave the cannula this far
                                  # above the midpoint of the two tool tips
K_INSTRUMENT = 8.0                # 1/s: how quickly arm 2 closes the gap
                                  # (4 -> 6.6 mm median lag, 8 -> 4.6 mm)
INSTRUMENT_DEADBAND = 0.002       # m: closer than this counts as "there"
# Arm 2 reproduces the surgeon's motion in the video. With the fixture ON it
# refuses to move its tip closer to the ureter / nerve / iliac vessels once
# inside the danger margin; in dissection video the tool works right next to
# them, so the arm then stops most of the time. OFF = warning only (console
# + CSV), the arm keeps following.
INSTRUMENT_FIXTURE = False

# ---------------------------------------------------------------- IBVS gains
# Image-based visual servoing: drive the instrument centroid to image centre.
# No depth needed. Flip the SIGN constants if the arm servos the wrong way —
# the correct sign depends on your trocar placement and camera roll.
K_TILT_X = 1.0
K_TILT_Y = 1.0
# Velocity feedforward: lead the instrument instead of chasing it. Pure
# proportional control always lags a moving target by err/gain.
K_FEEDFORWARD = 0.12
# Feedforward differentiates the centroid, so it amplifies segmentation noise.
# Velocity is smoothed hard and the term is clamped, or the scope shivers.
FF_SMOOTH = 0.90
FF_CLAMP = 0.25                     # rad/s, max contribution from feedforward

# Low-pass on the outgoing tilt command. This is the single most effective
# anti-shake control: it costs a little responsiveness and removes almost all
# of the frame-to-frame twitch that noisy masks inject.
CMD_SMOOTH = 0.80
K_INSERT = 0.10
# Ignore small area errors entirely. Without a deadband the scope hunts on
# zoom forever, which reads as a constantly breathing view.
AREA_DEADBAND = 0.020
SIGN_X = +1.0
SIGN_Y = +1.0

# Deadband in normalised image coords: ignore errors smaller than this so the
# arm sits still when the instrument is roughly centred.
IBVS_DEADBAND = 0.06
# Target fraction of the frame the instrument mask should occupy. Larger mask
# means the tool is closer, so the endoscope retracts.
TARGET_AREA_FRAC = 0.038

# ---------------------------------------------------------------- safety
# Proximity thresholds in pixels, measured from the instrument tip to the
# nearest point of a critical-structure mask.
# Expressed as a fraction of FRAME WIDTH, so they scale with video resolution.
# Fixed pixel thresholds are meaningless across different clips.
WARN_DIST_FRAC = 0.10
DANGER_DIST_FRAC = 0.045

# Predictive warning: time-to-contact from the instrument's closing speed.
TTC_WARN_S = 1.2              # warn if contact is predicted within this
PREDICT_RANGE_FRAC = 0.22     # ...and the instrument is within this range
TTC_MIN_SPEED_PX = 25.0       # px/s; slower than this counts as not closing
TTC_SMOOTH = 0.75
# Strength of the repulsive virtual fixture applied to the suction arm.
REPULSION_GAIN = 1.8

# ---------------------------------------------------------------- runtime
SIM_HZ = 120
CONTROL_HZ = 30
SHOW_SIM_VIEW = False    # PyBullet already has its own window; keep this
                         # off or the composite gets too wide to fit the screen
SIM_VIEW_SIZE = (640, 380)     # widescreen OR render for the console
METRICS_CSV = "runs/metrics.csv"

# ---------------------------------------------------------------- virtual endoscope
# The recorded frame is treated as the "world"; the endoscope view is a
# steerable crop of it, driven by the camera arm's RCM state. This is what
# closes the visual servo loop over replayed video.
VIEW_SIZE = (640, 512)        # rendered endoscope view (w, h)
TISSUE_DEPTH = 0.16           # metres below the trocar where the tissue plane sits
ZOOM_WIDE = 0.62              # crop width as a fraction of the frame, retracted
ZOOM_TIGHT = 0.30             # crop width fraction, fully inserted

# How far full tilt pans the view, as a fraction of frame width. Kept separate
# from ZOOM_WIDE: tying the two together capped travel at +/-0.25*W, so an
# instrument out past 90% of the frame could never be centred at all.
# The crop is still clamped to the frame, so this cannot sample outside.
PAN_FRAC = 0.46
# Largest fraction of the endoscope view allowed to fall outside the video
# before the tilt is clamped. A real scope shows vignette at the edge of its
# field; this keeps that bounded instead of letting the view drift into black.
MAX_OVERHANG = 0.0

# What the two monitors actually show.
#   "full"  - the whole recorded frame, never cropped, never zoomed. No black
#             borders and a steady picture. The scope's framing is drawn as a
#             rectangle so you can still see what the robot is aiming at.
#   "scope" - the cropped endoscope view (what the robot's camera sees).
# The visual servo loop runs identically either way; this only changes the
# display, so tracking metrics stay valid in both modes.
DISPLAY_MODE = "full"

# Fixed field of view. With zoom disabled the crop never changes size, so the
# picture cannot breathe and the pan bounds are constant.
ZOOM_ENABLED = False
FIXED_VIEW_FRAC = 0.66

# Reacquisition: gain used when the instrument is detected in the world frame
# but has fallen outside the endoscope view. Without this the servo has no
# error signal and the arm deadlocks wherever it happens to be.
K_REACQUIRE = 0.9
MAX_DISPLAY_WIDTH = 1500      # console is downscaled to fit your screen
MAX_DISPLAY_HEIGHT = 900
SIM_RENDER_EVERY = 6          # re-render the OR thumbnail every N frames

# ---------------------------------------------------------------- commands
PAN_NUDGE = 0.15

# Suction arm: stand off from the instrument tip by this fraction of the frame
# so it is ready without obstructing the working tool or the view.
SUCTION_OFFSET = (-0.16, 0.12)
K_SUCTION = 0.8
SUCTION_DEADBAND = 0.05              # how far one "left"/"right" shifts the view
VOSK_MODEL_DIR = "models/vosk-small-en"
VOICE_SAMPLE_RATE = 16000
VOICE_WAKE_WORD = "ana"           # say "ana zoom in" (Ana = "AH-nuh");
                                  # "" = no wake word
VOICE_WAKE_ALIASES = ["anna"]     # also accepted (a common mishearing)
VOICE_BEEP = True                 # short beep when a command is accepted

# ---------------------------------------------------------------- figures
# PyBullet draws primitives, not people. For a genuinely realistic patient or
# surgeon, drop an .obj here and it is loaded instead of the built-in figure.
PATIENT_MESH = "assets/patient.obj"
PATIENT_MESH_SCALE = 1.0
SURGEON_MESH = "assets/surgeon.obj"
SURGEON_MESH_SCALE = 1.0

# ---------------------------------------------------------------- arm damping
# A 7-DoF arm is redundant, so the IK solver can hop between elbow branches
# frame to frame. That reads as violent shaking even when the tool tip is
# smooth. These two settings keep the arm on one branch and filter the rest.
JOINT_SMOOTH = 0.40           # low-pass on joint targets. Was 0.75, needed only
                              # while the old IK flipped elbow branches; with
                              # the continuous DLS solver 0.40 halves the arm's
                              # lag (grip error p95 7.7 -> 3.7 mm)
MAX_JOINT_STEP = 0.55         # rad; a bigger one-step jump is a branch flip

# Damped-least-squares IK (scene.py). Iterations per control step, damping,
# and a weak null-space pull toward the home posture (0 disables it).
IK_ITERS = 60
IK_DAMPING = 0.03
IK_POSTURE_GAIN = 0.02
IK_TOL = 0.003                # m; a pose the arm misses by more is refused
MIN_REACHABLE_FRAC = 0.85     # reachability check fails below this share

# ---------------------------------------------------------------- OR panel rendering
# "pyrender": realistic PBR renderer (or_render.py). "pybullet": the old flat
# renderer. If pyrender cannot start, the old renderer is used automatically.
OR_RENDERER = "pyrender"
OR_QUALITY = "medium"              # low | medium | high
OR_QUALITY_PRESETS = {
    #            supersampling  shadows  ambient            lamp   room
    #  bounce = fake indirect light (pyrender has no global illumination)
    #  normal_maps = surface relief on floor / walls / drapes
    #  post = photographic finish (gentle contrast curve + lens vignette)
    "low":    dict(supersample=1, shadows=False, ambient=[0.6] * 3,
                   lamp=13.0, room=2.1, bounce=0.16,
                   normal_maps=False, post=False),
    "medium": dict(supersample=2, shadows=True, ambient=[0.6] * 3,
                   lamp=13.0, room=2.1, bounce=0.16,
                   normal_maps=True, post=True),
    "high":   dict(supersample=3, shadows=True, ambient=[0.6] * 3,
                   lamp=13.0, room=2.1, bounce=0.16,
                   normal_maps=True, post=True),
}
OR_SIZE = (640, 380)               # panel resolution (before downscaling)
OR_FOV = 55.0                      # deg, vertical
OR_RENDER_EVERY = 2                # hand the panel new poses every N frames (it draws on
                                   # its own thread, so this never slows the console)
OR_FAST_GL = True                  # fewer OpenGL calls per frame (fastgl.py)
OR_PANEL_MODES = ("process", "thread", "inline")   # tried in this order
OR_MAX_FPS = 10.0                  # panel pictures per second (caps GPU load)
VIDEO_REALTIME = True              # play the video at its own frame rate
                                   # (False = as fast as possible: hot GPU)
PERF_LOG = True                    # runs/perf_<tag>.csv + slowdown report
# Panel views (press v). The first one is the original panel framing.
OR_VIEWS = [
    # name,                distance, yaw, pitch, target
    ("overview",            2.2,  -18, -24, [0.05, 0.00, 0.95]),
    ("over the shoulder",   1.10, -62, -38, [0.02, 0.00, 0.95]),
    ("ports close-up",      0.65, -35, -40, [0.04, 0.02, 0.97]),
    ("surgeon's view",      1.45,  -90, -58, [0.05, 0.00, 0.95]),
    ("head end",            2.3,   62, -34, [0.05, 0.00, 0.95]),
    ("anaesthetist",        1.2,  110, -35, [0.40, 0.00, 0.95]),
    ("camera arm",          1.6, -140, -25, [-0.04, 0.35, 1.10]),
    ("instrument arm",      1.6,   60, -25, [0.12, -0.35, 1.10]),
    ("left side",           2.4,    0, -15, [0.00, 0.00, 1.00]),
    ("right side",          2.4,  180, -15, [0.00, 0.00, 1.00]),
    ("top down",            1.3,  -90, -89, [0.00, 0.00, 0.90]),
    ("whole room",          4.2,  -40, -28, [0.00, 0.00, 1.00]),
    ("monitors",            0.9,  -90,  -6, [1.35, 0.00, 1.32]),
]
OR_WINDOW_SIZE = (1280, 720)       # the big zoomable window (key o)
OR_WINDOW_FPS = 60.0               # 3D window refresh rate (its own process)
ASSET_DIR = "assets"
OR_REALISTIC = True                # realistic people / room / equipment
OR_STERILE_SLEEVES = True          # clear sterile drapes over the robot arms
OR_CAPTIONS = True                 # "CAMERA ARM" / "INSTRUMENT ARM" tags + activity
# Realistic humans (humans.py)
SURGEON_HEIGHT = 1.78             # m
PATIENT_HEIGHT = 1.68             # m
TABLE_PAD = 0.05                  # m, OR table mattress
PATIENT_PELVIS_X = -0.04          # m: ports over the LOWER abdomen (navel at the top edge of the window)
