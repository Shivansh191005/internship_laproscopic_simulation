# Technical notes

> Detailed design notes, test sections, experiments and troubleshooting.
> For setup, running and the command list see the main [README](../README.md).
> A few sections describe earlier versions of the project (e.g. the old
> "suction arm" role); the code and the README describe the current one.

## Overview

A perception-in-the-loop kinematic simulation of a two-arm surgical assistant.
A YOLO26-seg instance-segmentation model (8 classes) watches real laparoscopic
video. Arm 1 holds the endoscope and light, fixed inside the patient; arm 2
is a two-finger instrument arm whose fingertips follow the surgical tools in
the video. A safety layer keeps the tools away from the ureter, obturator
nerve and iliac vessels, and the surgeon can direct arm 2 by voice.

This is a kinematic simulation, not a surgical digital twin: no deformable
tissue, no force feedback, and safety margins are measured in image pixels.

## Setup

```bash
conda create -n lapassist python=3.10 -y && conda activate lapassist
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Place files as:

```
weights/best.pt               trained YOLO26-seg weights (yolo26s-seg)
data/case01.avi               laparoscopic clip
models/vosk-small-en/         optional, for voice (alphacephei.com/vosk/models)
assets/                      human models + licences for the OR panel
                              (python tools/get_assets.py fetches them)
```

## Run

```bash
python main.py --demo --no-voice     # synthetic target; checks the scene loads
python main.py                       # real perception + voice
python main.py --show-sim            # add the operating-room view to the console
python main.py --record runs/demo.mp4
```

### Voice and keyboard commands

Say the wake word **"Ana"** first (pronounced *AH-nuh*), e.g. *"Ana zoom in"*. Only **"stop"**
works on its own, at any time. Each accepted command beeps (Windows) and shows
in the LAST COMMAND tile; the voice tile shows what was heard.

| Say | Key | What happens |
|---|---|---|
| Ana zoom in / Ana closer | = | camera zooms in (scope goes deeper) |
| Ana zoom out / Ana wider | - | camera zooms out |
| Ana left / right / up / lower | a d w x | view moves one step |
| Ana centre | c | view centres on the active tool |
| Ana follow | f | camera keeps the active tool in the middle |
| Ana freeze (or hold) | h | camera stops moving |
| Ana reset (or home) | g | back to the start view |
| Ana light up / Ana light down (or brighter / darker) | l / k | scope light brighter / dimmer (picture only; the AI always sees the original video) |
| Ana tool follow / Ana tool hold | 1 / 2 | arm 2 follows the tools / stays still |
| Ana picture | s | saves the console to `runs/shot_*.png` |
| Ana mark | m | writes a time mark to `runs/marks_<tag>.csv` |
| **stop** | z | **both arms freeze** (say "Ana tool follow" to resume arm 2) |
| | space / q | pause / quit |
| | v b o r | operating-room view: next / previous angle, 3D window, reset |

How the camera commands work with a recorded video: the RIGHT screen (AI
overlay) shows a zoomed / moved / brighter or darker window of the full
recording (digital zoom); the LEFT screen always stays the untouched camera
feed. The camera arm
in the simulation really tilts and goes deeper through its port. YOLO, the
safety warnings and arm 2 always use the FULL frame, so zooming in can never
hide a danger warning.

Check recognition without speaking: `python voice_test.py` (a computer voice
says every command; all should PASS). Speak and see what is heard:
`python voice_test.py --mic`. Voice needs `pip install vosk sounddevice`
(+ `pyttsx3` for the test) and the model in `models/vosk-small-en/`.

## Troubleshooting (Windows, RTX 4050)

Run the checks in this order; each one isolates a layer.

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python selftest.py --no-sim        # A kinematics, B control, C your model + video
python selftest.py                 # + D PyBullet
python main.py --demo --no-voice   # scene + console with a synthetic target
python main.py --no-voice          # real model on data/case01.avi
```

| Symptom | Cause | Fix |
|---|---|---|
| `Invalid CUDA 'device=0' requested` | pip installed the CPU build of torch | reinstall torch from the cu124 index (Setup); config.py now falls back to CPU automatically |
| Window never opens, console prints nothing after `[perception]` | video opens but frames do not decode | the run now stops with an error; re-encode: `ffmpeg -i data/case01.avi -c:v libx264 -crf 20 data/case01.mp4` then `--video data/case01.mp4` |
| `[voice] vosk/sounddevice not installed` | optional voice packages missing | fine - keyboard works; or use `--no-voice` |
| Very slow (< 5 fps) | running YOLO on CPU, or OR render too often | check CUDA above; raise `SIM_RENDER_EVERY`; drop `--show-sim` |
| Arm "BACKING OFF" all the time / arm off its port | old cart layout + single-shot IK (fixed) | update scene.py and config.py |

The console's **RCM dev** (both arm tiles) should stay under ~1 mm and the
ARMS tile should read clearance well above 6 cm. If a pose is outside what the
arm can physically hold through its port, it is now refused at the cone edge
(like a joint limit) instead of letting the arm leave the trocar.

## Instrument model: speed and accuracy

* **Speed.** YOLO runs from `weights/best.pt` in FP16 on the GPU (TensorRT is
  not used). The console prints `device: GPU (CUDA)` at start; if it says CPU,
  reinstall the CUDA build of torch (see Setup).
* **Inference settings (accuracy, no retraining).** `IMGSZ = 768` and per-class
  thresholds from the training repo's F1 tuning (`instruments` 0.17) -
  see `CONF_PER_CLASS` in `config.py`.
* **Merged tools.** The training labels were cut from semantic masks, so touching
  tools were one instance and the model outputs one mask for them.
  `perception.split_tools` re-splits such masks into straight shafts at run time
  (`SPLIT_MERGED_TOOLS`).
* **Retraining** (on the training machine) - `retrain_instruments.py`:

```bash
python retrain_instruments.py split-labels --data <data.yaml> --dry-run   # count merged tools
python retrain_instruments.py split-labels --data <data.yaml>             # fix labels (backup kept)
python retrain_instruments.py mine --video data/case01.avi --out hard_frames  # frames to correct
python retrain_instruments.py train --data <data.yaml>                    # fine-tune from best.pt
```

## Arm roles

* **Arm 1 – camera arm (endoscope + light).** Inside the patient and FIXED: the video
  is what this camera sees (`CAMERA_MODE = "stable"`; `"follow"` restores the old
  tool-tracking camera).
* **Arm 2 – instrument arm.** Carries the surgical tool. The active tool's tip in the
  video is turned into a 3D point on the tissue through the fixed camera
  (`camera_model.py`), and arm 2 moves its own tool tip there through its trocar.
  The console shows the follow error in mm.
* **Both tools always stay inside the patient** (`INSERTION_LIMITS` minimum 6 cm).
  There is no suction role and no "park" any more.
* **Safety.** Approaching the ureter / nerve / iliac vessels raises a warning. With
  `INSTRUMENT_FIXTURE = True` arm 2 also refuses to move closer once inside the
  danger margin. If the arms come within `ARM_MIN_CLEARANCE`, arm 2 stops moving
  toward the camera arm (it never pulls out).

## Operating-room panel (`--show-sim`)

The bottom-left panel shows the theatre: surgeon, patient in low lithotomy
(prepped lower abdomen, drape window, leg boots, breathing tube), the two
KUKA LBR Med-style arms in clear sterile sleeves, the monitor tower showing the
live endoscope feed, surgical lights, anaesthesia machine and instrument
trolley. Only the picture changes: physics, kinematics, RCM, safety and the
HUD are untouched.

How it works: PyBullet still runs the simulation (DIRECT mode, no window) and
`or_render.py` mirrors the robot links every frame into a **pyrender** scene with
physically based materials, spot lights with shadows, supersampling and a
light photographic finish. The people (`humans.py`) and props (`props.py`) are
built once at start-up. Each arm gets a name tag with its live activity.

```bash
python main.py --show-sim                    # realistic panel
python main.py --show-sim --bullet-window    # also open PyBullet's own window
```

**Angles and zoom.** In the console, **v** / **b** change the small panel's
angle (overview, over the shoulder, ports close-up, surgeon's view, head end,
anaesthetist, camera arm, instrument arm, left side, right side, top down,
whole room). Press **o** (or start with `--or-window`) for the big
**Operating room (3D)** window. It is a real OpenGL window in its own process
(`or_viewer.py`), so moving the camera never waits for YOLO or the console:

| In the 3D window | Action |
|---|---|
| left-drag | orbit (the floor stays level) |
| right-drag (or Shift+left) | pan |
| mouse wheel | smooth zoom |
| v / b | next / previous angle (the camera glides there) |
| r | back to the current angle |
| Esc / close button | close (press o in the console to reopen) |

The title bar shows the window's frame rate. Size: `OR_WINDOW_SIZE`, frame
rate: `OR_WINDOW_FPS`, angles: `OR_VIEWS`.

**Smoothness.** `fastgl.py` cuts the OpenGL calls pyrender makes per frame
(cached uniform locations; lights and camera bound once per shader instead of
once per object), the console reuses its image buffers instead of building
new ones every frame, the main loop runs on a fixed 30 Hz clock, and on
Windows the 1 ms system timer is requested while the program runs.

### Quality vs speed (`config.py`)

| `OR_QUALITY` | Anti-aliasing | Shadows | Surface relief + finish | Render time* |
|---|---|---|---|---|
| `low` | none | off | off | ~0.24 s |
| `medium` (default) | 2x | on | on | ~0.78 s |
| `high` | 3x | on | on | ~1.2 s |

\* Measured on a CPU-only software renderer; a GPU (e.g. RTX 4050) is several
times faster. The panel is only redrawn every `OR_RENDER_EVERY` frames (default
6), so the video and HUD stay at full speed.

Other settings:

| Setting | Effect |
|---|---|
| `OR_RENDERER` | `"pyrender"` (realistic) or `"pybullet"` (old flat look) |
| `OR_SIZE`, `OR_FOV` | panel resolution and lens |
| `OR_VIEWS` | list of `(name, distance, yaw, pitch, target)` camera views |
| `OR_RENDER_EVERY` | redraw interval in frames; raise it if the loop slows |
| `OR_REALISTIC` | `False` = simple shapes, fastest |
| `OR_STERILE_SLEEVES` | clear drapes over the robot arms |
| `OR_CAPTIONS` | arm name tags with live activity |
| `OR_QUALITY_PRESETS` | fine-tune lamp / room light, fake bounce light |
| `PATIENT_PELVIS_X` | where the patient lies along the table |

If the loop slows down: use `low`, raise `OR_RENDER_EVERY`, or lower `OR_SIZE`.

### Fallbacks

* pyrender (or OpenGL) can't start → the old PyBullet renderer is used
  automatically and a message is printed. If it fails mid-run, the same happens.
* MakeHuman files missing → simple figures are used; run
  `python tools/get_assets.py` to fetch them.
* Photo textures: the floor, wall, drape and ceiling textures are generated
  (with normal maps). To use real ones, drop `floor.jpg` / `wall.jpg` (plus
  optional `*_normal.png`, `*_rough.png`) into `assets/textures/`. Generated
  textures are cached in `assets/textures/.cache/`.
* Custom people: `assets/humans/surgeon_custom.glb` or `patient_custom.glb`
  replace the built-in ones.
* HDRI environment lighting is not used: pyrender doesn't support it.

### Troubleshooting

* `AttributeError: np.infty` — handled in `or_render.py` (pyrender + numpy 2).
* Blank or crashing panel on Windows → `pip install "pyglet<2" PyOpenGL==3.1.0`.
* Linux / headless → set `PYOPENGL_PLATFORM=egl` (done automatically when
  there is no display) and install `libegl1` / `libglu1-mesa`.

Asset licences are in `assets/LICENSES.md` (MakeHuman: CC0; everything else is
generated in code).

## Checking it works: `selftest.py`

```bash
python selftest.py                 # everything, including PyBullet
python selftest.py --frames 300    # sample more of your video
```

Every function is run against a known correct answer and graded
PASS / WARN / FAIL, with the measured value and the criterion. Saved to
`runs/selftest.md`, which can go straight into the report as a verification
table.

| Section | What it proves |
|---|---|
| A Kinematics | RCM exact; instrument rigid; insertion moves the arm; tilt limit; reach over the full workspace |
| B Control & safety | servo centres a static target; reacquires a lost one; HOLD freezes; suction reaches standoff; PARK retracts; safety levels correct at known distances; predictive warning fires for fast approach only; occlusion hold; fixture repels the suction tip and ignores the surgeon's tool; voice grammar |
| C Perception | on YOUR model and video: all 8 classes present; instrument detection rate; confidence; mask flicker; tip stability; latency; and what share of instrument positions the scope can physically centre |
| D Simulation | IK reaches commanded grip point; the simulated arm's tool axis really passes through the trocar (physical RCM); arms clear; retracting really lifts the arm |

**Reading a high tracking error.** Section C separates two causes. If
detection rate or confidence is low, the model is the problem - retrain, or
try `IMGSZ = 960` and a lower `CONF`. If detection is good but "positions the
scope CAN centre" is low, the instrument is working at the edge of the
recorded field and no controller could centre it; that is a limit of servoing
over pre-recorded video, not an accuracy fault.

## Experiments and results

Each run writes `runs/metrics_<tag>.csv`. `analyze.py` turns any number of runs
into a results table and figures in `runs/report/`.

```bash
python main.py --once --no-voice --tag baseline
python main.py --once --no-voice --tag drop50  --drop 0.50
python main.py --once --no-voice --tag drop90  --drop 0.90
python main.py --once --no-voice --tag jitter20 --jitter 20
python analyze.py runs/metrics_baseline.csv runs/metrics_drop50.csv runs/metrics_drop90.csv runs/metrics_jitter20.csv
```

`--drop` removes that fraction of instrument detections at random; `--jitter`
adds Gaussian position noise in pixels. Both are seeded, so runs repeat. This
measures how tracking degrades as perception gets worse, without needing
ground-truth masks for the video.

`runs/report/summary.md` contains: instrument-in-view %, centroid error
(mean/median/95th), inference latency, time in warn/danger, danger events,
predictive warnings, minimum distance and time-to-contact, arm-to-arm
clearance, arm-guard activations, and RCM pivot error.

## Architecture

```
video ─► YOLO26-seg (thread) ─► smoothing + occlusion hold
                                        │
             ┌──────────────────────────┼─────────────────────────┐
             ▼                          ▼                         ▼
   virtual endoscope ─► IBVS camera servo    suction standoff    safety layer
             │                          │    servo                (distance + TTC)
             │                          ▼         │                   │
             │                  RCM kinematics ◄──┴── virtual fixture ┘
             │                          │
             │                          ▼
             │                PyBullet: null-space IK, 2 × KUKA iiwa,
             │                arm-to-arm clearance guard
             ▼
   surgeon console (clean feed | AI overlay | status)   ◄── voice / keyboard
```

## Key design decisions

**Virtual endoscope.** Replaying video straight into a controller is open-loop:
moving the arm changes nothing the camera sees, so tracking metrics would
measure the cameraman who filmed the clip. The recorded frame is treated as the
world and the endoscope view is a crop steered by the camera arm, which closes
the loop.

**RCM by construction.** Each instrument is parameterised by its trocar
(two tilts plus insertion), and the arm is placed to match. The pivot is an
input, not a constraint an IK solver can drift from. Measured error ~1e-17 m.

**Orthogonal tilts, not yaw/pitch.** Polar yaw/pitch is singular pointing
straight down, which is exactly where a laparoscope sits.

**Rigid instrument.** The instrument has a fixed length, so insertion moves the
arm's grip point. (An earlier version moved only the tip.)

**Image-based visual servoing.** No depth estimation: the servo nulls the
instrument's image-space error.

**Occlusion hold.** When a critical structure disappears behind tissue, its
last position is held rather than dropped. A safety system that switches off
when it cannot see is worse than none.

**Predictive warning.** Besides distance thresholds, the safety layer estimates
the instrument's closing speed and warns when time-to-contact falls below
1.2 s, so a fast approach is flagged before it crosses the distance threshold.

**The robot guards its own tool.** The surgeon's instrument approaching a
critical structure raises an advisory alert (distance and time-to-contact);
the robot cannot move the surgeon's hand. The virtual fixture measures from the
suction arm's own tip and pushes it away from the nearest structure.

**Physical RCM check.** The console's "RCM dev" is measured from the simulated
arm's actual flange pose, not the command, so it shows real lag during motion.

**Voice cannot override safety.** Commands bias the controllers; the virtual
fixture still applies on top. The recogniser uses a restricted grammar to avoid
false triggers.

**Null-space IK.** A 7-DoF arm is redundant; seeding IK with the previous
solution and slew-limiting joint steps stops the elbow flipping between
configurations.

## Layout

Low lithotomy, as used for laparoscopic gynaecology: patient supine with legs
abducted, surgeon at the foot between the legs, assist arms on carts either
side of the table, monitor tower at the head end facing the surgeon.

## Limitations

- Kinematic only: no tissue deformation, contact, or force feedback.
- 2D perception and control; safety distances are pixels, not millimetres.
- The virtual endoscope reframes recorded pixels; it cannot create parallax or
  reveal occluded tissue, and it cannot pan beyond the recorded field.
- The suction arm's target is a fixed standoff from the instrument, not a
  learned or blood-seeking behaviour.
- The operating-room panel is illustrative: people are posed CC0 MakeHuman
  meshes and the room is procedural. It shows spatial context, not a
  validated digital twin.

## Future work

Isaac Sim / ORBIT-Surgical for photorealistic rendering and GPU physics;
SOFA for deformable tissue; stereo or monocular depth for metric safety
margins; a blood/smoke class to trigger suction automatically; user study with
surgeons on voice-command latency and false-trigger rate.
