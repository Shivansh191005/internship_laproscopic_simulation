"""
Why is the OPERATING ROOM panel still the old flat picture?

Run:   python check_or_panel.py

Checks each piece the realistic panel needs, in order, and prints the exact
command to fix the first thing that fails. Writes runs/or_panel_check.png
when everything works.
"""
import os
import sys
import traceback

if sys.platform.startswith("linux") and "PYOPENGL_PLATFORM" not in os.environ \
        and not os.environ.get("DISPLAY"):
    os.environ["PYOPENGL_PLATFORM"] = "egl"          # headless Linux

OK, BAD = "[ OK ]", "[FAIL]"


def fail(step, err, fix):
    print(f"{BAD} {step}")
    print(f"       error: {type(err).__name__}: {err}")
    print(f"       fix  : {fix}")
    print("\nFull error (send this if the fix does not help):")
    traceback.print_exc()
    sys.exit(1)


print(f"python {sys.version.split()[0]}  ({sys.executable})\n")

# 1. packages ---------------------------------------------------------------
pkgs = [("numpy", "numpy"), ("trimesh", "trimesh"), ("OpenGL", "PyOpenGL==3.1.0"),
        ("pyglet", '"pyglet<2"'), ("PIL", "Pillow"), ("scipy", "scipy"),
        ("pyrender", "pyrender==0.1.45")]
for mod, pipname in pkgs:
    try:
        m = __import__(mod) if mod != "pyrender" else None
        if mod == "pyrender":
            import numpy as np
            if not hasattr(np, "infty"):
                np.infty = np.inf
            m = __import__("pyrender")
        ver = getattr(m, "__version__", getattr(m, "version", "?"))
        print(f"{OK} {mod:9s} {ver}")
    except Exception as e:                                  # noqa: BLE001
        fail(f"import {mod}", e, f"pip install {pipname}")

import pyglet                                               # noqa: E402
if int(str(pyglet.version).split(".")[0]) >= 2:
    print(f"{BAD} pyglet {pyglet.version} is too new for pyrender")
    print('       fix  : pip install "pyglet<2"')
    sys.exit(1)

# 2. OpenGL context ----------------------------------------------------------
try:
    import pyrender
    r = pyrender.OffscreenRenderer(64, 64)
    sc = pyrender.Scene()
    import numpy as np
    sc.add(pyrender.PerspectiveCamera(yfov=1.0), pose=np.eye(4))
    r.render(sc)
    r.delete()
    print(f"{OK} OpenGL offscreen context")
except Exception as e:                                      # noqa: BLE001
    fail("OpenGL offscreen context (pyrender.OffscreenRenderer)", e,
         'pip install --force-reinstall PyOpenGL==3.1.0 "pyglet<2"  and '
         "update the NVIDIA / Intel graphics driver. On a laptop, set "
         "python.exe to 'High-performance NVIDIA processor' in the NVIDIA "
         "Control Panel or Windows Graphics settings.")

# 3. assets -------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import config as cfg                                        # noqa: E402
base = os.path.join(cfg.ASSET_DIR, "src", "makehuman", "base.obj")
if os.path.isfile(base):
    print(f"{OK} MakeHuman files ({base})")
else:
    print(f"[WARN] {base} missing -> simple figures. "
          "fix: python tools/get_assets.py")
print(f"{OK} config: OR_RENDERER={cfg.OR_RENDERER!r}  "
      f"OR_QUALITY={cfg.OR_QUALITY!r}")
if cfg.OR_RENDERER != "pyrender":
    print('       note: set OR_RENDERER = "pyrender" in config.py')

# 4. the real panel -------------------------------------------------------------
try:
    import cv2
    from scene import Scene
    from or_render import ORRenderer
    s = Scene(gui=False)
    rr = ORRenderer(s)
    img = rr.render()
    os.makedirs("runs", exist_ok=True)
    cv2.imwrite("runs/or_panel_check.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    rr.close()
    print(f"{OK} realistic panel rendered -> runs/or_panel_check.png")
except Exception as e:                                      # noqa: BLE001
    fail("building the realistic panel", e,
         "send the full error below")

print("\nAll good. Run:  python main.py --show-sim --no-voice")
