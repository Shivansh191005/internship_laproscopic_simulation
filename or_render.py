"""
Realistic renderer for the OPERATING ROOM panel.

PyBullet stays the physics / kinematics engine. This module only DRAWS:

  * the robot arms, instruments, fingers and everything else that MOVES are
    mirrored from PyBullet link by link; every frame only their poses are
    copied (cheap), so they are exactly where the simulation has them;
  * the primitive room / table / patient / surgeon / carts are replaced by
    realistic models built from the same config numbers (humans.py,
    props.py), so nothing moves - it just looks real;
  * the monitor tower shows the live endoscope feed;
  * anything that fails to build falls back to the primitive version with a
    warning, so the sim never stops because of a missing asset.

If pyrender cannot start at all, main.py falls back to PyBullet's renderer.
All settings are in config.py under "OR panel rendering".
"""
import os
import sys
import threading
import time

import numpy as np

# pyrender 0.1.45 still uses np.infty, which numpy 2 removed.
if not hasattr(np, "infty"):
    np.infty = np.inf

# Headless Linux needs EGL; Windows/macOS use pyglet's hidden window.
if sys.platform.startswith("linux") and "PYOPENGL_PLATFORM" not in os.environ \
        and not os.environ.get("DISPLAY"):
    os.environ["PYOPENGL_PLATFORM"] = "egl"

import cv2                                                  # noqa: E402
import pybullet as p                                        # noqa: E402
import trimesh                                              # noqa: E402

import config as cfg                                        # noqa: E402

_GEOM_SPHERE, _GEOM_BOX, _GEOM_CYLINDER, _GEOM_MESH, _GEOM_PLANE, \
    _GEOM_CAPSULE = 2, 3, 4, 5, 6, 7

# groups of PyBullet bodies that get a realistic replacement
_REPLACED = ("room", "table", "patient", "surgeon", "carts", "trocars")


def _quat_to_mat(pos, quat):
    T = np.eye(4)
    T[:3, :3] = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
    T[:3, 3] = pos
    return T


def _warn(msg):
    print(f"[or-panel] {msg}")


class ORRenderer:
    def __init__(self, scene, size=None, offscreen=True):
        import pyrender
        if getattr(cfg, "OR_FAST_GL", True):
            import fastgl
            fastgl.apply()
        self.pr = pyrender
        self.sim = scene
        self.size = size or cfg.OR_SIZE
        self.q = q = cfg.OR_QUALITY_PRESETS[cfg.OR_QUALITY]
        self.ssaa = q["supersample"]
        w, h = self.size
        t0 = time.perf_counter()
        # offscreen=False: the scene is drawn by an on-screen window instead
        # (or_viewer.py), so no hidden framebuffer is needed
        self.renderer = (pyrender.OffscreenRenderer(w * self.ssaa,
                                                    h * self.ssaa)
                         if offscreen else None)
        self.scene = pyrender.Scene(bg_color=[0.05, 0.06, 0.07, 1.0],
                                    ambient_light=q["ambient"])
        self.nodes = []          # (node, body, link, local T)
        self._mesh_cache = {}
        self._tex_cache = {}
        self._screen_nodes = {}
        replaced = set()
        for g in _REPLACED:
            replaced.update(getattr(scene, "groups", {}).get(g, []))
        self._build_from_pybullet(skip=replaced if cfg.OR_REALISTIC else set())
        if cfg.OR_REALISTIC:
            self._build_realistic()
        self._add_lights()
        self.cam_node = self.scene.add(
            pyrender.PerspectiveCamera(yfov=np.radians(cfg.OR_FOV),
                                       aspectRatio=w / h, znear=0.03,
                                       zfar=15.0))
        self.view_index = 0
        _warn(f"scene ready in {time.perf_counter() - t0:.1f} s "
              f"({len(self.scene.mesh_nodes)} meshes)")

    # ---------------------------------------------------------- materials
    def _texture(self, key):
        """
        Texture set for `key`: {"color", "normal", "rough"} pyrender
        Textures (normal / rough may be None). Real photo textures in
        assets/textures/ override the procedural ones:
            floor.jpg|png  (+ optional floor_normal.png, floor_rough.png)
            wall.jpg|png   (+ optional wall_normal.png,  wall_rough.png)
        `*_rough` is a greyscale roughness map.
        """
        if key not in self._tex_cache:
            import props
            gen = props.make_texture_set(key)
            fname = {"floor": "floor", "tiles": "wall"}.get(key)
            maps = {"color": gen["color"], "normal": gen["normal"],
                    "rough": gen["rough"]}
            if fname:
                found = {}
                for kind, suffix in (("color", ""), ("normal", "_normal"),
                                     ("rough", "_rough")):
                    for ext in ("jpg", "png"):
                        path = os.path.join(cfg.ASSET_DIR, "textures",
                                            f"{fname}{suffix}.{ext}")
                        if kind not in found and os.path.isfile(path):
                            im = cv2.imread(path)
                            if im is not None:
                                im = cv2.resize(cv2.cvtColor(
                                    im, cv2.COLOR_BGR2RGB), (1024, 1024))
                                found[kind] = im
                if "color" in found:            # a real photo set replaces
                    maps = {"color": found["color"],          # the whole set
                            "normal": found.get("normal"), "rough": None}
                    if "rough" in found:
                        r = found["rough"][..., 0]
                        maps["rough"] = np.dstack(
                            [r, np.full_like(r, 255), np.zeros_like(r)])
            if not self.q.get("normal_maps", True):
                maps["normal"] = None
            # REPEAT wrapping + full filtering. Without explicit filters the
            # GL texture is "incomplete" and samples as black.
            sampler = self.pr.Sampler(magFilter=9729, minFilter=9987,
                                      wrapS=10497, wrapT=10497)
            self._tex_cache[key] = {
                k: (None if v is None else
                    self.pr.Texture(source=np.ascontiguousarray(v),
                                    source_channels="RGB", sampler=sampler))
                for k, v in maps.items()}
        return self._tex_cache[key]

    @staticmethod
    def _lin(c):
        """sRGB colour (what we pick by eye) -> linear (what the shader
        expects). pyrender gamma-encodes its output, so passing sRGB values
        straight in washes every colour out."""
        c = np.clip(np.asarray(c, float), 0.0, 1.0)
        return np.where(c <= 0.04045, c / 12.92,
                        ((c + 0.055) / 1.055) ** 2.4).tolist()

    def _mat(self, spec):
        rgba = spec.get("color", [0.7, 0.7, 0.7, 1.0])
        rgba = self._lin(rgba[:3]) + [rgba[3]]
        kw = dict(baseColorFactor=rgba,
                  metallicFactor=spec.get("metal", 0.0),
                  roughnessFactor=spec.get("rough", 0.6),
                  alphaMode="BLEND" if rgba[3] < 0.99 else "OPAQUE",
                  doubleSided=rgba[3] < 0.99 or spec.get("double", False))
        # Fake bounce light: pyrender has no global illumination, so surfaces
        # only lit at a glancing angle (walls, sides of people) go nearly
        # black. Real rooms are lit mostly by bounced light; a small glow in
        # the surface's own colour stands in for it (cfg "bounce").
        g = self.q.get("bounce", 0.0)
        tex = None
        if spec.get("texture") and not spec["texture"].startswith("screen"):
            ts = self._texture(spec["texture"])
            tex = ts["color"]
            kw["baseColorTexture"] = tex
            if ts.get("normal") is not None:
                kw["normalTexture"] = ts["normal"]
            if ts.get("rough") is not None:
                kw["metallicRoughnessTexture"] = ts["rough"]
                kw["roughnessFactor"] = 1.0
        if spec.get("emissive"):
            kw["emissiveFactor"] = self._lin(spec["emissive"])
        elif g > 0:
            kw["emissiveFactor"] = [c * g * spec.get("bounce_k", 1.0)
                                    for c in rgba[:3]]
            if tex is not None:
                kw["emissiveTexture"] = tex
        return self.pr.MetallicRoughnessMaterial(**kw)

    def _material_guess(self, rgba):
        r, g, b, a = rgba
        metalish = abs(r - g) < 0.06 and abs(g - b) < 0.08 and r > 0.55
        return self._mat({"color": list(rgba),
                          "rough": 0.3 if metalish else 0.6,
                          "metal": 0.9 if metalish else 0.0})

    # ------------------------------------------------------------ geometry
    def _trimesh_for(self, vd):
        geom, dims, fname = vd[2], vd[3], vd[4]
        if geom == _GEOM_BOX:
            return trimesh.creation.box(extents=np.asarray(dims[:3]))
        if geom == _GEOM_SPHERE:
            return trimesh.creation.icosphere(subdivisions=3,
                                              radius=dims[0])
        if geom == _GEOM_CYLINDER:
            return trimesh.creation.cylinder(radius=dims[1], height=dims[0],
                                             sections=40)
        if geom == _GEOM_CAPSULE:
            return trimesh.creation.capsule(height=dims[0], radius=dims[1],
                                            count=[24, 24])
        if geom == _GEOM_MESH:
            path = fname.decode() if isinstance(fname, bytes) else fname
            key = (path, tuple(np.round(dims, 4)))
            if key not in self._mesh_cache:
                m = trimesh.load(path, force="mesh")
                m.apply_scale(np.asarray(dims[:3]))
                self._mesh_cache[key] = m
            return self._mesh_cache[key].copy()
        return None

    def _build_from_pybullet(self, skip):
        # clear polyethylene drape: mostly see-through with a glossy sheen,
        # not a milky shell (no fake bounce glow on it)
        self._sleeve_mat = self.pr.MetallicRoughnessMaterial(
            baseColorFactor=self._lin([0.90, 0.94, 0.97]) + [0.11],
            metallicFactor=0.0, roughnessFactor=0.06, alphaMode="BLEND",
            doubleSided=True)
        cid = self.sim.client
        arm_ids = set(self.sim.arms.values())
        for b in range(p.getNumBodies(physicsClientId=cid)):
            body = p.getBodyUniqueId(b, physicsClientId=cid)
            if body in skip:
                continue
            for vd in p.getVisualShapeData(body, physicsClientId=cid):
                rgba = vd[7]
                if rgba[3] <= 0.01:
                    continue
                tm = self._trimesh_for(vd)
                if tm is None:
                    continue
                link = vd[1]
                local = _quat_to_mat(vd[5], vd[6])
                if body in arm_ids:
                    mats = [self._robot_material(link)]
                    if cfg.OR_REALISTIC and cfg.OR_STERILE_SLEEVES and link >= 1:
                        mats.append("sleeve")
                else:
                    mats = [self._material_guess(rgba)]
                for mat in mats:
                    mesh_tm = tm
                    if mat == "sleeve":
                        mesh_tm = tm.copy()
                        c = mesh_tm.bounds.mean(axis=0)
                        mesh_tm.apply_translation(-c)
                        mesh_tm.apply_scale(1.05)
                        mesh_tm.apply_translation(c)
                        mat = self._sleeve_mat
                    mesh = self.pr.Mesh.from_trimesh(
                        mesh_tm, material=mat, smooth=vd[2] != _GEOM_BOX)
                    node = self.scene.add(mesh)
                    self.nodes.append((node, body, link, local))

    def _robot_material(self, link):
        """KUKA LBR Med look: white covers, brushed-aluminium joints."""
        if link in (1, 3, 5):
            return self._mat({"color": [0.85, 0.86, 0.88, 1.0], "metal": 0.9,
                              "rough": 0.28})
        return self._mat({"color": [0.96, 0.96, 0.95, 1.0], "metal": 0.0,
                          "rough": 0.30})

    _SMOOTH = ("surgeon", "cove", "clock", "patient", "drape", "ether-drape", "circuit",
               "boot", "et-tube", "head-ring",
               "iv-line", "iv-bag", "light-head", "light-arm", "gas",
               "bellows", "stirrup", "lap-shaft", "kidney", "caster")

    def _add_static(self, name, tm, spec):
        # Smooth normals only for organic / round shapes. On boxes, averaged
        # corner normals point diagonally and half the face goes dark.
        smooth = name.startswith(self._SMOOTH)
        try:
            mesh = self.pr.Mesh.from_trimesh(tm, material=self._mat(spec),
                                             smooth=smooth)
            return self.scene.add(mesh, name=name)
        except Exception as e:                              # noqa: BLE001
            _warn(f"could not add {name}: {e}")
            return None

    def _add_merged(self, items):
        """
        Add static items, merging all untextured ones that share a material
        into ONE mesh. pyrender pays a fixed Python cost per mesh per pass,
        so ~300 small props cost far more than ~60 merged meshes.
        """
        import json
        groups = {}
        for name, tm, spec in items:
            if spec.get("texture") or not isinstance(tm, trimesh.Trimesh):
                self._add_static(name, tm, spec)
                continue
            smooth = name.startswith(self._SMOOTH)
            key = (json.dumps(spec, sort_keys=True, default=str), smooth)
            groups.setdefault(key, (name, spec, []))[2].append(tm)
        for (_, smooth), (name, spec, tms) in groups.items():
            tm = tms[0] if len(tms) == 1 else trimesh.util.concatenate(
                [trimesh.Trimesh(t.vertices, t.faces, process=False)
                 for t in tms])
            self._add_static(name, tm, spec)

    def _build_realistic(self):
        import props
        import humans
        items = []
        # people
        for who, fn, custom in (("surgeon", humans.build_surgeon,
                                 "surgeon_custom.glb"),
                                ("patient", humans.build_patient,
                                 "patient_custom.glb")):
            cpath = os.path.join(cfg.ASSET_DIR, "humans", custom)
            if os.path.isfile(cpath):
                try:
                    m = trimesh.load(cpath, force="mesh")
                    items.append((who, m, {"color": [0.8, 0.7, 0.6, 1],
                                           "rough": 0.6}))
                    _warn(f"using {cpath}")
                    continue
                except Exception as e:                      # noqa: BLE001
                    _warn(f"{cpath} failed ({e}); using the built-in {who}")
            try:
                parts = fn()
                for mat, m in parts:
                    rgba, rough, metal = humans.MATERIALS[mat]
                    items.append((f"{who}-{mat}", m, {"color": rgba,
                                                      "rough": rough,
                                                      "metal": metal}))
                if who == "patient":
                    self._patient_parts = parts
            except Exception as e:                          # noqa: BLE001
                _warn(f"{who} model failed ({e}); MakeHuman files missing? "
                      f"run: python tools/get_assets.py")
                for g in (who,):
                    self._build_from_pybullet_group(g)
        # room and equipment
        builders = [("room", props.room), ("table", props.table),
                    ("trolley", props.trolley), ("iv", props.iv_stand),
                    ("surgeon tools", props.surgeon_instruments)]
        for name, fn in builders:
            try:
                items += fn()
            except Exception as e:                          # noqa: BLE001
                _warn(f"{name} failed ({e})")
        try:
            lights, self._lamp_heads = props.surgical_lights()
            items += lights
        except Exception as e:                              # noqa: BLE001
            _warn(f"lights failed ({e})")
            self._lamp_heads = []
        for base in (cfg.CART_CAMERA, cfg.CART_INSTRUMENT):
            items += props.robot_cart(base)
        if getattr(self, "_patient_parts", None):
            try:
                items += props.drapes(self._patient_parts)
            except Exception as e:                          # noqa: BLE001
                _warn(f"drapes failed ({e})")
            try:
                items += props.boots(humans.META.get("patient_legs", []),
                                     humans.META.get("patient_boots"))
            except Exception as e:                          # noqa: BLE001
                _warn(f"boots failed ({e})")
            try:
                mouth = humans.META.get("patient_mouth",
                                        np.array([0.5, 0, 0.95]))
                items += props.anaesthesia(mouth)
                items += props.et_tube(mouth)
            except Exception as e:                          # noqa: BLE001
                _warn(f"anaesthesia machine failed ({e})")
        self._add_merged(items)
        # live screens on the monitor tower
        for name, tm, spec in props.monitor_screens():
            self._screen_nodes[name] = (tm, None)
        self._screen_imgs = {}
        # trocar cannulas, re-posed every frame along each tool
        self._trocars = {}
        for arm in self.sim.arms:
            c = trimesh.creation.cylinder(radius=0.0075, height=0.09,
                                          sections=24)
            c.apply_translation([0, 0, -0.025])   # mostly outside the skin
            node = self.scene.add(self.pr.Mesh.from_trimesh(
                c, material=self._mat({"color": [0.25, 0.27, 0.30, 1],
                                       "rough": 0.35, "metal": 0.3})))
            cap = trimesh.creation.cylinder(radius=0.016, height=0.03,
                                            sections=24)
            cap.apply_translation([0, 0, -0.075])
            node2 = self.scene.add(self.pr.Mesh.from_trimesh(
                cap, material=self._mat({"color": [0.92, 0.92, 0.90, 1],
                                         "rough": 0.4})))
            self._trocars[arm] = (node, node2)
        self._build_twin_instruments()

    # ------------------------------------------- arm 2: two instruments
    def _build_twin_instruments(self):
        """
        Arm 2 holds TWO instruments (like an arm with two fingers): two drive
        units on its wrist and two shafts that meet at the single port, where
        they continue inside as the two fingers. Colour bands match TOOL 1
        (yellow) and TOOL 2 (magenta) on the AI overlay. Drawn here only -
        the simulation and its kinematics are unchanged.
        """
        self._twin = None
        inst = getattr(self.sim, "_inst", {}).get("instrument")
        if inst is None or "instrument" not in getattr(self, "_trocars", {}):
            return
        shaft_id = inst[0]
        shaft_node = None
        for node, body, link, local in self.nodes:
            if body == shaft_id:
                shaft_node = node
                node.mesh.is_visible = False      # replaced by the two shafts
        if shaft_node is None:
            return
        steel = self._mat({"color": [0.80, 0.82, 0.85, 1], "rough": 0.25,
                           "metal": 0.9})
        housing = self._mat({"color": [0.93, 0.94, 0.95, 1], "rough": 0.3,
                             "metal": 0.0})
        cols = [[0.98, 0.85, 0.20, 1], [0.95, 0.40, 0.90, 1]]
        unit = trimesh.creation.cylinder(radius=1.0, height=1.0, sections=20)
        box = trimesh.creation.box(extents=[0.032, 0.032, 0.070])
        band = trimesh.creation.cylinder(radius=0.0215, height=0.012,
                                         sections=24)
        parts = []
        for c in cols:
            sh = self.scene.add(self.pr.Mesh.from_trimesh(unit, material=steel,
                                                          smooth=True))
            hb = self.scene.add(self.pr.Mesh.from_trimesh(box,
                                                          material=housing))
            bd = self.scene.add(self.pr.Mesh.from_trimesh(
                band, material=self._mat({"color": c, "rough": 0.4,
                                          "metal": 0.0}), smooth=True))
            parts.append((sh, hb, bd))
        self._twin = {"shaft": shaft_node, "parts": parts,
                      "port": self._trocars["instrument"][0]}

    def _pose_twin(self):
        tw = getattr(self, "_twin", None)
        if not tw:
            return
        M = self.scene.get_pose(tw["shaft"])
        R = M[:3, :3]
        d, x = R[:, 2], R[:, 0]
        grip = M[:3, 3] - d * (cfg.INSTRUMENT_LENGTH / 2.0)
        port = self.scene.get_pose(tw["port"])[:3, 3]
        for s, (sh, hb, bd) in zip((-1.0, 1.0), tw["parts"]):
            # drive unit beside the wrist, shaft from it into the port
            h = grip + s * 0.060 * x + d * 0.020
            Th = np.eye(4)
            Th[:3, :3] = R
            Th[:3, 3] = h
            self.scene.set_pose(hb, Th)
            Tb = Th.copy()
            Tb[:3, 3] = h - d * 0.020
            self.scene.set_pose(bd, Tb)
            a = h + d * 0.035
            b = port + s * 0.0035 * x + d * 0.02
            v = b - a
            L = float(np.linalg.norm(v))
            if L < 1e-4:
                continue
            z = v / L
            xx = np.cross(R[:, 1], z)
            if np.linalg.norm(xx) < 1e-6:
                xx = x
            xx /= np.linalg.norm(xx)
            yy = np.cross(z, xx)
            Ts = np.eye(4)
            Ts[:3, :3] = np.column_stack([xx * 0.0042, yy * 0.0042, z * L])
            Ts[:3, 3] = (a + b) / 2
            self.scene.set_pose(sh, Ts)

    def _build_from_pybullet_group(self, group):
        ids = set(getattr(self.sim, "groups", {}).get(group, []))
        allb = {p.getBodyUniqueId(i, physicsClientId=self.sim.client)
                for i in range(p.getNumBodies(physicsClientId=self.sim.client))}
        self._build_from_pybullet(skip=allb - ids)

    # -------------------------------------------------------------- lights
    def _add_lights(self):
        pr, q = self.pr, self.q
        target = np.array([0.03, 0.0, cfg.WALL_Z])
        heads = getattr(self, "_lamp_heads", None) or \
            [np.array([0.0, 0.0, 2.3])]
        for h in heads:
            pose = np.eye(4)
            pose[:3, :3] = trimesh.geometry.align_vectors(
                [0, 0, -1], target - h)[:3, :3]
            pose[:3, 3] = h
            self.scene.add(pr.SpotLight(color=[1.0, 0.98, 0.94],
                                        intensity=q["lamp"] / len(heads),
                                        innerConeAngle=np.radians(10),
                                        outerConeAngle=np.radians(28)),
                           pose=pose)
        # Room light. pyrender's ambient term is very weak, so the diffuse
        # light of the OR's ceiling panels is built from several soft
        # directional lights coming from above at different azimuths, plus a
        # dim one from the side so vertical surfaces (walls, people) read.
        for az in (45, 135, 225, 315):
            d = np.array([np.cos(np.radians(az)) * 0.55,
                          np.sin(np.radians(az)) * 0.55, -1.0])
            pose = np.eye(4)
            pose[:3, :3] = trimesh.geometry.align_vectors([0, 0, -1], d)[:3, :3]
            self.scene.add(pr.DirectionalLight(color=[1.0, 0.99, 0.97],
                                               intensity=q["room"]), pose=pose)
        # (pyrender uses at most 4 directional lights; two former side
        # fills were never actually drawn and only cost time - removed)

    # -------------------------------------------------------------- camera
    def _camera_pose(self, dist, yaw, pitch, target):
        V = np.array(p.computeViewMatrixFromYawPitchRoll(
            target, dist, yaw, pitch, 0, 2)).reshape(4, 4).T
        return np.linalg.inv(V)

    def set_view(self, i):
        self.view_index = i

    def resize(self, size):
        """Change the output size (e.g. the big zoomable OR window)."""
        w, h = int(size[0]), int(size[1])
        self.size = (w, h)
        self.renderer.viewport_width = w * self.ssaa
        self.renderer.viewport_height = h * self.ssaa
        self.cam_node.camera.aspectRatio = w / h
        self._post_lut = None                   # vignette for the new size

    def _gl_info(self):
        """Which GPU is drawing the panel (e.g. NVIDIA vs Intel)."""
        try:
            from OpenGL import GL
            ven = GL.glGetString(GL.GL_VENDOR)
            ren = GL.glGetString(GL.GL_RENDERER)
            return f"{ven.decode()} / {ren.decode()}"
        except Exception as e:                              # noqa: BLE001
            return f"unknown ({e})"

    def set_screen_image(self, bgr):
        """Tower monitors: `bgr` = the endoscope frame, or a pair
        (raw endoscope frame, YOLO26 overlay) -> left screen raw video,
        right screen the segmentation overlay, like the console."""
        if bgr is None or not self._screen_nodes:
            return
        now = time.perf_counter()
        if now - getattr(self, "_screen_t", 0.0) < 0.33:
            return                       # the monitors don't need 30 Hz
        self._screen_t = now
        raw, seg = small_pair(bgr)
        self._screen_imgs = {"screen0": raw, "screen1": seg}
        self._screen_dirty = True

    def _update_screens(self):
        if not getattr(self, "_screen_dirty", True):
            return
        self._screen_dirty = False
        for name, (tm, node) in list(self._screen_nodes.items()):
            img = self._screen_imgs.get(name)
            if img is None:
                if node is not None:
                    continue
                img = np.full((18, 32, 3), (20, 30, 34), np.uint8)
            if node is not None:
                self.scene.remove_node(node)
            tex = self.pr.Texture(source=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                                  source_channels="RGB")
            mat = self.pr.MetallicRoughnessMaterial(
                baseColorFactor=[0.0, 0.0, 0.0, 1.0], metallicFactor=0.0,
                roughnessFactor=0.15, emissiveTexture=tex,
                emissiveFactor=[1.0, 1.0, 1.0])
            node = self.scene.add(self.pr.Mesh.from_trimesh(tm, material=mat))
            self._screen_nodes[name] = (tm, node)

    # -------------------------------------------------------------- render
    def snapshot(self):
        """
        Read everything the picture needs from PyBullet (MAIN thread only:
        PyBullet is not thread-safe). Cheap: ~50 link-state reads.
        """
        cid = self.sim.client
        poses = []
        base_cache = {}
        for node, body, link, local in self.nodes:
            if link == -1:
                if body not in base_cache:
                    pos, orn = p.getBasePositionAndOrientation(
                        body, physicsClientId=cid)
                    base_cache[body] = _quat_to_mat(pos, orn)
                W = base_cache[body]
            else:
                ls = p.getLinkState(body, link, computeForwardKinematics=True,
                                    physicsClientId=cid)
                W = _quat_to_mat(ls[4], ls[5])
            poses.append((node, W @ local))
        for arm, (n1, n2) in getattr(self, "_trocars", {}).items():
            shaft = self.sim._inst[arm][0]
            _, orn = p.getBasePositionAndOrientation(shaft,
                                                     physicsClientId=cid)
            T = _quat_to_mat(self.sim.pivots[arm], orn)
            poses += [(n1, T), (n2, T)]
        tops = {}
        for name in ("camera", "instrument"):
            body = self.sim.arms.get(name)
            if body is None:
                continue
            try:
                nj = p.getNumJoints(body, physicsClientId=cid)
                pts = [p.getLinkState(body, j, physicsClientId=cid)[4]
                       for j in range(min(nj, 5))]
                tops[name] = np.array(max(pts, key=lambda q: q[2])) + \
                    [0, 0, 0.10]
            except Exception:                               # noqa: BLE001
                pass
        return {"poses": poses, "tops": tops,
                "acts": dict(getattr(self.sim, "_label_text", {}) or {}),
                "view": self.view_index}

    def sync(self):
        cid = self.sim.client
        base_cache = {}
        for node, body, link, local in self.nodes:
            if link == -1:
                if body not in base_cache:
                    pos, orn = p.getBasePositionAndOrientation(
                        body, physicsClientId=cid)
                    base_cache[body] = _quat_to_mat(pos, orn)
                W = base_cache[body]
            else:
                ls = p.getLinkState(body, link, computeForwardKinematics=True,
                                    physicsClientId=cid)
                W = _quat_to_mat(ls[4], ls[5])
            self.scene.set_pose(node, W @ local)
        # trocar cannulas follow each held tool's axis through its port
        for arm, (n1, n2) in getattr(self, "_trocars", {}).items():
            shaft = self.sim._inst[arm][0]
            _, orn = p.getBasePositionAndOrientation(shaft,
                                                     physicsClientId=cid)
            T = _quat_to_mat(self.sim.pivots[arm], orn)
            self.scene.set_pose(n1, T)
            self.scene.set_pose(n2, T)

    def render(self):
        """Snapshot + draw in one go (single-threaded use)."""
        return self.draw(self.snapshot())

    def draw(self, snap):
        """
        Draw a snapshot. Only touches OpenGL / pyrender, never PyBullet, so
        it can run on the panel's own thread (PanelWorker).
        """
        for node, T in snap["poses"]:
            self.scene.set_pose(node, T)
        self._pose_twin()
        if self._screen_nodes:
            self._update_screens()
        if snap.get("size") and tuple(snap["size"]) != tuple(self.size):
            self.resize(snap["size"])
        if snap.get("cam") is not None:            # free camera (mouse)
            dist, yaw, pitch, tgt = snap["cam"]
        else:
            name, dist, yaw, pitch, tgt = cfg.OR_VIEWS[
                snap["view"] % len(cfg.OR_VIEWS)]
        self.scene.set_pose(self.cam_node,
                            self._camera_pose(dist, yaw, pitch, tgt))
        flags = self.pr.RenderFlags.NONE
        if self.q["shadows"]:
            flags |= self.pr.RenderFlags.SHADOWS_SPOT
        color, _ = self.renderer.render(self.scene, flags=flags)
        if self.ssaa > 1:
            color = cv2.resize(color, self.size, interpolation=cv2.INTER_AREA)
        if self.q.get("post", False):
            color = self._post(color)
        if getattr(cfg, "OR_CAPTIONS", True):
            color = self._captions(color, dist, yaw, pitch, tgt, snap)
        return color                     # RGB, like Scene.render()

    # ------------------------------------------------------ 2D finishing
    def _post(self, rgb):
        """Gentle filmic contrast curve + lens vignette (cheap LUT/mask)."""
        if getattr(self, "_post_lut", None) is None:
            x = np.arange(256) / 255.0
            s = np.clip(x - 0.20 * np.sin(2 * np.pi * x) / (2 * np.pi), 0, 1)
            self._post_lut = (s * 255).astype(np.uint8)
            h, w = rgb.shape[:2]
            yy, xx = np.mgrid[0:h, 0:w]
            r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
            self._post_vig = (1.0 - 0.16 * np.clip(r2 / 2, 0, 1) ** 1.3
                              )[..., None].astype(np.float32)
        out = cv2.LUT(rgb, self._post_lut)
        if self._post_vig.shape[:2] == out.shape[:2]:
            out = (out * self._post_vig).astype(np.uint8)
        return out

    def _project(self, P, dist, yaw, pitch, tgt):
        """World point -> panel pixel (or None if behind the camera)."""
        pose = self._camera_pose(dist, yaw, pitch, tgt)
        v = np.linalg.inv(pose) @ np.r_[np.asarray(P, float), 1.0]
        if v[2] >= -1e-3:
            return None
        w, h = self.size
        f = (h / 2) / np.tan(np.radians(cfg.OR_FOV) / 2)
        return (w / 2 + f * v[0] / -v[2], h / 2 - f * v[1] / -v[2])

    def _captions(self, rgb, dist, yaw, pitch, tgt, snap):
        """Name tag over each robot arm with its live activity line."""
        out = rgb.copy()
        H, W = out.shape[:2]
        acts = snap.get("acts", {})
        placed = []
        for name, title, sub, col in (
                ("camera", "CAMERA ARM", "endoscope + light, fixed",
                 (60, 130, 230)),
                ("instrument", "INSTRUMENT ARM", "2-finger tool",
                 (240, 140, 40))):
            top = snap.get("tops", {}).get(name)
            if top is None:
                continue
            uv = self._project(top, dist, yaw, pitch, tgt)
            if uv is None:
                continue
            x, y = int(uv[0]), int(uv[1])
            if not (-40 < x < W + 40 and -20 < y < H + 20):
                continue
            line2 = acts.get(name) or sub
            fs = 0.36 if W < 600 else 0.42 * min(1.8, (W / 640) ** 0.6)
            (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX,
                                          fs, 1)
            (sw, sh), _ = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX,
                                          fs * 0.85, 1)
            bw, bh = max(tw, sw) + 16, th + sh + 16
            bx = int(np.clip(x - bw // 2, 4, W - bw - 4))
            by = int(np.clip(y - bh - 14, 4, H - bh - 4))
            for (px, py, pw, ph) in placed:          # keep tags apart
                if bx < px + pw and px < bx + bw and by < py + ph and \
                        py < by + bh:
                    by = py + ph + 4 if py + ph + 4 + bh < H else py - bh - 4
            placed.append((bx, by, bw, bh))
            # leader line + dot on the arm
            cv2.line(out, (bx + bw // 2, by + bh), (x, y), (250, 250, 250), 1,
                     cv2.LINE_AA)
            cv2.circle(out, (x, y), 3, col, -1, cv2.LINE_AA)
            box = out[by:by + bh, bx:bx + bw].astype(np.float32)
            out[by:by + bh, bx:bx + bw] = (box * 0.35 +
                                           np.array([18, 22, 26]) * 0.65
                                           ).astype(np.uint8)
            cv2.rectangle(out, (bx, by), (bx + 3, by + bh - 1), col, -1)
            cv2.putText(out, title, (bx + 9, by + th + 5),
                        cv2.FONT_HERSHEY_DUPLEX, fs, (245, 245, 245), 1,
                        cv2.LINE_AA)
            cv2.putText(out, line2, (bx + 9, by + th + sh + 11),
                        cv2.FONT_HERSHEY_SIMPLEX, fs * 0.85, (190, 200, 205),
                        1, cv2.LINE_AA)
        return out

    def close(self):
        try:
            self.renderer.delete()
        except Exception:                                   # noqa: BLE001
            pass


class PanelWorker:
    """
    Runs the OR panel on its own thread so the console never waits for it.

    The OpenGL context lives on the worker thread (contexts are per-thread),
    so the renderer is built there too. PyBullet is only read on the main
    thread: submit() takes a cheap snapshot of the poses and hands it over;
    the worker draws the newest one whenever it is free (older requests are
    simply dropped). `image` always holds the last finished picture.
    """

    def __init__(self, scene):
        self.image = None
        self.error = None
        self.frames = 0
        self.render_ms = 0.0
        self._snap = None
        self._lock = threading.Lock()
        self._new = threading.Event()
        self._stop = threading.Event()
        self.renderer = None
        ready = threading.Event()

        def run():
            try:
                self.renderer = ORRenderer(scene)
            except Exception as e:                          # noqa: BLE001
                self.error = e
                ready.set()
                return
            ready.set()
            while not self._stop.is_set():
                if not self._new.wait(0.1):
                    continue
                self._new.clear()
                with self._lock:
                    snap, frame = self._snap
                try:
                    t = time.perf_counter()
                    if frame is not None:
                        self.renderer.set_screen_image(frame)
                    self.image = self.renderer.draw(snap)
                    self.render_ms = (time.perf_counter() - t) * 1000
                    self.frames += 1
                except Exception as e:                      # noqa: BLE001
                    self.error = e
                    break
            try:
                self.renderer.close()
            except Exception:                               # noqa: BLE001
                pass

        self._thread = threading.Thread(target=run, name="or-panel",
                                        daemon=True)
        self._thread.start()
        ready.wait()
        if self.error is not None:
            raise self.error
        # first picture right away, so the panel is never empty
        self.submit(None)
        for _ in range(100):
            if self.image is not None or self.error is not None:
                break
            time.sleep(0.05)

    @property
    def busy(self):
        return self._new.is_set()

    cam = None
    size = None

    def set_camera(self, cam):
        self.cam = cam

    def set_size(self, size):
        self.size = size

    def submit(self, frame):
        """Main thread: hand over the current state (never blocks)."""
        snap = self.renderer.snapshot()
        snap["cam"], snap["size"] = self.cam, self.size
        now = time.perf_counter()
        if frame is not None and now - getattr(self, "_ft", 0.0) >= 0.33:
            self._ft = now               # monitors: small copy, ~3 Hz
            frame = small_pair(frame)
        else:
            frame = None
        with self._lock:
            self._snap = (snap, frame)
        self._new.set()

    def set_view(self, i):
        self.renderer.set_view(i)

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2.0)


# ---------------------------------------------------------------------------
# Separate PROCESS for the panel (default).
#
# A thread is not enough: pyrender spends most of each frame in Python
# (one call per mesh, per light, per shadow pass), and Python lets only one
# thread run Python code at a time (the GIL). Drawing on a thread therefore
# steals time from the video / YOLO / control loop and the whole window
# stutters. In its own process the panel gets its own Python and its own
# CPU core, and the console is not slowed at all.
# ---------------------------------------------------------------------------
def _panel_main(conn):
    """Child process: own PyBullet copy (geometry only) + renderer."""
    try:
        from scene import Scene
        sc = Scene(gui=False)
        r = ORRenderer(sc)
        keys = sorted({(b, l) for _, b, l, _ in r.nodes})
        idx = {k: i for i, k in enumerate(keys)}
        conn.send(("ready", keys, r._gl_info()))
    except Exception as e:                                  # noqa: BLE001
        import traceback
        conn.send(("error", f"{type(e).__name__}: {e}",
                   traceback.format_exc()))
        return
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            break
        if msg is None:
            break
        Ws, troc, tops, acts, view, frame, cam, size = msg
        try:
            t = time.perf_counter()
            poses = [(node, Ws[idx[(b, l)]] @ local)
                     for node, b, l, local in r.nodes]
            for arm, (n1, n2) in getattr(r, "_trocars", {}).items():
                T = troc.get(arm)
                if T is not None:
                    poses += [(n1, T), (n2, T)]
            if frame is not None:
                r._screen_t = 0.0                # parent already throttles
                r.set_screen_image(frame)
            img = r.draw({"poses": poses, "tops": tops, "acts": acts,
                          "view": view, "cam": cam, "size": size})
            conn.send(("img", img, (time.perf_counter() - t) * 1000))
        except Exception as e:                              # noqa: BLE001
            conn.send(("error", f"{type(e).__name__}: {e}", ""))
            break
    r.close()


MONITOR_RES = (640, 360)


def small_pair(img):
    """(raw, overlay) or a single image -> two 640x360 monitor images
    (the overlay is drawn at 640 px wide, so it is used as-is: thin YOLO
    contours stay sharp instead of being shrunk away)."""
    if img is None:
        return None
    if isinstance(img, (tuple, list)):
        raw, seg = img[0], (img[1] if len(img) > 1 else None)
    else:
        raw, seg = img, None
    W, H = MONITOR_RES
    if raw is not None and raw.shape[:2] != (H, W):
        raw = cv2.resize(raw, (W, H), interpolation=cv2.INTER_AREA)
    if seg is None:
        seg = raw
    elif seg.shape[:2] != (H, W):
        seg = cv2.resize(seg, (W, H), interpolation=cv2.INTER_AREA)
    return raw, seg


def world_poses(sim, keys):
    """World transforms of the listed (body, link) pairs + trocar poses
    (main process: PyBullet is only read here)."""
    cid = sim.client
    Ws = np.empty((len(keys), 4, 4))
    base = {}
    for i, (b, l) in enumerate(keys):
        if l == -1:
            if b not in base:
                pos, orn = p.getBasePositionAndOrientation(
                    b, physicsClientId=cid)
                base[b] = _quat_to_mat(pos, orn)
            Ws[i] = base[b]
        else:
            ls = p.getLinkState(b, l, computeForwardKinematics=True,
                                physicsClientId=cid)
            Ws[i] = _quat_to_mat(ls[4], ls[5])
    troc = {}
    for arm, pv in getattr(sim, "pivots", {}).items():
        try:
            shaft = sim._inst[arm][0]
            _, orn = p.getBasePositionAndOrientation(shaft,
                                                     physicsClientId=cid)
            troc[arm] = _quat_to_mat(pv, orn)
        except Exception:                                   # noqa: BLE001
            pass
    return Ws, troc


class PanelProcess:
    """Same interface as PanelWorker, but drawing runs in another process."""

    def __init__(self, scene, timeout=120.0):
        import multiprocessing as mp
        self.sim = scene
        self.error = None
        self.render_ms = 0.0
        self.gl_info = ""
        self._image = None
        self._frames = 0
        self._busy = False
        self._view = 0
        self._ft = 0.0
        ctx = mp.get_context("spawn")
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(target=_panel_main, args=(child,),
                                 name="or-panel", daemon=True)
        self._proc.start()
        t0 = time.perf_counter()
        while not self._conn.poll(0.2):
            if not self._proc.is_alive():
                raise RuntimeError("panel process exited during start-up")
            if time.perf_counter() - t0 > timeout:
                self._proc.terminate()
                raise RuntimeError("panel process start-up timed out")
        msg = self._conn.recv()
        if msg[0] == "error":
            raise RuntimeError(f"{msg[1]}\n--- panel process traceback ---\n"
                               f"{msg[2]}")
        _, self._keys, self.gl_info = msg
        _warn(f"panel runs in its own process; OpenGL: {self.gl_info}")
        self.submit(None)
        t0 = time.perf_counter()
        while self._image is None and self.error is None and \
                time.perf_counter() - t0 < 20:
            self._poll(0.05)

    # -- results -----------------------------------------------------------
    def _poll(self, wait=0.0):
        while self._conn.poll(wait):
            msg = self._conn.recv()
            if msg[0] == "img":
                self._image, self.render_ms = msg[1], msg[2]
                self._frames += 1
                self._busy = False
            elif msg[0] == "error":
                self.error = RuntimeError(msg[1])
                self._busy = False
            wait = 0.0

    @property
    def image(self):
        self._poll()
        return self._image

    @property
    def frames(self):
        self._poll()
        return self._frames

    @property
    def busy(self):
        self._poll()
        return self._busy

    # -- requests ----------------------------------------------------------
    def submit(self, frame):
        """Main thread: send the current poses (never waits)."""
        self._poll()
        if self._busy or self.error is not None:
            return
        cid = self.sim.client
        Ws, troc = world_poses(self.sim, self._keys)
        tops = {}
        for name in ("camera", "instrument"):
            body = self.sim.arms.get(name)
            if body is None:
                continue
            nj = p.getNumJoints(body, physicsClientId=cid)
            pts = [p.getLinkState(body, j, physicsClientId=cid)[4]
                   for j in range(min(nj, 5))]
            tops[name] = np.array(max(pts, key=lambda q: q[2])) + [0, 0, 0.1]
        now = time.perf_counter()
        if frame is not None and now - self._ft >= 0.33:
            self._ft = now
            frame = small_pair(frame)
        else:
            frame = None
        acts = dict(getattr(self.sim, "_label_text", {}) or {})
        try:
            self._conn.send((Ws, troc, tops, acts, self._view, frame,
                             self._cam, self._size))
            self._busy = True
        except (OSError, EOFError) as e:
            self.error = e

    def set_view(self, i):
        self._view = i

    _cam = None
    _size = None

    def set_camera(self, cam):
        self._cam = cam

    def set_size(self, size):
        self._size = size

    def close(self):
        try:
            self._conn.send(None)
        except Exception:                                   # noqa: BLE001
            pass
        self._proc.join(timeout=3.0)
        if self._proc.is_alive():
            self._proc.terminate()
