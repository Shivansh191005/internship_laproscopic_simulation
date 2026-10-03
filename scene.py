"""
Operating-room scene, built from PyBullet primitives (no external meshes).

Looking down (+x toward the patient's head):

        monitors                 |  [cart A]   [cart B]
            []  []               |      \\        /
                                 |   .----------------.
                                 |   | draped patient |
                                 |   '----------------'
                                 |        [ surgeon ]
"""
import os

import numpy as np
import pybullet as p
import pybullet_data

import config as cfg
from rcm import quat_from_dir, pivot_error

# Drapes and scrubs MUST differ in hue. At a colour distance of 0.14 the
# patient's raised legs merged into the surgeon's silhouette.
DRAPE  = [0.38, 0.62, 0.84, 1.0]      # light surgical blue
# The abdomen is drawn translucent ("x-ray cutaway"). The trocar ports sit at
# z=0.95 inside a torso whose surface is at z=1.01, so with an opaque body the
# instrument tips - the part that actually does the work - were invisible.
DRAPE_CUT = [0.38, 0.62, 0.84, 0.28]
SKIN   = [0.88, 0.70, 0.60, 1.0]
STEEL  = [0.62, 0.64, 0.68, 1.0]
SCRUBS = [0.20, 0.46, 0.38, 1.0]      # surgical green
DARK   = [0.10, 0.11, 0.13, 1.0]
FLOOR  = [0.60, 0.64, 0.66, 1.0]
WALLC  = [0.84, 0.89, 0.87, 1.0]      # pale OR green
CEIL   = [0.93, 0.94, 0.95, 1.0]


def _box(half, pos, colour, orn=None):
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=colour)
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
    return p.createMultiBody(0, col, vis, pos, orn or [0, 0, 0, 1])


def _cyl(r, h, pos, colour, orn=None):
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=h,
                              rgbaColor=colour)
    return p.createMultiBody(0, -1, vis, pos, orn or [0, 0, 0, 1])


def _sphere(r, pos, colour, scale=None):
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=colour)
    b = p.createMultiBody(0, -1, vis, pos)
    return b


def _capsule(r, h, pos, colour, orn=None):
    vis = p.createVisualShape(p.GEOM_CAPSULE, radius=r, length=h,
                              rgbaColor=colour)
    return p.createMultiBody(0, -1, vis, pos, orn or [0, 0, 0, 1])


def _link(p0, p1, r, colour):
    """Capsule spanning two points - the natural way to build a limb."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = float(np.linalg.norm(d))
    if L < 1e-6:
        return None
    mid = (p0 + p1) / 2.0
    return _capsule(r, L, mid.tolist(), colour, quat_from_dir(d / L))


def _try_mesh(path, pos, colour, scale=1.0, yaw=0.0):
    """
    Load an external mesh if the user has supplied one.

    PyBullet renders primitives, not people - a truly photoreal patient or
    surgeon needs an actual mesh. Drop an .obj into assets/ and it is used
    automatically; otherwise the primitive figure below stands in.
    """
    if not os.path.isfile(path):
        return None
    try:
        vis = p.createVisualShape(p.GEOM_MESH, fileName=path,
                                  meshScale=[scale] * 3, rgbaColor=colour)
        return p.createMultiBody(0, -1, vis, pos,
                                 p.getQuaternionFromEuler([0, 0, yaw]))
    except Exception as e:                                  # noqa: BLE001
        print(f"[scene] mesh {path} failed to load: {e}")
        return None


JOINT_LOWER = [-2.96, -2.09, -2.96, -2.09, -2.96, -2.09, -3.05]
JOINT_UPPER = [2.96, 2.09, 2.96, 2.09, 2.96, 2.09, 3.05]


def dls_ik(cid, body, Rb, target, axis, seed, home=None, iters=None):
    """
    Damped-least-squares IK on position AND tool axis (5 constraints; roll
    about the shaft is left free - that is the arm's spare freedom).

    Replaces a single p.calculateInverseKinematics call with restPoses. That
    call does one null-space pass and stops; for poses where the grip point
    sits near the arm's own shoulder it returned solutions 47-54 mm off target
    (self-test D: "IK error 53.8 mm", "physical RCM 55.7 mm"). Iterating from
    the previous solution converges to < 0.1 mm wherever the pose is
    reachable, and stays on the same elbow branch so the arm cannot flip.
    Where a pose is NOT reachable it returns the nearest pose and the error,
    so the caller can refuse it instead of tearing the trocar.

    `body` lives in physics client `cid` (a DIRECT client used only for IK),
    so joints can be set freely without disturbing the simulated arms.
    Returns (q, position_error_m).
    """
    lo, hi = np.array(JOINT_LOWER), np.array(JOINT_UPPER)
    q = np.clip(np.array(seed, dtype=float), lo, hi)
    target = np.asarray(target, dtype=float)
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    lam2 = cfg.IK_DAMPING ** 2
    err = np.inf
    zero = [0.0] * 7
    for _ in range(iters or cfg.IK_ITERS):
        for j in range(7):
            p.resetJointState(body, j, q[j], physicsClientId=cid)
        ls = p.getLinkState(body, cfg.EE_LINK, computeForwardKinematics=True,
                            physicsClientId=cid)
        pos = np.array(ls[4])
        z = np.array(p.getMatrixFromQuaternion(ls[5])).reshape(3, 3)[:, 2]
        e_pos = target - pos
        e_rot = np.cross(z, axis)
        err = float(np.linalg.norm(e_pos))
        if err < 1e-4 and np.linalg.norm(e_rot) < 5e-4 and z @ axis > 0:
            break
        # PyBullet returns the Jacobian in the BASE frame; rotate to world.
        jv, jw = p.calculateJacobian(body, cfg.EE_LINK, [0, 0, 0], list(q),
                                     zero, zero, physicsClientId=cid)
        J = np.vstack([Rb @ np.array(jv), Rb @ np.array(jw)])
        Jp = J.T @ np.linalg.inv(J @ J.T + lam2 * np.eye(6))
        dq = Jp @ np.hstack([e_pos, e_rot])
        if home is not None and cfg.IK_POSTURE_GAIN > 0:
            # Weak pull toward the home posture inside the null space: it
            # changes nothing about the tool pose and stops slow elbow drift.
            dq += (np.eye(7) - Jp @ J) @ (cfg.IK_POSTURE_GAIN *
                                          (np.asarray(home) - q))
        q = np.clip(q + np.clip(dq, -0.25, 0.25), lo, hi)
    # final error including orientation
    for j in range(7):
        p.resetJointState(body, j, q[j], physicsClientId=cid)
    ls = p.getLinkState(body, cfg.EE_LINK, computeForwardKinematics=True,
                        physicsClientId=cid)
    z = np.array(p.getMatrixFromQuaternion(ls[5])).reshape(3, 3)[:, 2]
    err = float(np.linalg.norm(target - np.array(ls[4])))
    # Count axis misalignment as distance at the trocar (grip-to-port length).
    ang = float(np.arccos(np.clip(z @ axis, -1.0, 1.0)))
    err = max(err, ang * cfg.MIN_OUTSIDE)
    return list(q), err


def home_seed(base, pivot):
    """Shoulder turned toward the trocar, elbow up: a clean starting branch."""
    yaw = float(np.arctan2(pivot[1] - base[1], pivot[0] - base[0]))
    return [yaw, 0.75, 0.0, -1.5, 0.0, 0.9, 0.0]


class Scene:
    def __init__(self, gui: bool = True):
        self.gui = gui
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(fixedTimeStep=1.0 / cfg.SIM_HZ)
        if gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            # Shadows off: walls otherwise shadow most of the floor and the
            # scene renders murky from inside the room.
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
            self.set_view(0)

        # Which bodies make up which part of the scene, so the realistic
        # renderer (or_render.py) can swap the primitive figures for proper
        # models. Bookkeeping only: nothing here changes the simulation.
        self.groups = {}

        def _grouped(name, fn, *a):
            n0 = p.getNumBodies()
            fn(*a)
            self.groups.setdefault(name, []).extend(
                p.getBodyUniqueId(i) for i in range(n0, p.getNumBodies()))

        _grouped("room", self._build_room)
        _grouped("table", self._build_table)
        _grouped("patient", self._build_patient)
        _grouped("surgeon", self._build_surgeon)
        _grouped("monitors", self._build_monitors)

        self.arms, self.pivots, self._bases = {}, {}, {}
        for name, cart, pivot in [
                ("camera", cfg.CART_CAMERA, cfg.TROCAR_CAMERA),
                ("instrument", cfg.CART_INSTRUMENT, cfg.TROCAR_INSTRUMENT)]:
            _grouped("carts", self._build_cart, cart)
            self.arms[name] = self._load_arm(cart)
            self._bases[name] = np.asarray(cart, dtype=float)
            self.pivots[name] = np.asarray(pivot, dtype=float)
        _grouped("trocars", self._draw_trocars)
        self._tool_lines = {}
        self._build_instruments()
        self._cone = {}
        self._labels = {}
        self._label_text = {}
        if self.gui:
            self._build_labels()
        # KUKA iiwa joint limits, needed for null-space IK.
        self._lower, self._upper = JOINT_LOWER, JOINT_UPPER
        self._ranges = [u - l for l, u in zip(self._lower, self._upper)]
        # Last accepted joint solution per arm, used both as the IK seed and
        # as the filter state. This is what stops the arm shaking.
        self._last_q = {n: [0.0] * 7 for n in self.arms}
        self._ik_q, self._home_q, self._last_good = {}, {}, {}
        self.reach_limited = {n: False for n in self.arms}
        self._make_ik_ghosts()
        self._disable_arm_contacts()
        self._home_arms()

    def _disable_arm_contacts(self):
        """
        This is a KINEMATIC simulation: the arms follow IK targets and should
        never be shoved around by contact forces. With contacts on, the two
        arms' wrists (whose trocars are only ~16 cm apart) jammed against each
        other and the joint motors fought the contact - the arms shook, stalled
        short of their targets and the RCM deviation jumped. Contacts between
        the arms, and between the arms and the room/table/cart boxes, are
        switched off. The arm-to-arm CLEARANCE is still measured
        (getClosestPoints ignores these filters) and the controller still
        stops the instrument arm moving any closer.
        """
        arms = list(self.arms.values())
        others = [b for b in range(p.getNumBodies())
                  if p.getBodyUniqueId(b) not in arms]
        for i, a in enumerate(arms):
            links = range(-1, p.getNumJoints(a))
            for b in arms[i + 1:]:
                for la in links:
                    for lb in range(-1, p.getNumJoints(b)):
                        p.setCollisionFilterPair(a, b, la, lb, 0)
            for o in others:
                o = p.getBodyUniqueId(o)
                for la in links:
                    p.setCollisionFilterPair(a, o, la, -1, 0)

    def _home_arms(self):
        """
        Drive each arm to a valid starting configuration before the control
        loop begins, and seed the IK from it.

        The home pose is solved from an explicit "shoulder toward the trocar,
        elbow up and out" seed. PyBullet's own IK started from the URDF zero
        pose often lands on a folded branch with the two elbows meeting over
        the patient (the self-test measured 0.0 cm arm clearance at rest).
        """
        from rcm import RCMState
        for name, pivot in self.pivots.items():
            st = RCMState(pivot, insertion=0.13)
            _, shaft_base, d = st.pose()
            seed = home_seed(self._bases[name], pivot)
            q, err = self._solve_ik(name, shaft_base, d, seed, iters=400)
            for j, v in enumerate(q):
                p.resetJointState(self.arms[name], j, v)
                p.setJointMotorControl2(self.arms[name], j,
                                        p.POSITION_CONTROL,
                                        targetPosition=v, force=300)
            self._last_q[name] = list(q)
            self._ik_q[name] = list(q)
            self._home_q[name] = list(q)
            print(f"[scene] {name} arm homed (IK error {err * 1000:.2f} mm)")

    def place(self, name, state):
        """
        Put an arm straight into the pose for `state` (no motion), e.g. the
        fixed camera pose or the instrument's start pose. Without this the
        arms start upright and swing across on the first frames, which shows
        up as a large, meaningless RCM deviation at start-up.
        """
        _, shaft_base, d = state.pose()
        q, err = self._solve_ik(name, shaft_base, d, self._ik_q[name],
                                iters=400)
        for j, v in enumerate(q):
            p.resetJointState(self.arms[name], j, v)
            p.setJointMotorControl2(self.arms[name], j, p.POSITION_CONTROL,
                                    targetPosition=v, force=300)
        self._last_q[name] = list(q)
        self._ik_q[name] = list(q)
        self._last_good[name] = (state.tilt_x, state.tilt_y, state.insertion)
        return err

    # ------------------------------------------------------------------ IK
    def _make_ik_ghosts(self):
        """
        IK runs on invisible copies of the arms in a separate DIRECT physics
        client. That lets the solver iterate to convergence (set joints, read
        forward kinematics, step) without teleporting the real, dynamically
        simulated arms - and without the two real arms colliding mid-solve.
        """
        self._ik_cid = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self._ik_cid)
        self._ghost = {}
        for name, arm in self.arms.items():
            # NOT getBasePositionAndOrientation: that returns the base link's
            # centre of mass, which sits ~12 cm from the URDF base frame and
            # would put every ghost solution 12 cm off.
            pos, orn = self._bases[name].tolist(), [0, 0, 0, 1]
            g = p.loadURDF("kuka_iiwa/model.urdf", basePosition=pos,
                           baseOrientation=orn, useFixedBase=True,
                           physicsClientId=self._ik_cid)
            # PyBullet's Jacobian is expressed in the BASE frame; rotate it
            # to world so the solver also works for rotated mounts.
            Rb = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
            self._ghost[name] = (g, Rb)

    def _solve_ik(self, name, target, axis, seed, iters=None):
        """See dls_ik() below. Runs on this arm's invisible IK ghost."""
        g, Rb = self._ghost[name]
        return dls_ik(self._ik_cid, g, Rb, target, axis, seed,
                      home=self._home_q.get(name), iters=iters)

    def _build_room(self):
        """
        A plain OR room in place of the checkerboard ground plane: floor,
        four walls, ceiling with a light fixture. Built from boxes, same as
        everything else here - no textures or external assets required.
        """
        hx, hy, hz = cfg.ROOM_SIZE
        _box([hx, hy, 0.02], [0, 0, -0.02], FLOOR)            # floor
        # No ceiling: it blocks the orbit camera looking down and adds nothing.
        _box([0.02, hy, hz], [-hx, 0, hz], WALLC)             # -x wall
        _box([0.02, hy, hz], [hx, 0, hz], WALLC)               # +x wall
        _box([hx, 0.02, hz], [0, -hy, hz], WALLC)              # -y wall
        _box([hx, 0.02, hz], [0, hy, hz], [0.85, 0.87, 0.89, 1.0])  # +y wall
        # a low equipment shelf along the back wall for visual interest
        _box([0.55, 0.14, 0.02], [-hx + 0.16, hy - 0.16, 0.92],
             [0.72, 0.74, 0.77, 1.0])

    def _build_table(self):
        t = cfg.TABLE_TOP_Z
        _box([0.72, 0.26, 0.03], [0.0, 0.0, t - 0.03], DARK)
        _cyl(0.09, t - 0.06, [0.0, 0.0, (t - 0.06) / 2], STEEL)
        _cyl(0.30, 0.04, [0.0, 0.0, 0.02], STEEL)

    def _build_patient(self):
        """
        Supine patient, head at +x, draped except for the abdominal window.
        Proportions are roughly a 1.7 m adult so the trocar spacing and the
        arms' reach stay physically honest.
        """
        t = cfg.TABLE_TOP_Z
        z = t + 0.10                       # mid-height of the body
        if _try_mesh(cfg.PATIENT_MESH, [0.0, 0.0, t], SKIN,
                     cfg.PATIENT_MESH_SCALE, np.pi / 2) is not None:
            _box([0.17, 0.13, 0.004], [0.0, 0.0, cfg.WALL_Z + 0.004],
                 [0.93, 0.86, 0.72, 0.92])
            return

        # head and neck, exposed above the drape
        _sphere(0.098, [0.62, 0.0, z + 0.03], SKIN)
        _sphere(0.030, [0.70, 0.0, z + 0.01], SKIN)          # chin/jaw
        for sgn in (-1, 1):
            _sphere(0.016, [0.665, sgn * 0.052, z + 0.055],
                    [0.30, 0.26, 0.24, 1.0])                 # eyes
        _sphere(0.092, [0.60, 0.0, z + 0.055], [0.35, 0.28, 0.22, 1.0])  # hair
        _link([0.53, 0.0, z + 0.01], [0.46, 0.0, z], 0.055, SKIN)

        # chest tapering to waist, then pelvis
        _link([0.44, 0.0, z], [0.18, 0.0, z], 0.145, DRAPE)   # chest
        _link([0.18, 0.0, z], [-0.06, 0.0, z], 0.132, DRAPE_CUT)  # abdomen
        _link([-0.06, 0.0, z], [-0.26, 0.0, z - 0.01], 0.140, DRAPE_CUT)  # pelvis

        # arms tucked at the sides, as in laparoscopic positioning
        for sgn in (-1, 1):
            _link([0.40, sgn * 0.15, z - 0.02], [0.10, sgn * 0.19, z - 0.03],
                  0.050, DRAPE)
            _link([0.10, sgn * 0.19, z - 0.03], [-0.16, sgn * 0.17, z - 0.03],
                  0.044, DRAPE)

        # LOW lithotomy, the position actually used for laparoscopic
        # gynaecology: thighs abducted and only slightly raised so they do not
        # obstruct the instruments or the assist arms. (High lithotomy put the
        # ankles 1.3 m up, straight through the arms' working volume.)
        for sgn in (-1, 1):
            hip = [-0.26, sgn * 0.10, z]
            knee = [-0.56, sgn * 0.30, z + 0.12]
            ankle = [-0.84, sgn * 0.40, z + 0.04]
            _link(hip, knee, 0.082, DRAPE)          # thigh
            _link(knee, ankle, 0.058, DRAPE)        # calf
            _sphere(0.058, knee, DRAPE)
            _link(ankle, [ankle[0] - 0.02, ankle[1], ankle[2] + 0.14],
                  0.040, DRAPE)                     # foot in a boot stirrup
            _link([ankle[0], ankle[1] + sgn * 0.04, 0.05],
                  [ankle[0], ankle[1] + sgn * 0.04, ankle[2] - 0.04],
                  0.018, STEEL)                     # stirrup post

        # sterile field window over the abdomen (translucent, see DRAPE_CUT)
        _box([0.17, 0.13, 0.004], [0.0, 0.0, cfg.WALL_Z + 0.004],
             [0.93, 0.86, 0.72, 0.25])

    def _build_surgeon(self):
        """Standing surgeon, ~1.75 m, hands reaching toward the abdomen."""
        x, y = cfg.SURGEON_POS
        if _try_mesh(cfg.SURGEON_MESH, [x, y, 0.0], SCRUBS,
                     cfg.SURGEON_MESH_SCALE, np.pi / 2) is not None:
            return

        GLOVE = [0.93, 0.93, 0.96, 1.0]
        SKINT = [0.80, 0.63, 0.53, 1.0]

        # legs and feet
        for sgn in (-1, 1):
            _link([x, y + sgn * 0.09, 0.92], [x + sgn * 0.0, y + sgn * 0.10, 0.50],
                  0.075, SCRUBS)
            _link([x, y + sgn * 0.10, 0.50], [x, y + sgn * 0.10, 0.08],
                  0.058, SCRUBS)
            _link([x, y + sgn * 0.10, 0.06], [x + 0.12, y + sgn * 0.10, 0.04],
                  0.048, [0.15, 0.16, 0.18, 1.0])

        # torso: hips -> chest -> shoulders (shoulders span y, facing +x)
        _link([x, y, 0.90], [x, y, 1.12], 0.145, SCRUBS)
        _link([x, y, 1.12], [x, y, 1.42], 0.160, SCRUBS)
        _link([x, y - 0.19, 1.42], [x, y + 0.19, 1.42], 0.070, SCRUBS)

        # neck and head. FRONT is +x (toward the patient) - every asymmetric
        # feature (eyes, mask, cap peak) is offset in +x so the face reads
        # unambiguously from any camera angle, not just from directly above.
        _link([x, y, 1.42], [x, y, 1.52], 0.048, SKINT)
        _sphere(0.098, [x, y, 1.60], SKINT)
        # scrub cap: a shell that leaves the face open, plus a small peak/brim
        _sphere(0.103, [x - 0.01, y, 1.635], [0.30, 0.58, 0.48, 1.0])
        _box([0.05, 0.11, 0.018], [x + 0.03, y, 1.685],
             [0.26, 0.52, 0.43, 1.0])                             # cap brim
        for sgn in (-1, 1):
            _sphere(0.014, [x + 0.085, y + sgn * 0.038, 1.615],
                    [0.18, 0.14, 0.12, 1.0])                      # eyes
        _box([0.045, 0.078, 0.045], [x + 0.075, y, 1.565],
             [0.80, 0.85, 0.88, 1.0])                              # mask

        # arms reaching in toward the abdominal window
        for sgn in (-1, 1):
            shoulder = [x, y + sgn * 0.19, 1.40]
            hand = [x + 0.46, y + sgn * 0.13, 1.06]
            sh, hd = np.array(shoulder), np.array(hand)
            elbow = (sh + (hd - sh) * 0.52 + np.array([-0.05, sgn * 0.07, 0.03]))
            _link(sh, elbow, 0.055, SCRUBS)
            _link(elbow, hd, 0.046, SCRUBS)
            _sphere(0.048, hd, GLOVE)

            # The surgeon's own instruments. Laparoscopic tools are long: the
            # handles stay well outside the body and only the tip enters at the
            # trocar, which is why the hands sit back from the abdomen.
            port = np.array([cfg.SURGEON_PORTS[0],
                             sgn * cfg.SURGEON_PORTS[1], cfg.WALL_Z])
            shaft_dir = port - hd
            shaft_dir = shaft_dir / np.linalg.norm(shaft_dir)
            _link(hd, port + shaft_dir * 0.06, 0.010,
                  [0.72, 0.74, 0.78, 1.0])

    def _build_monitors(self):
        x, y, z = cfg.MONITOR_POS
        yaw = cfg.MONITOR_YAW
        q = p.getQuaternionFromEuler([0, 0, yaw])
        _cyl(0.05, z, [x, y, z / 2], STEEL)
        _cyl(0.22, 0.03, [x, y, 0.02], STEEL)
        c, sn = np.cos(yaw), np.sin(yaw)
        for i, s in enumerate((-1, 1)):
            ox, oy = -s * 0.27 * sn, s * 0.27 * c
            _box([0.02, 0.25, 0.16], [x + ox, y + oy, z], DARK, orn=q)
            _box([0.008, 0.23, 0.14],
                 [x + ox + 0.023 * c, y + oy + 0.023 * sn, z],
                 [0.18, 0.26, 0.22, 1.0] if i == 0 else [0.15, 0.29, 0.34, 1.0],
                 orn=q)

    def _build_cart(self, base):
        x, y, z = base
        _box([0.13, 0.13, z / 2], [x, y, z / 2], [0.22, 0.24, 0.27, 1.0])
        _box([0.17, 0.17, 0.02], [x, y, 0.02], [0.18, 0.19, 0.22, 1.0])

    def _load_arm(self, base_pos):
        arm = p.loadURDF("kuka_iiwa/model.urdf",
                         basePosition=list(base_pos), useFixedBase=True)
        for j in range(p.getNumJoints(arm)):
            p.setJointMotorControl2(arm, j, p.POSITION_CONTROL,
                                    targetPosition=0.0, force=250)
        return arm

    def _build_instruments(self):
        """
        What each arm is actually holding, as a real body rather than a thin
        debug line:
            camera arm     - black rigid endoscope with a light at the tip
            instrument arm - silver surgical instrument, steel jaws
        Visual only (no collision), repositioned every control step.
        """
        spec = {
            "camera": ([0.07, 0.07, 0.08, 1.0], [1.0, 0.95, 0.55, 1.0]),
            "instrument": ([0.76, 0.78, 0.82, 1.0], [0.95, 0.85, 0.20, 1.0]),
        }
        self._inst = {}
        for name, (shaft_c, tip_c) in spec.items():
            vs = p.createVisualShape(p.GEOM_CYLINDER, radius=0.0055,
                                     length=cfg.INSTRUMENT_LENGTH,
                                     rgbaColor=shaft_c)
            shaft = p.createMultiBody(0, -1, vs, [0, 0, -5])
            vt = p.createVisualShape(p.GEOM_SPHERE, radius=0.011,
                                     rgbaColor=tip_c)
            tipb = p.createMultiBody(0, -1, vt, [0, 0, -5])
            self._inst[name] = (shaft, tipb)
        self._build_fingers()

    def _build_fingers(self):
        """
        The instrument arm's head: ONE cannula through ONE port, with two
        articulated fingers coming out of its end (single-port style). Each
        finger has two rigid links (cfg.FINGER_LINKS) and a jaw. Visual only;
        placed every control step by set_fingers().
        """
        cols = [[0.98, 0.85, 0.20, 1.0], [0.95, 0.40, 0.90, 1.0]]  # tool 1/2
        self._fingers = []
        for col in cols:
            segs = []
            for L, r in zip(cfg.FINGER_LINKS, (0.0045, 0.0038)):
                v = p.createVisualShape(p.GEOM_CAPSULE, radius=r,
                                        length=max(L - 2 * r, 1e-3),
                                        rgbaColor=[0.70, 0.72, 0.76, 1.0])
                segs.append(p.createMultiBody(0, -1, v, [0, 0, -5]))
            vj = p.createVisualShape(p.GEOM_SPHERE, radius=0.006,
                                     rgbaColor=col)
            jaw = p.createMultiBody(0, -1, vj, [0, 0, -5])
            self._fingers.append((segs, jaw))

    def set_fingers(self, base, ends, elbows):
        """
        Place both fingers: from `base` (end of the cannula) through `elbows`
        to `ends` (each jaw). An entry of None hides that finger.
        """
        for (segs, jaw), end, elbow in zip(self._fingers, ends, elbows):
            if end is None:
                for sgb in segs:
                    p.resetBasePositionAndOrientation(sgb, [0, 0, -5],
                                                      [0, 0, 0, 1])
                p.resetBasePositionAndOrientation(jaw, [0, 0, -5],
                                                  [0, 0, 0, 1])
                continue
            for sgb, a, b in ((segs[0], base, elbow), (segs[1], elbow, end)):
                a, b = np.asarray(a, float), np.asarray(b, float)
                L = float(np.linalg.norm(b - a))
                p.resetBasePositionAndOrientation(
                    sgb, ((a + b) / 2).tolist(),
                    quat_from_dir((b - a) / max(L, 1e-6)))
            p.resetBasePositionAndOrientation(jaw, list(map(float, end)),
                                              [0, 0, 0, 1])

    def _build_labels(self):
        """Floating captions so it is obvious which arm is which."""
        x, y = cfg.SURGEON_POS
        p.addUserDebugText("SURGEON", [x, y, 1.86], [0.15, 0.45, 0.30], 1.3)
        for name, cart, col in [
                ("camera", cfg.CART_CAMERA, [0.10, 0.35, 0.85]),
                ("instrument", cfg.CART_INSTRUMENT, [0.85, 0.45, 0.05])]:
            title = ("CAMERA ARM  (endoscope + light, fixed)"
                     if name == "camera"
                     else "INSTRUMENT ARM  (surgical tool)")
            base = [cart[0], cart[1] * 1.25, 1.78]
            p.addUserDebugText(title, base, col, 1.15)
            self._labels[name] = p.addUserDebugText(
                "starting", [base[0], base[1], base[2] - 0.08],
                [0.25, 0.25, 0.28], 1.0)

    def annotate(self, name, text, colour=(0.25, 0.25, 0.28)):
        """Update an arm's live activity caption (only when it changes)."""
        if self._label_text.get(name) == text:
            return
        self._label_text[name] = text        # also read by the OR panel
        if not self.gui:
            return
        cart = cfg.CART_CAMERA if name == "camera" else cfg.CART_INSTRUMENT
        pos = [cart[0], cart[1] * 1.25, 1.70]
        self._labels[name] = p.addUserDebugText(
            text, pos, list(colour), 1.0,
            replaceItemUniqueId=self._labels.get(name, -1))
        self._label_text[name] = text

    def _draw_view_cone(self, tip, d):
        """Yellow cone from the scope tip: where the camera is looking."""
        if not self.gui:
            return
        ref = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0, 0])
        u = np.cross(d, ref)
        u /= np.linalg.norm(u)
        v = np.cross(d, u)
        half = np.radians(cfg.SCOPE_HALF_FOV)
        L = cfg.SCOPE_CONE_LEN
        for i, sdir in enumerate((u, -u, v, -v)):
            end = tip + L * (np.cos(half) * d + np.sin(half) * sdir)
            self._cone[i] = p.addUserDebugLine(
                tip.tolist(), end.tolist(), [1.0, 0.85, 0.2], 2,
                replaceItemUniqueId=self._cone.get(i, -1))

    def _draw_trocars(self):
        for name, pv in self.pivots.items():
            c = [0.1, 0.5, 1.0] if name == "camera" else [1.0, 0.6, 0.1]
            _sphere(0.010, list(pv), c + [1.0])

    def drive(self, name, state):
        """
        Solve IK and command the arm.

        A 7-DoF arm is redundant: infinitely many joint configurations reach the
        same tool pose. Called naively, the solver hops between elbow branches
        from frame to frame, which looks like violent shaking even when the tool
        tip is perfectly smooth. Two things prevent that:

          1. Iterative DLS IK seeded with the PREVIOUS solution, so the solver
             stays on the configuration already in use.
          2. A low-pass filter on the joint targets themselves.
        Poses the arm cannot hold through its trocar are refused and the RCM
        state is put back to the last reachable one (see reach_limited).
        """
        arm = self.arms[name]
        tip, shaft_base, d = state.pose()
        rest = self._last_q[name]

        # Seed from the previous IK solution (not the filtered command), so
        # the solver keeps converging even while the joints are being smoothed.
        joints, err = self._solve_ik(name, shaft_base, d, self._ik_q[name])
        if err > cfg.IK_TOL:
            # The arm physically cannot hold this pose through the trocar
            # (typically a steep tilt back toward its own cart: the grip point
            # would have to sit inside the shoulder). Refuse it: put the RCM
            # state back to the last pose the arm CAN hold. The controllers
            # see their command clamped, exactly like a joint limit, instead
            # of the arm silently drifting off the port.
            good = self._last_good.get(name)
            self.reach_limited[name] = True
            if good is not None:
                state.tilt_x, state.tilt_y, state.insertion = good
                tip, shaft_base, d = state.pose()
                joints, err = self._solve_ik(name, shaft_base, d,
                                             self._ik_q[name])
        else:
            self.reach_limited[name] = False
        if err <= cfg.IK_TOL:
            self._last_good[name] = (state.tilt_x, state.tilt_y,
                                     state.insertion)
            self._ik_q[name] = joints
        else:
            joints = list(self._ik_q[name])

        # Slew-limit rather than reject. Discarding an over-large solution
        # deadlocks the arm: on the first solve the distance from the home pose
        # always exceeds the threshold, so the solution is thrown away, the
        # seed never updates, and the arm sits at zero forever. Clamping the
        # step instead always makes progress toward the target while still
        # smoothing out branch flips over a few frames.
        joints = [r + float(np.clip(q - r, -cfg.MAX_JOINT_STEP,
                                    cfg.MAX_JOINT_STEP))
                  for r, q in zip(rest, joints)]

        a = cfg.JOINT_SMOOTH
        joints = [a * r + (1 - a) * q for r, q in zip(rest, joints)]
        self._last_q[name] = joints

        for j, q in enumerate(joints):
            p.setJointMotorControl2(arm, j, p.POSITION_CONTROL,
                                    targetPosition=q, force=300,
                                    maxVelocity=1.8,
                                    positionGain=0.25, velocityGain=1.0)
        self._draw_tool(name, tip, shaft_base, d)
        ee_pos, rcm_dev = self.flange_rcm_deviation(name)
        return {"ik_residual_m": float(np.linalg.norm(ee_pos - shaft_base)),
                # PHYSICAL: how far the arm's actual tool axis passes from the
                # trocar. This is the number that matters - a real deviation
                # tears the abdominal wall. It is non-zero while the arm lags
                # behind a moving command, and should settle to ~0.
                "pivot_error_m": rcm_dev,
                # ANALYTIC: exact by construction, kept only for reference.
                "pivot_error_cmd_m": pivot_error(tip, shaft_base,
                                                 self.pivots[name]),
                "tip": tip}

    def flange_rcm_deviation(self, name):
        """
        Distance from the trocar to the line along the arm's REAL flange
        z-axis. Measured from the simulated joint state, not the command.
        """
        ls = p.getLinkState(self.arms[name], cfg.EE_LINK,
                            computeForwardKinematics=True)
        pos = np.array(ls[4])
        R = np.array(p.getMatrixFromQuaternion(ls[5])).reshape(3, 3)
        axis = R[:, 2]
        dev = float(np.linalg.norm(np.cross(self.pivots[name] - pos, axis)))
        return pos, dev

    def _draw_tool(self, name, tip, shaft_base, d):
        """Place the held instrument body along the commanded shaft."""
        shaft, tipb = self._inst[name]
        mid = (np.asarray(tip) + np.asarray(shaft_base)) / 2.0
        p.resetBasePositionAndOrientation(shaft, mid.tolist(),
                                          quat_from_dir(d))
        p.resetBasePositionAndOrientation(tipb, list(tip), [0, 0, 0, 1])
        if name == "camera":
            self._draw_view_cone(np.asarray(tip), np.asarray(d))

    # --------------------------------------------------------------- views
    VIEWS = [
        # name,                 distance, yaw, pitch, target
        ("overview",            2.3,  -50, -30, [0.00, 0.00, 0.95]),
        ("behind surgeon",      1.9,  -90, -24, [0.10, 0.00, 1.00]),
        ("side",                2.2,    0, -18, [0.00, 0.00, 0.95]),
        ("top down",            2.4,  -90, -80, [0.00, 0.00, 0.90]),
        ("arms close-up",       1.0,  -35, -35, [0.03, 0.02, 1.02]),
    ]

    def set_view(self, i):
        """
        Camera presets. The old default looked straight along the y-axis, and
        with both carts at x=0 one arm hid exactly behind the other. The
        default is now a diagonal overview. Press 'v' in the console to cycle.
        """
        name, dist, yaw, pitch, tgt = self.VIEWS[i % len(self.VIEWS)]
        p.resetDebugVisualizerCamera(cameraDistance=dist, cameraYaw=yaw,
                                     cameraPitch=pitch,
                                     cameraTargetPosition=tgt)
        return name

    # ----------------------------------------------------------- collisions
    def arm_clearance(self):
        """
        Minimum distance between the two robot arms, from the physics engine's
        collision geometry. The arms work through trocars ~16 cm apart, so
        their wrists genuinely converge; this lets the controller back the
        instrument arm before they touch.
        """
        pts = p.getClosestPoints(self.arms["camera"], self.arms["instrument"],
                                 distance=cfg.ARM_CLEAR_QUERY)
        if not pts:
            return cfg.ARM_CLEAR_QUERY
        return float(min(pt[8] for pt in pts))

    def step(self, n=1):
        for _ in range(n):
            p.stepSimulation()

    def render(self, size=None):
        w, h = size or cfg.SIM_VIEW_SIZE
        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.05, 0.0, 0.95], distance=2.2,
            yaw=-18, pitch=-24, roll=0, upAxisIndex=2)
        proj = p.computeProjectionMatrixFOV(fov=55, aspect=w / h,
                                            nearVal=0.05, farVal=6.0)
        _, _, rgb, _, _ = p.getCameraImage(w, h, view, proj,
                                           renderer=p.ER_BULLET_HARDWARE_OPENGL)
        return np.reshape(np.array(rgb, dtype=np.uint8), (h, w, 4))[:, :, :3]

    def disconnect(self):
        try:
            p.disconnect(self._ik_cid)
        except Exception:                                   # noqa: BLE001
            pass
        p.disconnect(self.client)


REACH_FRACTION = {}       # filled by check_reachability(), read by selftest


def check_reachability(verbose=True):
    """
    Sweep the WHOLE workspace - every tilt in the cone and every insertion -
    and solve the actual IK for each pose on a ghost arm.

    The earlier check only compared the grip point's distance from the
    shoulder against a nominal 0.80 m reach. That passed while a whole
    quadrant of poses was unreachable: tilting the tool back toward its own
    cart puts the grip point almost inside the shoulder, which the arm cannot
    fold into. PyBullet's IK then failed silently and the arm left the trocar.
    """
    from rcm import RCMState
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
    ok = True
    try:
        for name, cart, pivot in [
                ("camera", cfg.CART_CAMERA, cfg.TROCAR_CAMERA),
                ("instrument", cfg.CART_INSTRUMENT, cfg.TROCAR_INSTRUMENT)]:
            g = p.loadURDF("kuka_iiwa/model.urdf",
                           basePosition=list(map(float, cart)),
                           useFixedBase=True, physicsClientId=cid)
            Rb = np.eye(3)
            _, sb, d = RCMState(pivot, insertion=0.13).pose()
            home, herr = dls_ik(cid, g, Rb, sb, d,
                                home_seed(cart, pivot), iters=400)
            if herr > cfg.IK_TOL:
                print(f"  !! {name} arm cannot reach its trocar at all "
                      f"({herr * 1000:.0f} mm off). Move CART_{name.upper()}.")
                ok = False
                continue
            tot = bad = 0
            tmax = np.tan(cfg.MAX_TILT)
            for tx in np.linspace(-cfg.MAX_TILT, cfg.MAX_TILT, 9):
                for ty in np.linspace(-cfg.MAX_TILT, cfg.MAX_TILT, 9):
                    if np.hypot(np.tan(tx), np.tan(ty)) > tmax + 1e-9:
                        continue
                    for ins in cfg.INSERTION_LIMITS:
                        _, sb, d = RCMState(pivot, tx, ty, ins).pose()
                        _, e = dls_ik(cid, g, Rb, sb, d, home, home,
                                      iters=200)
                        tot += 1
                        if e > cfg.IK_TOL:
                            bad += 1
            frac = 1.0 - bad / max(tot, 1)
            REACH_FRACTION[name] = frac
            if verbose:
                print(f"  {name:8s} reachable {100 * frac:5.1f}% of the "
                      f"{np.degrees(cfg.MAX_TILT):.0f} deg cone x insertion "
                      f"range ({bad}/{tot} poses refused at the cone edge)")
            if frac < cfg.MIN_REACHABLE_FRAC:
                ok = False
                print(f"  !! {name} arm reaches too little of its workspace. "
                      f"Move CART_{name.upper()} or lower MAX_TILT.")
    finally:
        p.disconnect(cid)
    return ok
