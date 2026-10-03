"""
Speed patches for pyrender 0.1.45 (applied by or_render.py).

Profiling the OR panel showed the time is NOT spent on the GPU but in
Python, issuing OpenGL calls one by one:

  * every uniform write first asks the driver for its location
    (glGetUniformLocation) - ~6000 extra driver calls per frame;
  * the camera matrices and ALL light uniforms are re-sent for EVERY mesh,
    in every pass (each light: 4-6 uniforms + a shadow matrix that is
    recomputed with a 4x4 inverse each time);
  * meshes are drawn in distance order, so the shader program flips back
    and forth constantly.

The patches keep pyrender's output identical but:

  1. cache uniform locations per shader program;
  2. draw opaque meshes grouped by shader program and bind the camera and
     lights ONCE per program instead of once per mesh (transparent meshes
     are still drawn back-to-front, as before);
  3. cache the (static) light / shadow-camera matrices, the shader-program
     lookup per primitive and the GPU's texture-unit limit.

Result: several times fewer OpenGL calls per frame, which is what makes the
panel and the zoomable 3D window smooth.
"""
import numpy as np

_APPLIED = False


def apply():
    global _APPLIED
    if _APPLIED:
        return
    import pyrender.renderer as R
    import pyrender.shader_program as SP
    from pyrender.constants import RenderFlags, ProgramFlags
    from OpenGL import GL

    # ------------------------------------------------------------ uniforms
    orig_set_uniform = SP.ShaderProgram.set_uniform
    ndarray = np.ndarray
    fmap = SP.ShaderProgram._FUNC_MAP

    def set_uniform(self, name, value, unsigned=False):
        try:
            cache = self.__dict__.setdefault("_loc_cache", {})
            key = (self._program_id, name)
            loc = cache.get(key)
            if loc is None:
                loc = GL.glGetUniformLocation(self._program_id, name)
                cache[key] = loc
            if loc == -1:
                return
            if isinstance(value, ndarray):
                if value.ndim == 1:
                    k = value.dtype.kind
                    if k == "u" or unsigned:
                        fmap[(value.shape[0], "u")](
                            loc, 1, value.astype(np.uint32, copy=False))
                    elif k == "i":
                        fmap[(value.shape[0], "i")](
                            loc, 1, value.astype(np.int32, copy=False))
                    else:
                        fmap[(value.shape[0], "f")](
                            loc, 1, value.astype(np.float32, copy=False))
                else:
                    fmap[(value.shape[0], value.shape[1])](
                        loc, 1, GL.GL_TRUE, value)
            elif isinstance(value, bool):
                (GL.glUniform1ui if unsigned else GL.glUniform1i)(
                    loc, int(value))
            elif isinstance(value, float):
                GL.glUniform1f(loc, value)
            elif isinstance(value, int):
                (GL.glUniform1ui if unsigned else GL.glUniform1i)(loc, value)
            else:
                orig_set_uniform(self, name, value, unsigned)
        except Exception:                                   # noqa: BLE001
            pass

    SP.ShaderProgram.set_uniform = set_uniform

    # -------------------------------------------------- cached lookups
    orig_get_prog = R.Renderer._get_primitive_program

    def get_prog(self, primitive, flags, program_flags):
        c = self.__dict__.setdefault("_prog_cache2", {})
        key = (id(primitive), int(flags), int(program_flags))
        prog = c.get(key)
        if prog is None or not prog._in_context():
            prog = orig_get_prog(self, primitive, flags, program_flags)
            c[key] = prog
        return prog

    R.Renderer._get_primitive_program = get_prog

    orig_light_mats = R.Renderer._get_light_cam_matrices

    def light_mats(self, scene, light_node, flags):
        c = self.__dict__.setdefault("_lightmat_cache", {})
        pose = scene.get_pose(light_node)
        hit = c.get(id(light_node))
        if hit is not None and hit[0] == scene.scale and \
                np.array_equal(hit[1], pose):
            return hit[2], hit[3]
        V, P = orig_light_mats(self, scene, light_node, flags)
        c[id(light_node)] = (scene.scale, pose.copy(), V, P)
        return V, P

    R.Renderer._get_light_cam_matrices = light_mats

    orig_max_lights = R.Renderer._compute_max_n_lights

    def max_lights(self, flags):
        c = self.__dict__.setdefault("_maxl_cache", {})
        k = int(flags)
        if k not in c:
            c[k] = orig_max_lights(self, flags)
        return c[k]

    R.Renderer._compute_max_n_lights = max_lights

    # ------------------------------------------------- draw ordering
    def _draw_list(self, scene, program_flags, flags, cam_loc):
        solid, trans = [], []
        for node in scene.mesh_nodes:
            mesh = node.mesh
            if not mesh.is_visible:
                continue
            pose = scene.get_pose(node)
            for prim in mesh.primitives:
                prog = self._get_primitive_program(prim, flags, program_flags)
                if mesh.is_transparent:
                    d = float(np.linalg.norm(pose[:3, 3] - cam_loc))
                    trans.append((-d, node, prim, prog, pose))
                else:
                    solid.append((id(prog), node, prim, prog, pose))
        solid.sort(key=lambda t: t[0])
        trans.sort(key=lambda t: t[0])
        return solid + trans

    # ------------------------------------------------- forward pass
    orig_forward = R.Renderer._forward_pass

    def forward_pass(self, scene, flags, seg_node_map=None):
        if flags & (RenderFlags.SEG | RenderFlags.FLAT):
            return orig_forward(self, scene, flags, seg_node_map)
        self._configure_forward_pass_viewport(flags)
        GL.glClearColor(*scene.bg_color)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glEnable(GL.GL_MULTISAMPLE)
        V, P = self._get_camera_matrices(scene)
        cam_loc = scene.get_pose(scene.main_camera_node)[:3, 3]
        lit = not (flags & RenderFlags.DEPTH_ONLY)
        program = None
        base = 0
        for _, node, prim, prog, pose in _draw_list(
                self, scene, ProgramFlags.USE_MATERIAL, flags, cam_loc):
            if prog is not program:
                program = prog
                program._bind()
                program.set_uniform("V", V)
                program.set_uniform("P", P)
                program.set_uniform("cam_pos", cam_loc)
                self._reset_active_textures()
                if lit:
                    self._bind_lighting(scene, program, node, flags)
                base = self._texture_alloc_idx
            self._texture_alloc_idx = base      # keep the shadow-map units
            self._bind_and_draw_primitive(primitive=prim, pose=pose,
                                          program=program, flags=flags)
        self._reset_active_textures()
        if program is not None:
            program._unbind()
        GL.glFlush()
        if flags & RenderFlags.OFFSCREEN:
            return self._read_main_framebuffer(scene, flags)
        return

    R.Renderer._forward_pass = forward_pass

    # ------------------------------------------------- shadow pass
    def shadow_pass(self, scene, light_node, flags):
        light = light_node.light
        self._configure_shadow_mapping_viewport(light, flags)
        V, P = self._get_light_cam_matrices(scene, light_node, flags)
        cam_loc = scene.get_pose(scene.main_camera_node)[:3, 3]
        program = None
        for _, node, prim, prog, pose in _draw_list(
                self, scene, ProgramFlags.NONE, flags, cam_loc):
            if prog is not program:
                program = prog
                program._bind()
                program.set_uniform("V", V)
                program.set_uniform("P", P)
                program.set_uniform("cam_pos", cam_loc)
            self._bind_and_draw_primitive(primitive=prim, pose=pose,
                                          program=program,
                                          flags=RenderFlags.DEPTH_ONLY)
            self._reset_active_textures()
        if program is not None:
            program._unbind()
        GL.glFlush()

    R.Renderer._shadow_mapping_pass = shadow_pass
    _APPLIED = True
