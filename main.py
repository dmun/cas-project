import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import pyray as rl

GRAVITY = 9.8
num_particles = 10000

key = random.PRNGKey(0)
positions = random.uniform(key, (num_particles, 2), minval=0, maxval=600)
velocities = random.uniform(key, (num_particles, 2), minval=-2, maxval=2)
forces = random.uniform(key, (num_particles, 2), minval=-2, maxval=2)
masses = random.uniform(key, (num_particles, 2), minval=0, maxval=10)


@jax.jit
def update_positions(
    positions: jax.Array,
    masses: jax.Array,
    forces: jax.Array,
    velocities: jax.Array,
    dt: float,
    mouse_pos: jax.Array,
):
    direction = mouse_pos - positions
    distance = jnp.linalg.norm(direction, axis=1, keepdims=True)
    direction = direction / (distance + 1e-8)  # Normalize and avoid division by zero
    forces = direction * GRAVITY * masses
    velocities += forces / masses * dt
    positions += velocities * dt

    # Collision detection with window boundaries
    positions = jnp.where(positions <= 0, 0, positions)
    positions = jnp.where(
        positions >= jnp.array([WIDTH, HEIGHT]),
        jnp.array([WIDTH, HEIGHT]),
        positions,
    )
    velocities = jnp.where(positions <= 0, 0, velocities)
    velocities = jnp.where(
        positions >= jnp.array([WIDTH, HEIGHT]),
        0,
        velocities,
    )
    colors = jnp.clip(jnp.abs(velocities.sum(axis=1)) * 2, 0, 255)

    return positions, velocities, colors


WIDTH = 800
HEIGHT = 600
rl.init_window(WIDTH, HEIGHT, "Simulation")
rl.set_target_fps(120)

mouse_pos = rl.get_mouse_position()

shader = rl.load_shader("shader.vert.glsl", "shader.frag.glsl")
positions_loc = rl.get_shader_location(shader, "positions")

camera = rl.Camera3D()
camera.position = rl.Vector3(0.0, 5.0, 1.0)
camera.target = rl.Vector3(0.0, 0.0, 0.0)
camera.up = rl.Vector3(0.0, 1.0, 0.0)
camera.fovy = 90.0
camera.projection = rl.CameraProjection.CAMERA_PERSPECTIVE

vertices = np.zeros((num_particles, 2), dtype=np.float32).flatten()
vertex_buffer = rl.rl_load_vertex_buffer(
    rl.ffi.new("float[1000]"),
    num_particles,
    True,
)
# rl.rl_load_vertex_buffer()

while not rl.window_should_close():
    dt = rl.get_frame_time() * 2
    mouse = rl.get_mouse_position()
    mouse_pos = jnp.array([mouse.x, mouse.y])
    positions, velocities, colors = update_positions(
        positions,
        masses,
        forces,
        velocities,
        dt,
        mouse_pos,
    )

    # positions_flat = np.array(positions).flatten().astype(np.float32).tolist()
    # model = rl.upload_mesh(positions_flat, False)
    # rl.set_shader_value_v(
    #     shader,
    #     positions_loc,
    #     rl.ffi.new("float[]", positions_flat),
    #     rl.ShaderUniformDataType.SHADER_UNIFORM_VEC2,
    #     num_particles,
    # )

    rl.begin_drawing()
    # rl.begin_mode_3d(camera)
    rl.clear_background(rl.BLACK)
    for pos, color in zip(np.array(positions), np.array(colors)):
        rl.draw_rectangle(
            pos[0],
            pos[1],
            1,
            1,
            (color, 155, 255 - color, 128),
        )
        # rl.draw_point_3d((pos[0], pos[1], 0), rl.RED)

    # rl.rl_enable_point_mode()
    # rl.begin_shader_mode(shader)
    # rl.rl_draw_vertex_array_elements(
    #     0, num_particles, vertex_buffer
    # )
    # rl.end_shader_mode()

    # rl.end_mode_3d()
    rl.draw_fps(0, 0)
    rl.end_drawing()

rl.close_window()
