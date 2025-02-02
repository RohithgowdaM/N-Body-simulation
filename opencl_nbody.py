import pyopencl as cl
import numpy as np
import cv2
import os
import time  # for timing

# ------------------------------
# Simulation Parameters
# ------------------------------
G = 6.67430e-11          # gravitational constant
dt = 1e6                 # time step in seconds
num_frames = 300         # total frames (300 frames @ 15 fps → 20-second video)
num_small = 25           # number of small bodies
num_bodies = num_small + 1  # total bodies (1 heavy + 25 small)
axis_limit = 1.2e11 * 10   # simulation axis limits (scaled up by 10 → 1.2e12 meters)
canvas_size = 600        # image size in pixels
fps = 15                 # frame rate

# ------------------------------
# Initial Conditions (Random every run)
# ------------------------------
# Heavy object (one massive body)
heavy_mass = np.float32(1e30)
heavy_position = (np.random.uniform(-1e11, 1e11, 2) * 10).astype(np.float32)
heavy_velocity = np.array([0.0, 0.0], dtype=np.float32)

# 25 small bodies with random masses between 1e24 and 1e26 kg.
small_masses = np.random.uniform(1e24, 1e26, num_small).astype(np.float32)
small_positions = (np.random.uniform(-1e11, 1e11, (num_small, 2)) * 10).astype(np.float32)
small_velocities = np.random.uniform(-1e3, 1e3, (num_small, 2)).astype(np.float32)

# Combine heavy and small bodies.
masses = np.insert(small_masses, 0, heavy_mass)
positions = np.vstack((heavy_position, small_positions))
velocities = np.vstack((heavy_velocity, small_velocities))

# ------------------------------
# Helper Function: Compute Drawing Radius
# ------------------------------
def get_radius(mass):
    # Use logarithm to scale the drawing radius.
    return max(3, int((np.log10(mass) - 23) * 2))

# ------------------------------
# Assign Vibrant Colors
# ------------------------------
colors = np.random.randint(0, 255, (num_bodies, 3))

# ------------------------------
# OpenCL Kernel Code
# ------------------------------
kernel_code = """
__kernel void nbody_simulation(__global float2 *pos,
                               __global float2 *vel,
                               __global float *mass,
                               int n, float G, float dt) {
    int i = get_global_id(0);
    if(i >= n) return;
    float2 acc = (float2)(0.0f, 0.0f);
    for (int j = 0; j < n; j++) {
        if(i == j) continue;
        float2 diff = pos[j] - pos[i];
        float r2 = dot(diff, diff) + 1e4f; // avoid singularity
        float inv_r = native_rsqrt(r2);
        float inv_r3 = inv_r * inv_r * inv_r;
        acc += G * mass[j] * diff * inv_r3;
    }
    vel[i] += acc * dt;
    pos[i] += vel[i] * dt;
}
"""

# ------------------------------
# OpenCL Setup
# ------------------------------
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)
program = cl.Program(context, kernel_code).build()

mf = cl.mem_flags
pos_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=positions)
vel_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=velocities)
mass_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=masses)

# ------------------------------
# Video Writer Setup
# ------------------------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter("opencl_nbody.mp4", fourcc, fps, (canvas_size, canvas_size))

def generate_frame(pos):
    """Generate an image frame from the current positions."""
    img = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    for i in range(num_bodies):
        x = int((pos[i, 0] / axis_limit + 1) * canvas_size / 2)
        y = int((pos[i, 1] / axis_limit + 1) * canvas_size / 2)
        radius = get_radius(masses[i])
        cv2.circle(img, (x, y), radius, (int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2])), -1)
    return img

# ------------------------------
# Main Simulation Loop (OpenCL)
# ------------------------------
start_time = time.time()

first_frame_opencl = None
last_frame_opencl = None

for f in range(num_frames):
    program.nbody_simulation(queue, (num_bodies,), None, pos_buf, vel_buf, mass_buf,
                             np.int32(num_bodies), np.float32(G), np.float32(dt))
    pos_out = np.empty_like(positions)
    cl.enqueue_copy(queue, pos_out, pos_buf).wait()
    frame_img = generate_frame(pos_out)
    if f == 0:
        first_frame_opencl = frame_img.copy()
        cv2.imwrite("opencl_first_frame.png", frame_img)
    elif f == num_frames - 1:
        last_frame_opencl = frame_img.copy()
        cv2.imwrite("opencl_last_frame.png", frame_img)
    video_writer.write(frame_img)

video_writer.release()
end_time = time.time()

elapsed = end_time - start_time
print(f"OpenCL Simulation completed in {elapsed:.2f} seconds.")

# Display the first and last frames.
cv2.imshow("OpenCL First Frame", first_frame_opencl)
cv2.imshow("OpenCL Last Frame", last_frame_opencl)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("OpenCL simulation video saved as opencl_nbody.mp4")
