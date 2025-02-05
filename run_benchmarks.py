import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import time
import pyopencl as cl

# ================================
# Common Simulation Parameters
# ================================
G = 6.67430e-11          # gravitational constant [m^3/(kg·s^2)]
dt = 1e6                 # time step in seconds
num_frames = 300         # total frames (300 frames @ 15 fps → 20-second video)
num_small = 25           # number of small bodies
num_bodies = num_small + 1  # total bodies (1 heavy + 25 small)
axis_limit = 1.2e11 * 10   # visualization limits (scaled up by 10 → 1.2e12 meters)
canvas_size = 600        # output canvas size in pixels
fps = 15                 # frame rate for the video

# ================================
# Helper Functions (Common)
# ================================
def get_marker_size(mass):
    """Compute marker size for the CPU simulation based on mass (for matplotlib scatter)."""
    return max(5, int((np.log10(mass) - 23) * 10))

def get_radius(mass):
    """Compute circle radius for the OpenCL simulation based on mass (for OpenCV drawing)."""
    return max(3, int((np.log10(mass) - 23) * 2))

def assign_colors(n):
    """Generate a list of n random vibrant RGB color tuples."""
    return [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(n)]

# ================================
# CPU-Based Simulation Function
# ================================
def run_cpu_simulation():
    heavy_mass = 1e30
    heavy_position = np.random.uniform(-1e11, 1e11, 2) * 10  
    heavy_velocity = np.array([0.0, 0.0])
    
    small_masses = np.random.uniform(1e24, 1e26, num_small)
    small_positions = np.random.uniform(-1e11, 1e11, (num_small, 2)) * 10  
    small_velocities = np.random.uniform(-1e3, 1e3, (num_small, 2))
    
    masses = np.insert(small_masses, 0, heavy_mass)
    positions = np.vstack((heavy_position, small_positions))
    velocities = np.vstack((heavy_velocity, small_velocities))
    
    colors = assign_colors(num_bodies)
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter("cpu_nbody.mp4", fourcc, fps, (canvas_size, canvas_size))
    
    def compute_forces_cpu(pos):
        forces = np.zeros_like(pos)
        for i in range(num_bodies):
            for j in range(num_bodies):
                if i == j:
                    continue
                diff = pos[j] - pos[i]
                r = np.linalg.norm(diff) + 1e-9
                force = G * masses[i] * masses[j] / (r ** 2)
                forces[i] += force * diff / r
        return forces
    
    # Main simulation loop (CPU)
    start_time = time.time()
    for f in range(num_frames):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-axis_limit, axis_limit)
        ax.set_ylim(-axis_limit, axis_limit)
        ax.set_facecolor("black")  
        for i in range(num_bodies):
            marker_size = get_marker_size(masses[i])
            ax.scatter(positions[i, 0], positions[i, 1], s=marker_size**2,
                       color=np.array(colors[i]) / 255.0)
        ax.axis("off")
        temp_filename = "temp_cpu_frame.png"
        plt.savefig(temp_filename, dpi=100, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        frame = cv2.imread(temp_filename)
        frame = cv2.resize(frame, (canvas_size, canvas_size))
        video.write(frame)
        os.remove(temp_filename)
        
        forces = compute_forces_cpu(positions)
        velocities += (forces / masses[:, None]) * dt
        positions += velocities * dt
    video.release()
    elapsed = time.time() - start_time
    return elapsed

# ================================
# OpenCL-Based Simulation Function
# ================================
def run_opencl_simulation():
    heavy_mass = np.float32(1e30)
    heavy_position = (np.random.uniform(-1e11, 1e11, 2) * 10).astype(np.float32)
    heavy_velocity = np.array([0.0, 0.0], dtype=np.float32)
    
    small_masses = np.random.uniform(1e24, 1e26, num_small).astype(np.float32)
    small_positions = (np.random.uniform(-1e11, 1e11, (num_small, 2)) * 10).astype(np.float32)
    small_velocities = np.random.uniform(-1e3, 1e3, (num_small, 2)).astype(np.float32)
    
    masses = np.insert(small_masses, 0, heavy_mass)
    positions = np.vstack((heavy_position, small_positions))
    velocities = np.vstack((heavy_velocity, small_velocities))
    
    colors = np.random.randint(0, 255, (num_bodies, 3))
    
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)
    
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
    program = cl.Program(context, kernel_code).build()
    mf = cl.mem_flags
    pos_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=positions)
    vel_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=velocities)
    mass_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=masses)
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter("opencl_nbody.mp4", fourcc, fps, (canvas_size, canvas_size))
    
    def generate_frame(pos_arr):
        img = np.full((canvas_size, canvas_size, 3), (0, 0, 0), dtype=np.uint8)  # Black background
        for i in range(num_bodies):
            x = int((pos_arr[i, 0] / axis_limit + 1) * canvas_size / 2)
            y = int((pos_arr[i, 1] / axis_limit + 1) * canvas_size / 2)
            radius = get_radius(masses[i])
            cv2.circle(img, (x, y), radius,
                       (int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2])), -1)
        return img
    
    start_time = time.time()
    for f in range(num_frames):
        program.nbody_simulation(queue, (num_bodies,), None, pos_buf, vel_buf, mass_buf,
                                 np.int32(num_bodies), np.float32(G), np.float32(dt))
        pos_out = np.empty_like(positions)
        cl.enqueue_copy(queue, pos_out, pos_buf).wait()
        frame = generate_frame(pos_out)
        video_writer.write(frame)
    video_writer.release()
    elapsed = time.time() - start_time
    return elapsed

# ================================
# Main Routine: Run Simulations and Generate Report
# ================================
print("Running CPU simulation...")
cpu_time = run_cpu_simulation()
print("Running OpenCL simulation...")
opencl_time = run_opencl_simulation()

report = f"""
--- Performance Report ---
Total Frames: {num_frames}
Frame Rate: {fps} fps
Simulation Time Step (dt): {dt} seconds

CPU Simulation:
    Total Execution Time: {cpu_time:.2f} seconds
    Average Time per Frame: {cpu_time / num_frames:.4f} seconds

OpenCL Simulation:
    Total Execution Time: {opencl_time:.2f} seconds
    Average Time per Frame: {opencl_time / num_frames:.4f} seconds

Speedup Factor (CPU/OpenCL): {cpu_time / opencl_time:.2f}x
"""
print(report)
