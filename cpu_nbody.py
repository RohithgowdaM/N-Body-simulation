import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import time  # for timing

# ------------------------------
# Simulation Parameters
# ------------------------------
G = 6.67430e-11          # gravitational constant [m^3/(kg·s^2)]
dt = 1e6                 # time step in seconds
num_frames = 300         # total frames (300 frames @ 15 fps → 20-second video)
num_small = 25           # number of small bodies
num_bodies = num_small + 1  # total bodies (1 heavy + 25 small)
axis_limit = 1.2e11 * 10   # visualization limits (scaled up by 10 → 1.2e12 meters)
canvas_size = 600        # canvas size in pixels
fps = 15                 # frame rate

# ------------------------------
# Initial Conditions (Random every run)
# ------------------------------
# Heavy object (one massive body)
heavy_mass = 1e30
heavy_position = np.random.uniform(-1e11, 1e11, 2) * 10  
heavy_velocity = np.array([0.0, 0.0])  # stationary

# 25 small bodies with masses between 1e24 and 1e26 kg,
# random positions (scaled up by 10) and small random velocities.
small_masses = np.random.uniform(1e24, 1e26, num_small)
small_positions = np.random.uniform(-1e11, 1e11, (num_small, 2)) * 10  
small_velocities = np.random.uniform(-1e3, 1e3, (num_small, 2))

# Combine heavy and small bodies.
masses = np.insert(small_masses, 0, heavy_mass)
positions = np.vstack((heavy_position, small_positions))
velocities = np.vstack((heavy_velocity, small_velocities))

# ------------------------------
# Helper Function: Marker Size Based on Mass
# ------------------------------
def get_marker_size(mass):
    # Using logarithm base 10 for scaling.
    return max(5, int((np.log10(mass) - 23) * 10))

# ------------------------------
# Assign Vibrant Colors
# ------------------------------
colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(num_bodies)]

# ------------------------------
# Video Writer Setup
# ------------------------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter("cpu_nbody.mp4", fourcc, fps, (canvas_size, canvas_size))

# ------------------------------
# Simulation Functions (CPU)
# ------------------------------
def compute_forces_cpu():
    """Compute gravitational forces on each body using nested loops."""
    forces = np.zeros_like(positions)
    for i in range(num_bodies):
        for j in range(num_bodies):
            if i == j:
                continue
            diff = positions[j] - positions[i]
            r = np.linalg.norm(diff) + 1e-9  # avoid division by zero
            force = G * masses[i] * masses[j] / (r ** 2)
            forces[i] += force * diff / r
    return forces

def update_cpu():
    """Update velocities and positions using Euler integration."""
    global positions, velocities
    forces = compute_forces_cpu()
    velocities += (forces / masses[:, None]) * dt
    positions += velocities * dt

def save_frame(frame_num):
    """Plot the current positions and save as an image frame."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.set_facecolor("black")
    for i in range(num_bodies):
        marker_size = get_marker_size(masses[i])
        ax.scatter(positions[i, 0], positions[i, 1], s=marker_size ** 2,
                   color=np.array(colors[i]) / 255.0, label=f"Body {i}")
    ax.axis("off")
    filename = "temp_cpu_frame.png"
    plt.savefig(filename, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    frame = cv2.imread(filename)
    frame = cv2.resize(frame, (canvas_size, canvas_size))
    video.write(frame)
    return frame

# ------------------------------
# Main Simulation Loop (CPU)
# ------------------------------
start_time = time.time()

first_frame_cpu = None
last_frame_cpu = None

for f in range(num_frames):
    if f == 0:
        frame_img = save_frame(f)
        first_frame_cpu = frame_img.copy()
    elif f == num_frames - 1:
        frame_img = save_frame(f)
        last_frame_cpu = frame_img.copy()
    else:
        save_frame(f)
    update_cpu()

video.release()
os.remove("temp_cpu_frame.png")

end_time = time.time()
elapsed = end_time - start_time
print(f"CPU Simulation completed in {elapsed:.2f} seconds.")

# Display the first and last frames.
cv2.imshow("CPU First Frame", first_frame_cpu)
cv2.imshow("CPU Last Frame", last_frame_cpu)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("CPU simulation video saved as cpu_nbody.mp4")
