# 🚀 N-Body Simulation with OpenCL

This repository contains an **OpenCL-accelerated N-body simulation**, which simulates the gravitational interactions between celestial bodies in space. The simulation features **25 small bodies** and **1 massive central body**, with **randomized initial positions, velocities, and colors**.

## ✨ Features
- Uses **OpenCL** to run computations on the **GPU** for faster performance.
- Simulates **gravity-based motion** using **Newton’s laws**.
- **Parallel execution** ensures **speedup over CPU-based computation**.
- Outputs a **visual simulation** stored as an `.mp4` file.
- Displays the **first and last frames** for quick visualization.

---

## 📜 How It Works
1. **Bodies are initialized** with random positions, velocities, and masses.
2. **The OpenCL kernel computes gravitational forces** in parallel.
3. **Positions and velocities are updated** using Euler’s method.
4. **Frames are generated using OpenCV** and saved as a video.
5. The simulation **runs for 10 seconds**, generating a `.mp4` output.

---

## ⚡ Performance Comparison

| Feature | CPU Version | GPU Version (OpenCL) |
|---------|------------|----------------------|
| Force Computation | **Nested loops** (slow) | **Parallelized on GPU** (fast) |
| Execution Time | **Several seconds/minutes** | **10×–100× faster** |
| Visualization | **Matplotlib** | **OpenCV (efficient)** |
| Scalability | **Limited by CPU cores** | **Can handle thousands of bodies** |

---

## 🛠️ Requirements
- **Python 3.x**
- `numpy`
- `pyopencl`
- `opencv-python`
- `os` (Built-in)
- `time` (Built-in)

### 📌 Install Dependencies
Run the following command to install the required packages:
```bash
pip install numpy pyopencl opencv-python
git clone https://github.com/your-username/nbody-opencl.git
cd nbody-opencl
python nbody_opencl.py
```
If you want to run the simlation without opencl, use the command
```bash
python cpu_nbody.py
```