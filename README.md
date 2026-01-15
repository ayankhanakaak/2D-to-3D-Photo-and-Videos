# 2D → 3D Video Converter (MiDaS)

A simple Python tool that converts **normal 2D videos** into **immersive 3D** using **MiDaS depth estimation**.  
It supports **Red/Cyan Anaglyph** (red–cyan glasses) and **Side‑by‑Side** (SBS) stereo output.  
Optional **CUDA acceleration**, **mixed precision** (faster on NVIDIA GPUs), and **FFmpeg pre‑encoding** are built‑in for smooth, high‑quality results.

> **Author:** Ayan Khan  
> **Current script:** `2D to 3D Photo and Video V.29.12.2025-1.py`

---

## ✨ Highlights

- **Depth from single frames** with MiDaS models (DPT_Large / DPT_Hybrid / MiDaS_small)
- **Two 3D formats**: Red/Cyan Anaglyph and Side‑by‑Side (SBS)
- **CUDA + AMP** (automatic mixed precision) for speed on NVIDIA
- **FFmpeg pre‑encode** option for reliable decoding of tricky source files
- **Safe depth normalization** to avoid NaNs/Infs and crashes
- **Progress bar** shows live progress and a final saved path

---

## 📦 Requirements

- **Python** 3.9+ (3.10/3.11 recommended) | 100% Tested on: 3.13.5
- **Pip packages:**
  - `torch` (CPU or CUDA build)
  - `opencv-python`
  - `numpy`
- **FFmpeg** (optional, but recommended for the pre‑encode step)

> ⚠️ For NVIDIA GPUs, install a **CUDA-enabled** build of PyTorch that matches your driver/CUDA version.  
> See the official PyTorch site for the correct `pip` command for your system.

### Install dependencies

```bash
# Create/activate a virtual env (very optional but recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Core deps (necessary)
pip install --upgrade pip
pip install torch opencv-python numpy
# If you have an NVIDIA GPU, install the CUDA build of torch per pytorch.org
```

### FFmpeg (optional but useful)

- **Windows:** Install FFmpeg and add it to PATH.  
- **Linux/macOS:** Install via your package manager or official builds.  

The tool will still work without FFmpeg, but **pre‑encoding** can fix many decode issues and make processing smoother.

---

## 📥 Models (MiDaS)

The app loads MiDaS models directly via **`torch.hub`** from `intel-isl/MiDaS`. On first run, it will **download** the model weights to your Torch cache:

- `DPT_Large` → `dpt_large_384.pt` (highest quality, slowest)  
- `DPT_Hybrid` → `dpt_hybrid_384.pt` (quality/speed balance)  
- `MiDaS_small` → `midas_v21_small_256.pt` (fastest, lower quality)

You can also **provide your own model file** (for offline use). The tool will copy it into `~/.cache/torch/hub/checkpoints/` with the expected filename.

---

## ▶️ Usage (Interactive CLI)

Run the script (quote the filename since it has spaces):

```bash
python "2D to 3D Photo and Video V.29.12.2025-1.py"
```

You will be asked a few simple questions:

1) **Choose model**  
   `1. DPT_Large` · `2. DPT_Hybrid` · `3. MiDaS_small`

2) **Model source**  
   `1. Already Downloaded` (give path to your local .pt file)  
   `2. Download Automatically / Use Cached` (recommended)

3) **Use CUDA?**  
   `1. Yes` (if you have an NVIDIA GPU + CUDA PyTorch)  
   `2. No (CPU)`

4) **Enable mixed precision (AMP)?** *(only shown when CUDA is on)*  
   `1. Yes` (faster on modern GPUs) · `2. No`

5) **Input video path**  
   Paste or drag‑drop the full path to your source video file.

6) **Pre‑encode with FFmpeg?**  
   `1. Yes` → re‑encodes to H.264/AAC for reliable decoding  
   `2. No` → use original video as is

7) **Choose output format**  
   `1. Red/Cyan Anaglyph` (works with cheap red‑cyan glasses)  
   `2. SBS Stereo` (two views side‑by‑side; good for VR players/TVs)

8) **Max shift (pixels)**  
   Horizontal pixel shift based on depth map. Default is **15**.  
   Larger values = stronger 3D effect, but too large can look uncomfortable.

The app will print progress every 50 frames and save the output video next to your input:

- **Anaglyph:** `output_anaglyph_video-<n>.mp4`  
- **SBS:** `output_sbs_video-<n>.mp4`

---

## 🧠 How it works

- For each frame, MiDaS predicts a **depth map** (which parts are near/far).  
- We **normalize** that depth safely and create a **shift map**.  
- We **warp** the original frame left/right using that shift to make **two views** (left eye, right eye).  
- For **Anaglyph**, we mix channels (Left → Red, Right → Green+Blue).  
- For **SBS**, we place the two views **side-by-side**.

This is a **2D‑to‑3D approximation**. It won’t be perfect like true stereo capture, but with the right settings it looks surprisingly good.

---

## ⚙️ Options & Tips

- **Model choice:**  
  - *Best quality:* `DPT_Large`  
  - *Balanced:* `DPT_Hybrid`  
  - *Fastest:* `MiDaS_small`
- **Max shift:** Start with **15**. Try 8–24 depending on content and comfort.
- **FFmpeg pre‑encode:** If your video fails to open or stutters, enable it.
- **AMP (mixed precision):** Usually faster on RTX GPUs. If you see artifacts or errors, turn it **off**.
- **Frame rate & size:** Output uses the source **FPS** and **resolution**.  
  SBS doubles width (W → **2W**), height stays the same.

---

## 🧪 Example

```bash
python "2D to 3D Photo and Video V.29.12.2025-1.py"
# Choose: 2 (DPT_Hybrid)
# Source: 2 (Auto/Cache)
# CUDA: 1 (Yes)  → AMP: 1 (Yes)
# Input: C:\Videos\clip.mp4
# Pre-encode: 1 (Yes)
# Output: 2 (SBS Stereo)
# Max shift: 16
# → Saved: output_sbs_video-1.mp4
```

---

## 🛠 Troubleshooting

- **“Failed to open video.”**  
  Use the **FFmpeg pre‑encode** option. If it still fails, ensure FFmpeg is installed and that the path has **no special characters**.

- **Very slow / Out of memory (GPU).**  
  Use **MiDaS_small**, disable **AMP**, or switch to **CPU**. Close other GPU-heavy apps.

- **Weird color/ghosting in Anaglyph.**  
  This is normal if the 3D shift is too strong. **Lower Max shift** or try **SBS**.

- **Artifacts at edges.**  
  The script uses **border replication** to avoid holes. Minor stretching at frame borders is expected.

---

## 📚 Project Structure

- `2D to 3D Photo and Video V.29.12.2025-1.py` → main interactive script  
  - MiDaS load via `torch.hub`  
  - Safe depth normalization  
  - Optional **FFmpeg** re‑encode  
  - **Anaglyph** and **SBS** writers

You can rename the script if you like, just keep the code intact.

---

## 🤝 Contributing

Suggestions and complaints are welcome. Please describe your system, Python version, and exact steps to reproduce any bug.

---

## 📜 License

GPL-3.0

---

## 🙏 Credits

- **MiDaS** by Intel ISL (loaded via `torch.hub`)  
- Thanks to the open‑source community for PyTorch, OpenCV, NumPy, and FFmpeg.

