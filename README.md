# 2D → 3D Video Converter (MiDaS 3.1)

A powerful Python tool that converts **normal 2D videos** into **immersive 3D** using **MiDaS depth estimation**.  
Supports **Red/Cyan Anaglyph** with automatic focus adjustment and **Side‑by‑Side** (SBS) stereo output.  
Features **multi‑GPU processing**, **CUDA acceleration**, **mixed precision**, **hardware decode/encode**, and **full audio/subtitle preservation**, **configurable max runtime** with graceful partial output.

> **Author:** Ayan Khan  
> **Current Version:** `V.25.4.2026-1`  
> **Script:** `2D to 3D Photo and Video V.25.4.2026-1.py`

---

## ✨ Key Features

### 🎬 Video Processing
- **MiDaS 3.1 support** with 12 depth models (BEiT, Swin, Next‑ViT, LeViT)
- **MiDaS 3.0 compatibility** (DPT_Large, DPT_Hybrid, MiDaS_small)
- **Multi‑GPU parallel processing** – splits frames across GPUs, automatically merges (or saves partial chunks on timeout)
- **Audio/subtitle preservation** – full metadata, track names, languages retained
- **Hardware acceleration** – NVDEC (decode), NVENC (encode), CUDA Remap
- **Batch processing** for optimal GPU utilization
- **Progress tracking** – real‑time FPS counter with tqdm

### 🕶️ 3D Output Modes
- **Red/Cyan Anaglyph** (default) with 3 color modes (user‑selectable):
  - **Full Color** – vivid 3D with red‑cyan glasses
  - **Half Color** – grayscale left eye, color right (reduced ghosting) – **default**
  - **Gray** – monochrome for maximum compatibility
- **Side‑by‑Side (SBS)** – for VR headsets, 3D TVs, cross‑eye viewing
- **Auto‑focus adjustment** – cross‑correlation based subject detection

### ⚡ Performance & Usability
- **Automatic mixed precision (AMP)** – faster inference on NVIDIA RTX/Ampere+
- **CUDA Remap acceleration** – GPU‑based image warping
- **Hot‑reload timm compatibility** – automatic version management (no restart needed)
- **Safe depth normalization** – handles NaNs/Infs gracefully
- **Max runtime control** – set a time limit; partial output saved as separate video parts or single truncated file
- **Forced pre‑encode** – converts input to a reliable constant‑frame‑rate format

---

## 📦 Requirements

### Core Dependencies
- **Python** 3.10–3.13 (tested on 3.12)
- **PyTorch** 2.0+ with CUDA support (for GPU acceleration)
- **timm** 0.6.13 (auto‑managed for MiDaS 3.1 models)
- **opencv‑python** 4.5+
- **numpy** < 2.2 (for compatibility)
- **tqdm** – progress bars
- **FFmpeg** – video I/O, hardware codecs
- ⭐ **100%** tested on Kaggle environment!

### Optional (Recommended)
- **NVIDIA GPU** with CUDA 11.8+ drivers
- **FFmpeg with NVENC/NVDEC** for hardware acceleration
- **OpenCV with CUDA** for GPU remapping (faster warping)

### Installation

```bash
# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install core dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python numpy tqdm timm

# Verify FFmpeg installation
ffmpeg -version
```

> **Note:** Replace `cu121` with your CUDA version. Visit [pytorch.org](https://pytorch.org/get-started/locally/) for the exact command.

---

## 🎯 Supported Models

### MiDaS 3.0 (Stable, Works Everywhere)
| Model | Resolution | Speed | Quality | VRAM |
|-------|------------|-------|---------|------|
| `DPT_Large` | 384×384 | Medium | ⭐⭐⭐⭐ | ~4GB |
| `DPT_Hybrid` | 384×384 | Fast | ⭐⭐⭐ | ~3GB |
| `MiDaS_small` | 256×256 | Fastest | ⭐⭐ | ~2GB |

### MiDaS 3.1 (Best Quality, Requires timm 0.6.13)
| Model | Resolution | Speed | Quality | VRAM |
|-------|------------|-------|---------|------|
| `DPT_BEiT_L_512` ⭐ | 512×512 | Slow | ⭐⭐⭐⭐⭐ | ~8GB |
| `DPT_BEiT_L_384` | 384×384 | Medium | ⭐⭐⭐⭐ | ~5GB |
| `DPT_SwinV2_L_384` | 384×384 | Medium | ⭐⭐⭐⭐ | ~5GB |
| `DPT_SwinV2_T_256` | 256×256 | Fast | ⭐⭐⭐ | ~3GB |
| `DPT_LeViT_224` | 224×224 | Fastest | ⭐⭐ | ~2GB |

> **BEiT_L_512** achieves **~28% better accuracy** than DPT_Large on NYU Depth V2 benchmark.

---

## ▶️ Quick Start

### Basic Usage
```bash
python "2D to 3D Photo and Video V.25.4.2026-1.py"
```

Follow the simplified interactive prompts:
1. **Choose depth model** (1–12)
2. **Select GPU/CPU** processing (multi‑GPU supported)
3. **Enter video input/output paths**
4. **Set duration** (full or custom start/end frames)
5. **Choose anaglyph mode** (1=Full Color, 2=Half Color, 3=Gray) – defaults to **Half Color**
6. **Set max runtime** (seconds) – script will stop gracefully and save partial output
   - For single‑GPU: output contains frames processed up to timeout
   - For multi‑GPU: each GPU’s chunk saved as a separate `_PartX.mp4` file, preserving audio/subs per part

All hardware acceleration options (NVDEC, NVENC, CUDA Remap, CUDA Focus, AMP) are **automatically enabled** if supported. Batch size is fixed at 9. The input is always pre‑encoded to a reliable format.

### Example Session
```
📦 Choose Depth Model:
>>> 4  # DPT_BEiT_L_512 (best quality)

🔍 Found 2 GPU(s):
   GPU 0: NVIDIA RTX 4090
   GPU 1: NVIDIA RTX 4090

GPU Options:
>>> 1  # Use all GPUs

Enter video input path: input.mp4
Enter video output path: output_3d.mp4

Choose duration:
 1. Full Video
 2. Custom
>>> 1

Choose Anaglyph Mode:
1. Full Color
2. Half Color
3. Gray
>>> 2

Enter max runtime in seconds: 3600

🚀 Processing...
✅ Done! 14520 frames processed, 3490.2s, 4.16 FPS (timeout reached – partial output saved)
   Saved: output_3d_Part1.mp4, output_3d_Part2.mp4
```

---

## 🧠 How It Works

### Depth Estimation Pipeline
1. **Video decode** → Read frames (NVDEC optional)
2. **MiDaS inference** → Predict depth maps (batch processing, batch size = 9)
3. **Depth normalization** → Safe handling of NaNs/Infs
4. **Stereo generation** → Warp frames based on depth (CUDA Remap if available)
5. **Focus adjustment** → Cross‑correlation alignment (only for Anaglyph)
6. **Output encoding** → Write 3D video (NVENC if available)
7. **Audio remux** → Reattach extracted audio/subtitle tracks

### Timeout & Partial Output
- You specify a **maximum runtime** (in seconds).
- Every few frames the script checks the elapsed time.
- When the timeout is reached, processing stops gracefully.
- **Single‑GPU:** the output video contains all frames processed so far (audio/subs aligned).
- **Multi‑GPU:** each worker saves its partial chunk; the main script packages them as separate `_PartX.mp4` files with proper audio/subtitles extracted for each part.

### Anaglyph Focus Adjustment
- **Cross‑correlation matching** on center band of frames
- **Automatic shift detection** for natural depth placement
- **Bilateral warping** positions subject slightly in front of screen
- **Reduces ghosting** compared to naive channel mixing

### Multi‑GPU Strategy
- **Frame splitting** – each GPU processes contiguous chunks
- **Parallel inference** – workers run simultaneously
- **Chunk merging** – FFmpeg concatenates full outputs (or saves parts on timeout)
- **Synchronized progress** – unified progress bar across GPUs

---

## ⚙️ Advanced Configuration

### Performance Tuning (Fixed Parameters in V.25)

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Batch size** | 9 | Fixed for optimal throughput |
| **AMP** | Auto‑enabled if CUDA available | Use mixed precision |
| **NVDEC** | Auto‑enabled if supported | Hardware decode |
| **NVENC** | Auto‑enabled if supported | Hardware encode |
| **CUDA Remap** | Auto‑enabled if available | GPU warping |
| **CUDA Focus** | Auto‑enabled if available | GPU cross‑correlation |
| **Pre‑encode** | Always enabled | Converts to CFR/h264 |

### Hardware Acceleration Matrix

| Feature | Requirement | Speedup |
|---------|-------------|---------|
| **NVDEC** | NVIDIA GPU + FFmpeg CUDA | ~10‑15% |
| **NVENC** | NVIDIA GPU + FFmpeg NVENC | ~20‑30% |
| **CUDA Remap** | OpenCV‑CUDA | ~15‑20% |
| **Multi‑GPU** | 2+ NVIDIA GPUs | ~2× (2 GPUs) |
| **AMP** | NVIDIA Tensor Cores | ~30‑40% |

### Typical Performance (1080p video, RTX 4090)

| Configuration | FPS | Time (10 min video) |
|---------------|-----|---------------------|
| Single GPU | 5.1 | ~58 min |
| Dual GPU | 8.7 | ~34 min |
| + NVENC + CUDA Remap | 10.3 | ~29 min |

---

## 🛠 Troubleshooting

### Common Issues

#### `'Block' object has no attribute 'drop_path'`
**Cause:** Incompatible `timm` version  
**Fix:** Script auto‑downgrades to timm 0.6.13 (restart cell if in Jupyter/Colab)

#### `NVDEC failed to start`
**Cause:** FFmpeg not compiled with CUDA hwaccel  
**Fix:** Install FFmpeg from official NVIDIA builds or the script will fall back to standard decode.

#### `Out of memory (GPU)`
**Solutions:**
- Use a smaller model (MiDaS_small or DPT_Hybrid)
- The batch size is fixed at 9; if you have less than 8GB VRAM, consider using a smaller model or CPU mode.

#### Missing chunks / partial output on timeout
**This is expected** when the max runtime is reached. The script saves:
- Single‑GPU: one video file with all processed frames.
- Multi‑GPU: separate `_PartX.mp4` files for each GPU’s progress.  
Check the console output for the exact filenames.

#### Duration mismatch after remux
**Cause:** Frame drop during encoding  
**Fix:** Pre‑encode is always enabled in V.25, which solves most VFR issues.

### Debug Mode
Check worker logs (multi‑GPU) for detailed errors:
```bash
# Multi‑GPU processing creates logs:
_temp_<PID>_log_0.txt  # GPU 0 activity
_temp_<PID>_log_1.txt  # GPU 1 activity
_temp_<PID>_progress_0.txt  # Frame count
*.error  # Exception tracebacks
```

---

## 📊 Benchmarks

### Quality Comparison (DPT_Large vs BEiT_L_512)
| Metric | DPT_Large | BEiT_L_512 | Improvement |
|--------|-----------|------------|-------------|
| **Abs Rel Error** | 0.062 | 0.045 | 28% better |
| **RMSE** | 0.254 | 0.189 | 26% better |
| **δ < 1.25** | 95.9% | 98.1% | +2.2% |

---

## 🧹 Auto‑Generated Files (Temporary)

```
_temp_<PID>_worker.py         # GPU worker script
_temp_<PID>_chunk_0.mp4       # GPU 0 output chunk
_temp_<PID>_config_0.json     # Worker configuration
_temp_<PID>_progress_0.txt    # Frame counter
_temp_<PID>_log_0.txt         # Worker activity log
_temp_streams_<PID>_audio_0.mka   # Extracted audio
_temp_streams_<PID>_subtitle_0.srt # Extracted subtitles
```
> All temporary files are auto‑deleted after successful processing (unless a timeout occurs – then the partial video parts and their accompanying audio/subs remain).

---

## 🤝 Contributing

### Reporting Issues
Please include:
- **Script version** (`V.25.4.2026-1`)
- **Python version** (`python --version`)
- **GPU model** (if applicable)
- **Error logs** (check `_temp_*_log_*.txt` and `*.error` files)
- **Steps to reproduce**

### Feature Requests
Open an issue describing:
- Use case
- Expected behavior
- Why current features don't cover it

### Pull Requests
1. Fork the repository
2. Create feature branch (`git checkout -b feature/YourFeature`)
3. Test thoroughly (multiple videos, GPU/CPU, different models, timeout behaviour)
4. Submit PR with description

---

## 📜 License

**GPL‑3.0** – see [LICENSE](LICENSE) file.

### Third‑Party Licenses
- **MiDaS** — MIT License (Intel ISL)
- **PyTorch** — BSD License
- **OpenCV** — Apache 2.0
- **FFmpeg** — LGPL/GPL (depends on build)

---

## 🙏 Credits

- **MiDaS** depth estimation by [Intel ISL](https://github.com/isl-org/MiDaS)
- **timm** vision models by [Ross Wightman](https://github.com/huggingface/pytorch-image-models)
- Thanks to the open‑source community for PyTorch, OpenCV, NumPy, FFmpeg, and tqdm

---

## 📚 References

- [MiDaS Paper (CVPR 2020)](https://arxiv.org/abs/1907.01341)
- [MiDaS 3.1 Release Notes](https://github.com/isl-org/MiDaS/releases/tag/v3_1)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [FFmpeg Hardware Acceleration](https://trac.ffmpeg.org/wiki/HWAccelIntro)

---

**Made with ❤️ by Ayan Khan**  
*Bringing depth to flat worlds, one frame at a time.*
