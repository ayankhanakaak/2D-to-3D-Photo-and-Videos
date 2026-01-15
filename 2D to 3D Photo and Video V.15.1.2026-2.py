# @title 2D to 3D Photo and Video V.15.1.2026-2
'''
Made by: Ayan Khan
Version: 15.1.2026-2
'''

print("Loading...")
import os, shutil, subprocess, sys
# Suppress OpenCV warnings that break tqdm lines
os.environ["OPENCV_LOG_LEVEL"] = "FATAL" 

import torch
import cv2
import numpy as np
from tqdm import tqdm

def is_safe_format(video_path: str) -> bool:
    try:
        vid_info = subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,pix_fmt",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path
        ], text=True).splitlines()

        is_mp4 = os.path.splitext(video_path)[1].lower() == ".mp4"
        is_h264 = any("h264" in x for x in vid_info)
        is_yuv420p = any("yuv420p" in x for x in vid_info)

        return is_mp4 and is_h264 and is_yuv420p
    except Exception:
        return False

# ---------- Helpers ----------
def normalize_depth_safe(depth: np.ndarray) -> np.ndarray:
    depth = depth.astype(np.float32, copy=False)
    mask = np.isfinite(depth)
    if not mask.any():
        return np.full(depth.shape, 0.5, dtype=np.float32)
    dmin = float(np.min(depth[mask]))
    dmax = float(np.max(depth[mask]))
    denom = (dmax - dmin) if dmax > dmin else 1.0
    out = np.empty_like(depth, dtype=np.float32)
    out[mask] = (depth[mask] - dmin) / denom
    out[~mask] = 0.5
    np.clip(out, 0.0, 1.0, out=out)
    return out

def try_reencode_video(input_path: str, crf: int = 18) -> str:
    base, ext = os.path.splitext(input_path)
    out_path = f"{base}_reencoded.mp4"

    try:
        duration = float(subprocess.check_output([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            input_path
        ], text=True).strip())
    except Exception:
        duration = None

    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        "-progress", "pipe:1",
        "-nostats",
        out_path
    ]

    print("\nPre-encoding with FFmpeg for reliable decoding...")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1
        )

        if duration:
            pbar = tqdm(total=duration, desc="FFmpeg Pre-encode", unit="sec", mininterval=1)

        last_time = 0.0
        for line in process.stdout:
            if line.startswith("out_time_ms"):
                current_time = int(line.split("=")[1]) / 1_000_000
                if duration:
                    pbar.update(max(0.0, current_time - last_time))
                last_time = current_time

        process.wait()
        if duration:
            pbar.close()

        if process.returncode != 0:
            raise RuntimeError("FFmpeg failed")

        return out_path
    except Exception:
        return input_path

# ---------- Model selection ----------
while True:
    try:
        choice = int(input("Choose model:\n1. DPT_Large\n2. DPT_Hybrid\n3. MiDaS_small\n>>> "))
        if choice == 1:
            model_type, model_name = "DPT_Large", "dpt_large_384.pt"
            break
        elif choice == 2:
            model_type, model_name = "DPT_Hybrid", "dpt_hybrid_384.pt"
            break
        elif choice == 3:
            model_type, model_name = "MiDaS_small", "midas_v21_small_256.pt"
            break
        print("Enter 1–3!\n")
    except ValueError:
        print("Enter a number 1–3!\n")

# ---------- Model source ----------
while True:
    try:
        answer = int(input("Model source:\n1. Already Downloaded\n2. Download Automatically/Use Cached\n>>> "))
        if answer == 1:
            while True:
                try:
                    model_path = input("Enter full model path: ").strip().strip('"').strip("'")
                    print("Checking...\n")
                    dest_dir = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints")
                    os.makedirs(dest_dir, exist_ok=True)
                    shutil.copy2(model_path, os.path.join(dest_dir, model_name))
                    break
                except FileNotFoundError:
                    print("File not found!\n")
            break
        elif answer == 2:
            break
        print("Enter 1–2!\n")
    except ValueError:
        print("Enter 1–2!\n")

# ---------- Device ----------
use_cuda, use_amp = False, False
while True:
    try:
        dev_choice = int(input("Use CUDA if available?\n1. Yes\n2. No (CPU)\n>>> "))
        if dev_choice in (1, 2):
            use_cuda = (dev_choice == 1) and torch.cuda.is_available()
            break
        print("Enter 1–2!\n")
    except ValueError:
        print("Enter 1–2!\n")

if use_cuda:
    while True:
        try:
            amp_choice = int(input("Enable mixed precision (faster)?\n1. Yes\n2. No\n>>> "))
            if amp_choice in (1, 2):
                use_amp = (amp_choice == 1)
                break
            print("Enter 1–2!\n")
        except ValueError:
            print("Enter 1–2!\n")

device = torch.device("cuda" if use_cuda else "cpu")
if use_cuda:
    torch.backends.cudnn.benchmark = True

# ---------- Video / Output Paths ----------
while True:
    video_path = input("\nInput Video Path >>> ").strip().strip('"').strip("'")
    if os.path.isfile(video_path):
        break
    print("File not found!")

while True:
    out_path = input("Output Video Path (e.g., output.mp4) >>> ").strip().strip('"').strip("'")
    if out_path:
        if not out_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            out_path += ".mp4"
        break
    print("Output path cannot be empty!")

while True:
    try:
        recode_choice = int(input("\nPre-encode video with FFmpeg for reliability?\n1. Yes\n2. No\n>>> "))
        if recode_choice in (1, 2):
            break
        print("Enter 1–2!\n")
    except ValueError:
        print("Enter 1–2!\n")

while True:
    try:
        output_choice = int(input("\nChoose output type:\n1. Red/Cyan Anaglyph\n2. SBS Stereo\n>>> "))
        if output_choice in (1, 2):
            break
        print("Enter 1–2!\n")
    except ValueError:
        print("Enter 1–2!\n")

try:
    max_shift = int(input("\nMax shift px [default 15] >>> ") or "15")
except ValueError:
    max_shift = 15

# ---------- Load model ----------
print("\nInitializing model...")
midas = torch.hub.load("intel-isl/MiDaS", model_type).to(device).eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform if model_type == "MiDaS_small" else midas_transforms.dpt_transform

if recode_choice == 1:
    if is_safe_format(video_path):
        print("✅ Video already in safe format, skipping pre-encode.")
    else:
        video_path = try_reencode_video(video_path)

# ---------- Open video ----------
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError("Failed to open video.")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Precompute grids
grid_x, grid_y = np.meshgrid(
    np.arange(w, dtype=np.float32),
    np.arange(h, dtype=np.float32)
)

# ---------- Writer ----------
if output_choice == 1:
    out_w, out_h = w, h
else:
    out_w, out_h = 2*w, h

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))

if not writer.isOpened():
    print(f"ERROR: Could not open writer for path: {out_path}")
    cap.release()
    exit(1)

print(f"\nProcessing {total or '?'} frames on {device}"
      f"{' (AMP enabled)' if (use_cuda and use_amp) else ''}...")

# UPDATED: Added file=sys.stdout and dynamic_ncols=True to fix progress bar issues
pbar = tqdm(total=total, desc="Processing", unit="frame", mininterval=1)
frame_idx = 0

# ---------- Processing loop ----------
try:
    while True:
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            break

        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        input_batch = transform(img_rgb).to(device)

        with torch.no_grad():
            if use_cuda and use_amp:
                try:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        prediction = midas(input_batch)
                except Exception:
                    prediction = midas(input_batch)
            else:
                prediction = midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False
            ).squeeze(1)

            depth = prediction.squeeze().detach().to("cpu").float().numpy()

            if not np.isfinite(depth).all() and use_cuda and use_amp:
                prediction_fp32 = midas(input_batch)
                prediction_fp32 = torch.nn.functional.interpolate(
                    prediction_fp32.unsqueeze(1),
                    size=(h, w),
                    mode="bicubic",
                    align_corners=False
                ).squeeze(1)
                depth = prediction_fp32.squeeze().detach().to("cpu").float().numpy()

        depth_norm = normalize_depth_safe(depth)

        shift_map = depth_norm * float(max_shift)
        map_x_left = grid_x - shift_map
        map_x_right = grid_x + shift_map

        left_view = cv2.remap(
            frame_bgr, map_x_left, grid_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
        right_view = cv2.remap(
            frame_bgr, map_x_right, grid_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        if output_choice == 1:
            anaglyph = np.zeros_like(frame_bgr)
            anaglyph[:, :, 2] = left_view[:, :, 2]   # R
            anaglyph[:, :, 1] = right_view[:, :, 1]  # G
            anaglyph[:, :, 0] = right_view[:, :, 0]  # B
            writer.write(anaglyph)
        else:
            sbs = np.hstack((left_view, right_view))
            writer.write(sbs)

        frame_idx += 1
        pbar.update(1)

finally:
    cap.release()
    writer.release()
    pbar.close()

print(f"\n✅ Done. Frames written: {frame_idx}. Saved: {out_path}")