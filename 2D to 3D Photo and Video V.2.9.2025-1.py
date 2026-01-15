'''
Made by: Ayan Khan
Version: 2.9.2025-1
'''

print("Loading...")
import os, shutil, subprocess
import torch
import cv2
import numpy as np

# ---------- Helpers ----------
def next_available_filename(base_dir, base_name, ext):
    i = 1
    while True:
        candidate = os.path.join(base_dir, f"{base_name}-{i}{ext}")
        if not os.path.exists(candidate):
            return candidate
        i += 1

def normalize_depth_safe(depth: np.ndarray) -> np.ndarray:
    # Map to [0,1] robustly, handling NaNs/Infs and constant arrays
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
    # Clip for numerical safety
    np.clip(out, 0.0, 1.0, out=out)
    return out

def try_reencode_video(input_path: str, crf: int = 18) -> str:
    base, ext = os.path.splitext(input_path)
    out_path = f"{base}_reencoded.mp4"
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
        out_path
    ]
    try:
        print("Pre-encoding with FFmpeg for reliable decoding...")
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Re-encoded: {out_path}")
        return out_path
    except Exception:
        print("FFmpeg re-encode skipped (not available or failed). Using original video.")
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
                    model_path = input("Enter full model path: ").strip()
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

# ---------- Video / Output ----------
while True:
    video_path = input("\nInput Video >>> ").strip()
    if os.path.isfile(video_path):
        break
    print("File not found!")

# Optional: normalize container/codec for reliable decoding
while True:
    try:
        recode_choice = int(input("Pre-encode video with FFmpeg for reliability?\n1. Yes\n2. No\n>>> "))
        if recode_choice in (1, 2):
            break
        print("Enter 1–2!\n")
    except ValueError:
        print("Enter 1–2!\n")
if recode_choice == 1:
    video_path = try_reencode_video(video_path)

while True:
    try:
        output_choice = int(input("Choose output:\n1. Red/Cyan Anaglyph\n2. SBS Stereo\n>>> "))
        if output_choice in (1, 2):
            break
        print("Enter 1–2!\n")
    except ValueError:
        print("Enter 1–2!\n")

try:
    max_shift = int(input("Max shift px [default 15] >>> ") or "15")
except ValueError:
    max_shift = 15

# ---------- Load model ----------
print("\nInitializing model...")
midas = torch.hub.load("intel-isl/MiDaS", model_type).to(device).eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform if model_type == "MiDaS_small" else midas_transforms.dpt_transform

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
base_dir = os.path.dirname(video_path)
if output_choice == 1:
    base_name, out_w, out_h = "output_anaglyph_video", w, h
else:
    base_name, out_w, out_h = "output_sbs_video", 2*w, h

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_path = next_available_filename(base_dir, base_name, ".mp4")
writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))

print(f"\nProcessing {total or '?'} frames on {device}"
      f"{' (AMP enabled)' if (use_cuda and use_amp) else ''}...")
frame_idx = 0

# ---------- Processing loop ----------
try:
    while True:
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            # Clean end of stream
            break

        # RGB for MiDaS
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        input_batch = transform(img_rgb).to(device)

        with torch.no_grad():
            # First attempt (AMP if chosen)
            if use_cuda and use_amp:
                try:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        prediction = midas(input_batch)
                except Exception:
                    prediction = midas(input_batch)
            else:
                prediction = midas(input_batch)

            # Resize to (H, W)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False
            ).squeeze(1)

            depth = prediction.squeeze().detach().to("cpu").float().numpy()

            # If non-finite depth appears, retry once in full precision
            if not np.isfinite(depth).all():
                if use_cuda and use_amp:
                    # Recompute this frame without AMP
                    prediction_fp32 = midas(input_batch)  # no autocast here
                    prediction_fp32 = torch.nn.functional.interpolate(
                        prediction_fp32.unsqueeze(1),
                        size=(h, w),
                        mode="bicubic",
                        align_corners=False
                    ).squeeze(1)
                    depth = prediction_fp32.squeeze().detach().to("cpu").float().numpy()

        # Robust normalization (never errors)
        depth_norm = normalize_depth_safe(depth)

        # Backward warping (no holes)
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
            # Red/Cyan anaglyph
            anaglyph = np.zeros_like(frame_bgr)
            anaglyph[:, :, 2] = left_view[:, :, 2]   # R
            anaglyph[:, :, 1] = right_view[:, :, 1]  # G
            anaglyph[:, :, 0] = right_view[:, :, 0]  # B
            writer.write(anaglyph)
        else:
            # Side-by-side stereo
            sbs = np.hstack((left_view, right_view))
            writer.write(sbs)

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx} frames...")

finally:
    cap.release()
    writer.release()

print(f"\n✅ Done. Frames written: {frame_idx}. Saved: {out_path}")
