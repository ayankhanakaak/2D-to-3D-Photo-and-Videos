# @title 2D to 3D Photo and Video V.15.1.2026-3
'''
Made by: Ayan Khan
Version: 15.1.2026-3 (Multi-GPU Subprocess-Based, Fixed Race Condition)
'''

print("Loading...")
import os, shutil, subprocess, sys, time, json
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from tqdm import tqdm

# ---------- Helper Functions ----------
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

def try_reencode_video(input_path: str, crf: int = 18) -> str:
    base, _ = os.path.splitext(input_path)
    out_path = f"{base}_reencoded.mp4"
    try:
        duration = float(subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", input_path
        ], text=True).strip())
    except Exception:
        duration = None

    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", "-progress", "pipe:1", "-nostats", out_path
    ]
    print("\nPre-encoding with FFmpeg...")
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        pbar = tqdm(total=duration, desc="FFmpeg Pre-encode", unit="sec", mininterval=1) if duration else None
        last_time = 0.0
        for line in process.stdout:
            if line.startswith("out_time_ms"):
                current_time = int(line.split("=")[1]) / 1_000_000
                if pbar:
                    pbar.update(max(0.0, current_time - last_time))
                last_time = current_time
        process.wait()
        if pbar:
            pbar.close()
        if process.returncode != 0:
            raise RuntimeError()
        return out_path
    except Exception:
        return input_path

def merge_videos_ffmpeg(chunk_paths: list, output_path: str) -> bool:
    """Merge video chunks using FFmpeg concat demuxer"""
    list_path = output_path + "_concat_list.txt"
    try:
        with open(list_path, 'w') as f:
            for path in chunk_paths:
                escaped_path = os.path.abspath(path).replace("\\", "/")
                f.write(f"file '{escaped_path}'\n")
        
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", list_path, "-c", "copy", output_path
        ]
        
        print("\nMerging video chunks...")
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove(list_path)
        return result.returncode == 0
    except Exception as e:
        print(f"FFmpeg merge failed: {e}")
        return False


def precache_model(model_type: str):
    """Pre-download and cache the model before spawning workers"""
    print(f"\n📥 Pre-caching {model_type} model (avoids race condition)...")
    
    # This downloads and caches the model
    _ = torch.hub.load("intel-isl/MiDaS", model_type, verbose=True)
    _ = torch.hub.load("intel-isl/MiDaS", "transforms", verbose=False)
    
    # Clear from GPU memory (we just wanted to cache it)
    del _
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print("✅ Model cached successfully")


# ---------- Create Worker Script ----------
def create_worker_script(script_path: str):
    """Create a standalone Python script for GPU workers"""
    
    worker_code = '''#!/usr/bin/env python3
"""GPU Worker Script - Runs in separate process"""

import os, sys, json, time

# Read config from command line argument
config_path = sys.argv[1]
with open(config_path, 'r') as f:
    config = json.load(f)

# Set GPU visibility BEFORE importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu_id"])
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"

import torch
import torch.nn.functional as F
import cv2
import numpy as np

def normalize_depth_safe(depth):
    depth = depth.astype(np.float32, copy=False)
    mask = np.isfinite(depth)
    if not mask.any():
        return np.full(depth.shape, 0.5, dtype=np.float32)
    dmin, dmax = float(np.min(depth[mask])), float(np.max(depth[mask]))
    denom = (dmax - dmin) if dmax > dmin else 1.0
    out = np.empty_like(depth, dtype=np.float32)
    out[mask] = (depth[mask] - dmin) / denom
    out[~mask] = 0.5
    np.clip(out, 0.0, 1.0, out=out)
    return out

def main():
    gpu_id = config["gpu_id"]
    model_type = config["model_type"]
    video_path = config["video_path"]
    output_chunk_path = config["output_chunk_path"]
    start_frame = config["start_frame"]
    end_frame = config["end_frame"]
    fps = config["fps"]
    w, h = config["w"], config["h"]
    max_shift = config["max_shift"]
    output_choice = config["output_choice"]
    use_amp = config["use_amp"]
    progress_file = config["progress_file"]
    
    try:
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True
        
        # Load model (should use cache, no download needed)
        midas = torch.hub.load("intel-isl/MiDaS", model_type, verbose=False).to(device).eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", verbose=False)
        transform = midas_transforms.small_transform if model_type == "MiDaS_small" else midas_transforms.dpt_transform
        
        # Open video and seek
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"GPU {gpu_id}: Failed to open video")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Precompute grids
        grid_x, grid_y = np.meshgrid(
            np.arange(w, dtype=np.float32),
            np.arange(h, dtype=np.float32)
        )
        
        # Setup writer
        out_w, out_h = (w, h) if output_choice == 1 else (2 * w, h)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_chunk_path, fourcc, fps, (out_w, out_h))
        
        if not writer.isOpened():
            raise RuntimeError(f"GPU {gpu_id}: Failed to create writer")
        
        frames_done = 0
        last_progress_write = time.time()
        
        # Processing loop
        for frame_idx in range(start_frame, end_frame):
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                break
            
            img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            input_batch = transform(img_rgb).to(device)
            
            with torch.no_grad():
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        prediction = midas(input_batch)
                else:
                    prediction = midas(input_batch)
                
                prediction = F.interpolate(
                    prediction.unsqueeze(1), size=(h, w),
                    mode="bicubic", align_corners=False
                ).squeeze(1)
                
                depth = prediction.squeeze().cpu().float().numpy()
            
            if not np.isfinite(depth).all():
                depth = np.nan_to_num(depth, nan=0.5, posinf=1.0, neginf=0.0)
            
            depth_norm = normalize_depth_safe(depth)
            
            shift_map = depth_norm * float(max_shift)
            map_x_left = grid_x - shift_map
            map_x_right = grid_x + shift_map
            
            left_view = cv2.remap(frame_bgr, map_x_left, grid_y,
                                  interpolation=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
            right_view = cv2.remap(frame_bgr, map_x_right, grid_y,
                                   interpolation=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
            
            if output_choice == 1:
                output_frame = np.zeros_like(frame_bgr)
                output_frame[:, :, 2] = left_view[:, :, 2]
                output_frame[:, :, 1] = right_view[:, :, 1]
                output_frame[:, :, 0] = right_view[:, :, 0]
            else:
                output_frame = np.hstack((left_view, right_view))
            
            writer.write(output_frame)
            frames_done += 1
            
            # Update progress file periodically (every 0.2 seconds)
            current_time = time.time()
            if current_time - last_progress_write >= 0.2:
                with open(progress_file, 'w') as f:
                    f.write(str(frames_done))
                last_progress_write = current_time
        
        # Final progress update
        with open(progress_file, 'w') as f:
            f.write(str(frames_done))
        
        cap.release()
        writer.release()
        
        # Write success marker
        with open(progress_file + ".done", 'w') as f:
            f.write("OK")
        
    except Exception as e:
        # Write error marker
        with open(progress_file + ".error", 'w') as f:
            f.write(str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    with open(script_path, 'w') as f:
        f.write(worker_code)
    
    return script_path


# ---------- Single GPU Fallback ----------
def normalize_depth_safe(depth: np.ndarray) -> np.ndarray:
    depth = depth.astype(np.float32, copy=False)
    mask = np.isfinite(depth)
    if not mask.any():
        return np.full(depth.shape, 0.5, dtype=np.float32)
    dmin, dmax = float(np.min(depth[mask])), float(np.max(depth[mask]))
    denom = (dmax - dmin) if dmax > dmin else 1.0
    out = np.empty_like(depth, dtype=np.float32)
    out[mask] = (depth[mask] - dmin) / denom
    out[~mask] = 0.5
    np.clip(out, 0.0, 1.0, out=out)
    return out


def single_gpu_process(
    model_type: str,
    video_path: str,
    out_path: str,
    max_shift: int,
    output_choice: int,
    use_amp: bool,
    use_cuda: bool
):
    """Fallback for single GPU or CPU processing"""
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        torch.backends.cudnn.benchmark = True
    
    print("\nInitializing model...")
    midas = torch.hub.load("intel-isl/MiDaS", model_type).to(device).eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform if model_type == "MiDaS_small" else midas_transforms.dpt_transform
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video.")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32)
    )
    
    out_w, out_h = (w, h) if output_choice == 1 else (2 * w, h)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))
    
    if not writer.isOpened():
        print(f"ERROR: Could not open writer for path: {out_path}")
        cap.release()
        return 0
    
    print(f"\nProcessing {total or '?'} frames on {device}"
          f"{' (AMP enabled)' if (use_cuda and use_amp) else ''}...")
    
    pbar = tqdm(total=total, desc="Processing", unit="frame", mininterval=1)
    frames_written = 0
    
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                break
            
            img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            input_batch = transform(img_rgb).to(device)
            
            with torch.no_grad():
                if use_cuda and use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        prediction = midas(input_batch)
                else:
                    prediction = midas(input_batch)
                
                prediction = F.interpolate(
                    prediction.unsqueeze(1),
                    size=(h, w),
                    mode="bicubic",
                    align_corners=False
                ).squeeze(1)
                
                depth = prediction.squeeze().cpu().float().numpy()
            
            if not np.isfinite(depth).all():
                depth = np.nan_to_num(depth, nan=0.5, posinf=1.0, neginf=0.0)
            
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
                anaglyph[:, :, 2] = left_view[:, :, 2]
                anaglyph[:, :, 1] = right_view[:, :, 1]
                anaglyph[:, :, 0] = right_view[:, :, 0]
                writer.write(anaglyph)
            else:
                writer.write(np.hstack((left_view, right_view)))
            
            frames_written += 1
            pbar.update(1)
    
    finally:
        cap.release()
        writer.release()
        pbar.close()
    
    return frames_written


# ============================================================
#                        MAIN PROGRAM
# ============================================================

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

# ---------- Device Detection ----------
num_gpus = torch.cuda.device_count()
if num_gpus > 0:
    print(f"\n🔍 Found {num_gpus} GPU(s):")
    for i in range(num_gpus):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("\n🔍 No GPUs found, will use CPU")

use_cuda = False
use_multi_gpu = False

if num_gpus > 0:
    while True:
        try:
            if num_gpus > 1:
                dev_choice = int(input(f"\nGPU Options:\n1. Use all {num_gpus} GPUs (fastest)\n2. Use single GPU\n3. Use CPU only\n>>> "))
                if dev_choice == 1:
                    use_cuda = True
                    use_multi_gpu = True
                    break
                elif dev_choice == 2:
                    use_cuda = True
                    use_multi_gpu = False
                    break
                elif dev_choice == 3:
                    use_cuda = False
                    use_multi_gpu = False
                    break
                print("Enter 1–3!\n")
            else:
                dev_choice = int(input("\nUse GPU?\n1. Yes\n2. No (CPU)\n>>> "))
                if dev_choice == 1:
                    use_cuda = True
                    break
                elif dev_choice == 2:
                    use_cuda = False
                    break
                print("Enter 1–2!\n")
        except ValueError:
            print("Enter a valid number!\n")

use_amp = False
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

# ---------- Pre-encode if needed ----------
if recode_choice == 1:
    if is_safe_format(video_path):
        print("✅ Video already in safe format, skipping pre-encode.")
    else:
        video_path = try_reencode_video(video_path)

# ---------- Get video info ----------
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError("Failed to open video.")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

print(f"\n📹 Video info: {total_frames} frames, {w}x{h}, {fps:.2f} FPS")

# ============================================================
#                   MULTI-GPU PROCESSING
# ============================================================

if use_multi_gpu and num_gpus > 1:
    
    # ⭐ PRE-CACHE MODEL BEFORE SPAWNING WORKERS ⭐
    precache_model(model_type)
    
    # Free GPU memory after caching (model is saved to disk cache)
    torch.cuda.empty_cache()
    
    print(f"\n⚡ Starting Multi-GPU processing with {num_gpus} GPUs...")
    
    # Calculate frame ranges for each GPU
    frames_per_gpu = total_frames // num_gpus
    frame_ranges = []
    
    for i in range(num_gpus):
        start = i * frames_per_gpu
        end = total_frames if i == num_gpus - 1 else (i + 1) * frames_per_gpu
        frame_ranges.append((start, end))
        print(f"   GPU {i}: Frames {start} → {end - 1} ({end - start} frames)")
    
    # Create temp/output directory (ensure it exists)
    temp_dir = os.path.dirname(out_path) or "."
    if temp_dir and temp_dir != ".":
        os.makedirs(temp_dir, exist_ok=True)  # ← ADD THIS LINE
    temp_prefix = f"_temp_{os.getpid()}_"
    
    # Create worker script
    worker_script_path = os.path.join(temp_dir, f"{temp_prefix}worker.py")
    create_worker_script(worker_script_path)
    
    # Create config files and paths for each GPU
    chunk_paths = []
    config_paths = []
    progress_files = []
    
    for i in range(num_gpus):
        chunk_path = os.path.join(temp_dir, f"{temp_prefix}chunk_{i}.mp4")
        config_path = os.path.join(temp_dir, f"{temp_prefix}config_{i}.json")
        progress_file = os.path.join(temp_dir, f"{temp_prefix}progress_{i}.txt")
        
        chunk_paths.append(chunk_path)
        config_paths.append(config_path)
        progress_files.append(progress_file)
        
        # Initialize progress file
        with open(progress_file, 'w') as f:
            f.write("0")
        
        # Write config
        start_frame, end_frame = frame_ranges[i]
        config = {
            "gpu_id": i,
            "model_type": model_type,
            "video_path": os.path.abspath(video_path),
            "output_chunk_path": os.path.abspath(chunk_path),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "fps": fps,
            "w": w,
            "h": h,
            "max_shift": max_shift,
            "output_choice": output_choice,
            "use_amp": use_amp,
            "progress_file": os.path.abspath(progress_file)
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f)
    
    # Start subprocess for each GPU
    processes = []
    print(f"\n🚀 Launching {num_gpus} worker processes...")
    
    for i in range(num_gpus):
        cmd = [sys.executable, worker_script_path, config_paths[i]]
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True
        )
        processes.append(p)
        print(f"   Started GPU {i} worker (PID: {p.pid})")
    
    # Monitor progress
    print(f"\nProcessing {total_frames} frames on {num_gpus} GPUs"
          f"{' (AMP enabled)' if use_amp else ''}...")
    
    pbar = tqdm(total=total_frames, desc="Processing", unit="frame", mininterval=1)
    last_total = 0
    start_time = time.time()
    
    try:
        while True:
            # Check if all processes are done
            all_done = all(p.poll() is not None for p in processes)
            
            # Read progress from all workers
            current_total = 0
            for pf in progress_files:
                try:
                    with open(pf, 'r') as f:
                        content = f.read().strip()
                        if content:
                            current_total += int(content)
                except (FileNotFoundError, ValueError):
                    pass
            
            # Update progress bar
            if current_total > last_total:
                pbar.update(current_total - last_total)
                last_total = current_total
            
            if all_done:
                break
            
            time.sleep(0.2)
        
        # Final progress update
        current_total = 0
        for pf in progress_files:
            try:
                with open(pf, 'r') as f:
                    content = f.read().strip()
                    if content:
                        current_total += int(content)
            except (FileNotFoundError, ValueError):
                pass
        
        if current_total > last_total:
            pbar.update(current_total - last_total)
    
    finally:
        pbar.close()
    
    elapsed = time.time() - start_time
    
    # Check for errors
    errors = []
    for i, pf in enumerate(progress_files):
        error_file = pf + ".error"
        if os.path.exists(error_file):
            with open(error_file, 'r') as f:
                errors.append(f"GPU {i}: {f.read()}")
    
    if errors:
        print("\n⚠️ Warnings occurred (may not affect output):")
        for err in errors:
            print(f"   {err}")
    
    # Check all chunks exist
    missing_chunks = [p for p in chunk_paths if not os.path.exists(p)]
    
    if missing_chunks:
        print(f"\n❌ Missing chunks: {missing_chunks}")
        
        # Print stderr from failed processes
        for i, p in enumerate(processes):
            if p.returncode != 0:
                stderr = p.stderr.read()
                if stderr:
                    print(f"\n   GPU {i} stderr:\n{stderr[:1000]}")
    else:
        # Merge chunks
        print(f"\n📦 Merging {num_gpus} video chunks...")
        
        if merge_videos_ffmpeg(chunk_paths, out_path):
            # Cleanup temp files
            for path in chunk_paths + config_paths + progress_files:
                if os.path.exists(path):
                    os.remove(path)
                # Also remove .done and .error files
                for suffix in [".done", ".error"]:
                    if os.path.exists(path + suffix):
                        os.remove(path + suffix)
            
            if os.path.exists(worker_script_path):
                os.remove(worker_script_path)
            
            frames_processed = last_total
            avg_fps = frames_processed / elapsed if elapsed > 0 else 0
            
            print(f"\n✅ Done! Frames: {frames_processed}, Time: {elapsed:.1f}s, Avg: {avg_fps:.2f} FPS")
            print(f"   Saved: {out_path}")
        else:
            print("\n⚠️ FFmpeg merge failed. Keeping chunk files:")
            for path in chunk_paths:
                if os.path.exists(path):
                    print(f"   {path}")

# ============================================================
#                 SINGLE GPU / CPU PROCESSING
# ============================================================

else:
    frames_written = single_gpu_process(
        model_type=model_type,
        video_path=video_path,
        out_path=out_path,
        max_shift=max_shift,
        output_choice=output_choice,
        use_amp=use_amp,
        use_cuda=use_cuda
    )
    print(f"\n✅ Done. Frames written: {frames_written}. Saved: {out_path}")