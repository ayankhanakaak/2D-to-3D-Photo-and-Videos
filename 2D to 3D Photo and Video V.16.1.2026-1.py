# @title 2D to 3D Photo and Video V.16.1.2026-1
'''
Made by: Ayan Khan
Version: 16.1.2026-1 (Error Capture + Progress Bar Fix)
'''

print("Loading...")
import os, shutil, subprocess, sys, time, json
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
os.environ["PYTHONUNBUFFERED"] = "1"

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from tqdm import tqdm

# ---------- Feature Detection ----------
def check_cuda_remap_support():
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            test_mat = cv2.cuda_GpuMat(10, 10, cv2.CV_32FC1)
            return True
    except:
        pass
    return False

def check_nvenc_support():
    try:
        result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-encoders'],
            capture_output=True, text=True, timeout=10
        )
        return 'h264_nvenc' in result.stdout
    except:
        return False

def check_nvdec_support():
    try:
        result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-hwaccels'],
            capture_output=True, text=True, timeout=10
        )
        return 'cuda' in result.stdout
    except:
        return False

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
    except:
        return False

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

def try_reencode_video(input_path: str, crf: int = 18) -> str:
    base, _ = os.path.splitext(input_path)
    out_path = f"{base}_reencoded.mp4"
    try:
        duration = float(subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", input_path
        ], text=True).strip())
    except:
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
        pbar = tqdm(total=duration, desc="FFmpeg Pre-encode", unit="sec", mininterval=1, dynamic_ncols=True) if duration else None
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
    except:
        return input_path

def merge_videos_ffmpeg(chunk_paths: list, output_path: str) -> bool:
    list_path = output_path + "_concat_list.txt"
    try:
        with open(list_path, 'w') as f:
            for path in chunk_paths:
                escaped_path = os.path.abspath(path).replace("\\", "/")
                f.write(f"file '{escaped_path}'\n")
        
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", output_path]
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove(list_path)
        return result.returncode == 0
    except Exception as e:
        print(f"FFmpeg merge failed: {e}")
        return False

def precache_model(model_type: str):
    print(f"\n📥 Pre-caching {model_type} model...")
    _ = torch.hub.load("intel-isl/MiDaS", model_type, verbose=True)
    _ = torch.hub.load("intel-isl/MiDaS", "transforms", verbose=False)
    del _
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("✅ Model cached successfully")

# ---------- Worker Script ----------
def create_worker_script(script_path: str):
    worker_code = '''#!/usr/bin/env python3
"""GPU Worker V16 - Error Capture + Progress Bar Fix"""
import os, sys, json, time, subprocess, traceback

config_path = sys.argv[1]
with open(config_path, 'r') as f:
    config = json.load(f)

gpu_id = config["gpu_id"]
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
os.environ["PYTHONUNBUFFERED"] = "1"

import torch
import torch.nn.functional as F
import cv2
import numpy as np

# Config
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
batch_size = config.get("batch_size", 4)
use_nvdec = config.get("use_nvdec", False)
use_nvenc = config.get("use_nvenc", False)
use_cuda_remap = config.get("use_cuda_remap", False)
log_file = config.get("log_file", progress_file + ".log")

def log(msg):
    with open(log_file, 'a') as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\\n")

log(f"Worker started: GPU {gpu_id}, batch={batch_size}, nvdec={use_nvdec}, nvenc={use_nvenc}")

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

class NVDECReader:
    def __init__(self, video_path, start_frame, end_frame, w, h, fps):
        self.w, self.h = w, h
        self.frame_size = w * h * 3
        self.frames_to_read = end_frame - start_frame
        self.frames_read = 0
        start_time = start_frame / fps
        
        cmd = [
            'ffmpeg', '-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda',
            '-ss', str(start_time), '-i', video_path,
            '-frames:v', str(self.frames_to_read),
            '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-vsync', '0', 'pipe:1'
        ]
        
        self.process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            bufsize=self.frame_size * 10,
            env={**os.environ, 'PYTHONUNBUFFERED': '1'}
        )
        
        test_data = self.process.stdout.read(1)
        if not test_data:
            raise RuntimeError("NVDEC failed to start")
        self._first_byte = test_data
        self._first_frame = True
    
    def read(self):
        if self.frames_read >= self.frames_to_read:
            return False, None
        if self._first_frame:
            raw = self._first_byte + self.process.stdout.read(self.frame_size - 1)
            self._first_frame = False
        else:
            raw = self.process.stdout.read(self.frame_size)
        if len(raw) != self.frame_size:
            return False, None
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.h, self.w, 3)).copy()
        self.frames_read += 1
        return True, frame
    
    def release(self):
        try:
            self.process.terminate()
            self.process.wait(timeout=5)
        except:
            self.process.kill()

class StandardReader:
    def __init__(self, video_path, start_frame, end_frame, w, h, fps):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open video")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        self.frames_to_read = end_frame - start_frame
        self.frames_read = 0
    
    def read(self):
        if self.frames_read >= self.frames_to_read:
            return False, None
        ret, frame = self.cap.read()
        if ret:
            self.frames_read += 1
        return ret, frame
    
    def release(self):
        self.cap.release()

class NVENCWriter:
    def __init__(self, output_path, w, h, fps):
        self.w, self.h = w, h
        cmd = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f'{w}x{h}', '-r', str(fps), '-i', 'pipe:0',
            '-c:v', 'h264_nvenc', '-preset', 'p4', '-rc', 'vbr',
            '-cq', '20', '-b:v', '0', '-pix_fmt', 'yuv420p', output_path
        ]
        self.process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            bufsize=w * h * 3 * 10,
            env={**os.environ, 'PYTHONUNBUFFERED': '1'}
        )
    
    def write(self, frame):
        try:
            self.process.stdin.write(frame.tobytes())
        except BrokenPipeError:
            pass
    
    def release(self):
        try:
            self.process.stdin.close()
            self.process.wait(timeout=30)
        except:
            pass

class StandardWriter:
    def __init__(self, output_path, w, h, fps):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to create writer")
    
    def write(self, frame):
        self.writer.write(frame)
    
    def release(self):
        self.writer.release()

class CUDARemapper:
    def __init__(self, w, h, max_shift):
        grid_x_np, grid_y_np = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        self.grid_x_gpu = cv2.cuda_GpuMat()
        self.grid_y_gpu = cv2.cuda_GpuMat()
        self.grid_x_gpu.upload(grid_x_np)
        self.grid_y_gpu.upload(grid_y_np)
        self.frame_gpu = cv2.cuda_GpuMat()
        self.map_x_gpu = cv2.cuda_GpuMat()
        self.left_gpu = cv2.cuda_GpuMat()
        self.right_gpu = cv2.cuda_GpuMat()
        self.grid_x_np = grid_x_np
        self.grid_y_np = grid_y_np
        self.max_shift = max_shift
    
    def remap(self, frame_bgr, depth_norm):
        shift_map = depth_norm * float(self.max_shift)
        map_x_left = (self.grid_x_np - shift_map).astype(np.float32)
        map_x_right = (self.grid_x_np + shift_map).astype(np.float32)
        self.frame_gpu.upload(frame_bgr)
        self.map_x_gpu.upload(map_x_left)
        cv2.cuda.remap(self.frame_gpu, self.map_x_gpu, self.grid_y_gpu, cv2.INTER_LINEAR, cv2.BORDER_REPLICATE, dst=self.left_gpu)
        left_view = self.left_gpu.download()
        self.map_x_gpu.upload(map_x_right)
        cv2.cuda.remap(self.frame_gpu, self.map_x_gpu, self.grid_y_gpu, cv2.INTER_LINEAR, cv2.BORDER_REPLICATE, dst=self.right_gpu)
        right_view = self.right_gpu.download()
        return left_view, right_view

class CPURemapper:
    def __init__(self, w, h, max_shift):
        self.grid_x, self.grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        self.max_shift = max_shift
    
    def remap(self, frame_bgr, depth_norm):
        shift_map = depth_norm * float(self.max_shift)
        map_x_left = self.grid_x - shift_map
        map_x_right = self.grid_x + shift_map
        left_view = cv2.remap(frame_bgr, map_x_left, self.grid_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        right_view = cv2.remap(frame_bgr, map_x_right, self.grid_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return left_view, right_view

def create_output_frame(left_view, right_view, output_choice):
    if output_choice == 1:
        out = np.zeros_like(left_view)
        out[:, :, 2] = left_view[:, :, 2]
        out[:, :, 1] = right_view[:, :, 1]
        out[:, :, 0] = right_view[:, :, 0]
        return out
    return np.hstack((left_view, right_view))

def main():
    try:
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True
        log(f"CUDA available: {torch.cuda.is_available()}")
        
        log("Loading model...")
        midas = torch.hub.load("intel-isl/MiDaS", model_type, verbose=False).to(device).eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", verbose=False)
        transform = midas_transforms.small_transform if model_type == "MiDaS_small" else midas_transforms.dpt_transform
        log("Model loaded")
        
        # Reader with fallback
        reader = None
        if use_nvdec:
            try:
                log("Trying NVDEC...")
                reader = NVDECReader(video_path, start_frame, end_frame, w, h, fps)
                log("NVDEC OK")
            except Exception as e:
                log(f"NVDEC failed: {e}")
        if reader is None:
            reader = StandardReader(video_path, start_frame, end_frame, w, h, fps)
            log("Using StandardReader")
        
        # Writer with fallback
        out_w, out_h = (w, h) if output_choice == 1 else (2 * w, h)
        writer = None
        if use_nvenc:
            try:
                log("Trying NVENC...")
                writer = NVENCWriter(output_chunk_path, out_w, out_h, fps)
                log("NVENC OK")
            except Exception as e:
                log(f"NVENC failed: {e}")
        if writer is None:
            writer = StandardWriter(output_chunk_path, out_w, out_h, fps)
            log("Using StandardWriter")
        
        # Remapper with fallback
        remapper = None
        if use_cuda_remap:
            try:
                remapper = CUDARemapper(w, h, max_shift)
                log("CUDA Remap OK")
            except Exception as e:
                log(f"CUDA Remap failed: {e}")
        if remapper is None:
            remapper = CPURemapper(w, h, max_shift)
            log("Using CPU Remap")
        
        frames_done = 0
        last_progress_write = time.time()
        total_frames = end_frame - start_frame
        log(f"Processing {total_frames} frames...")
        
        while frames_done < total_frames:
            batch_frames = []
            for _ in range(batch_size):
                ret, frame = reader.read()
                if not ret or frame is None:
                    break
                batch_frames.append(frame)
            
            if not batch_frames:
                break
            
            current_batch_size = len(batch_frames)
            input_tensors = [transform(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in batch_frames]
            batch_tensor = torch.cat(input_tensors, dim=0).to(device)
            
            with torch.no_grad():
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        predictions = midas(batch_tensor)
                else:
                    predictions = midas(batch_tensor)
                
                predictions = F.interpolate(predictions.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False).squeeze(1)
                depths = predictions.cpu().float().numpy()
            
            for i in range(current_batch_size):
                depth = depths[i]
                if not np.isfinite(depth).all():
                    depth = np.nan_to_num(depth, nan=0.5, posinf=1.0, neginf=0.0)
                depth_norm = normalize_depth_safe(depth)
                left_view, right_view = remapper.remap(batch_frames[i], depth_norm)
                output_frame = create_output_frame(left_view, right_view, output_choice)
                writer.write(output_frame)
                frames_done += 1
            
            if time.time() - last_progress_write >= 0.2:
                with open(progress_file, 'w') as f:
                    f.write(str(frames_done))
                last_progress_write = time.time()
        
        with open(progress_file, 'w') as f:
            f.write(str(frames_done))
        
        reader.release()
        writer.release()
        
        with open(progress_file + ".done", 'w') as f:
            f.write(f"OK:{frames_done}")
        log(f"Done: {frames_done} frames")
        
    except Exception as e:
        error_msg = f"{e}\\n{traceback.format_exc()}"
        log(f"ERROR: {error_msg}")
        with open(progress_file + ".error", 'w') as f:
            f.write(error_msg)
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    with open(script_path, 'w') as f:
        f.write(worker_code)

# ---------- Single GPU Processing ----------
def single_gpu_process(model_type, video_path, out_path, max_shift, output_choice, use_amp, use_cuda, batch_size=4, use_nvenc=False, use_cuda_remap=False):
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        torch.backends.cudnn.benchmark = True
    
    print("\nInitializing model...")
    midas = torch.hub.load("intel-isl/MiDaS", model_type).to(device).eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform if model_type == "MiDaS_small" else midas_transforms.dpt_transform
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w, out_h = (w, h) if output_choice == 1 else (2 * w, h)
    
    # Writer with fallback
    writer_process = None
    if use_nvenc:
        try:
            cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{out_w}x{out_h}', '-r', str(fps), '-i', 'pipe:0', '-c:v', 'h264_nvenc', '-preset', 'p4', '-rc', 'vbr', '-cq', '20', '-b:v', '0', '-pix_fmt', 'yuv420p', out_path]
            writer_process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, bufsize=out_w*out_h*3*10, env={**os.environ, 'PYTHONUNBUFFERED': '1'})
            print("✅ NVENC enabled")
        except:
            writer_process = None
    
    if writer_process is None:
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    
    print(f"\nProcessing {total} frames on {device} [Batch: {batch_size}]...")
    pbar = tqdm(total=total, desc="Processing", unit="frame", mininterval=1, dynamic_ncols=True, file=sys.stdout)
    frames_written = 0
    
    try:
        while True:
            batch_frames = []
            for _ in range(batch_size):
                ret, frame = cap.read()
                if not ret:
                    break
                batch_frames.append(frame)
            if not batch_frames:
                break
            
            input_tensors = [transform(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in batch_frames]
            batch_tensor = torch.cat(input_tensors, dim=0).to(device)
            
            with torch.no_grad():
                if use_cuda and use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        predictions = midas(batch_tensor)
                else:
                    predictions = midas(batch_tensor)
                predictions = F.interpolate(predictions.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False).squeeze(1)
                depths = predictions.cpu().float().numpy()
            
            for i, frame in enumerate(batch_frames):
                depth = depths[i]
                if not np.isfinite(depth).all():
                    depth = np.nan_to_num(depth, nan=0.5, posinf=1.0, neginf=0.0)
                depth_norm = normalize_depth_safe(depth)
                shift_map = depth_norm * float(max_shift)
                
                left = cv2.remap(frame, grid_x - shift_map, grid_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                right = cv2.remap(frame, grid_x + shift_map, grid_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                
                if output_choice == 1:
                    out = np.zeros_like(frame)
                    out[:,:,2], out[:,:,1], out[:,:,0] = left[:,:,2], right[:,:,1], right[:,:,0]
                else:
                    out = np.hstack((left, right))
                
                if writer_process:
                    writer_process.stdin.write(out.tobytes())
                else:
                    writer.write(out)
                frames_written += 1
                pbar.update(1)
    finally:
        cap.release()
        if writer_process:
            try:
                writer_process.stdin.close()
            except BrokenPipeError:
                pass
            try:
                writer_process.wait(timeout=10)
            except:
                writer_process.kill()
        else:
            writer.release()
        pbar.close()
    
    return frames_written

# ============================================================
#                        MAIN PROGRAM
# ============================================================

# Model selection
while True:
    try:
        choice = int(input("Choose model:\n1. DPT_Large\n2. DPT_Hybrid\n3. MiDaS_small\n>>> "))
        if choice == 1:
            model_type, model_name = "DPT_Large", "dpt_large_384.pt"; break
        elif choice == 2:
            model_type, model_name = "DPT_Hybrid", "dpt_hybrid_384.pt"; break
        elif choice == 3:
            model_type, model_name = "MiDaS_small", "midas_v21_small_256.pt"; break
        print("Enter 1–3!\n")
    except ValueError:
        print("Enter a number 1–3!\n")

# Model source
while True:
    try:
        answer = int(input("Model source:\n1. Already Downloaded\n2. Download Automatically/Use Cached\n>>> "))
        if answer == 1:
            while True:
                try:
                    model_path = input("Enter full model path: ").strip().strip('"').strip("'")
                    dest_dir = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints")
                    os.makedirs(dest_dir, exist_ok=True)
                    shutil.copy2(model_path, os.path.join(dest_dir, model_name)); break
                except FileNotFoundError:
                    print("File not found!\n")
            break
        elif answer == 2:
            break
        print("Enter 1–2!\n")
    except ValueError:
        print("Enter 1–2!\n")

# Device detection
num_gpus = torch.cuda.device_count()
if num_gpus > 0:
    print(f"\n🔍 Found {num_gpus} GPU(s):")
    for i in range(num_gpus):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("\n🔍 No GPUs found")

HAS_NVENC = check_nvenc_support()
HAS_NVDEC = check_nvdec_support()
HAS_CUDA_REMAP = check_cuda_remap_support()

print(f"\n🔧 Acceleration Support:")
print(f"   NVDEC: {'✅' if HAS_NVDEC else '❌'}  NVENC: {'✅' if HAS_NVENC else '❌'}  CUDA Remap: {'✅' if HAS_CUDA_REMAP else '❌'}")

use_cuda = use_multi_gpu = False
if num_gpus > 0:
    while True:
        try:
            if num_gpus > 1:
                dev = int(input(f"\nGPU Options:\n1. Use all {num_gpus} GPUs\n2. Single GPU\n3. CPU only\n>>> "))
                if dev == 1: use_cuda = use_multi_gpu = True; break
                elif dev == 2: use_cuda = True; break
                elif dev == 3: break
            else:
                dev = int(input("\nUse GPU?\n1. Yes\n2. No\n>>> "))
                if dev == 1: use_cuda = True; break
                elif dev == 2: break
            print("Invalid!\n")
        except ValueError:
            print("Enter a number!\n")

use_amp = use_nvdec = use_nvenc = use_cuda_remap = False
batch_size = 4

if use_cuda:
    use_amp = int(input("Mixed precision?\n1. Yes\n2. No\n>>> ") or "1") == 1
    
    print("\n⚡ Acceleration Options:")
    if HAS_NVDEC:
        use_nvdec = int(input("NVDEC (HW decode)?\n1. Yes\n2. No\n>>> ") or "2") == 1
    if HAS_NVENC:
        use_nvenc = int(input("NVENC (HW encode)?\n1. Yes\n2. No\n>>> ") or "1") == 1
    if HAS_CUDA_REMAP:
        use_cuda_remap = int(input("CUDA Remap?\n1. Yes\n2. No\n>>> ") or "2") == 1
    
    try:
        batch_size = int(input("Batch size [4]: ") or "4")
    except:
        batch_size = 4

# Paths
while True:
    video_path = input("\nInput Video >>> ").strip().strip('"').strip("'")
    if os.path.isfile(video_path): break
    print("Not found!")

while True:
    out_path = input("Output Video >>> ").strip().strip('"').strip("'")
    if out_path:
        if not out_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            out_path += ".mp4"
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        break
    print("Cannot be empty!")

recode = int(input("\nPre-encode for reliability?\n1. Yes\n2. No\n>>> ") or "2") == 1
output_choice = int(input("\nOutput type:\n1. Red/Cyan Anaglyph\n2. SBS Stereo\n>>> ") or "2")
max_shift = int(input("Max shift [15]: ") or "15")

if recode:
    if is_safe_format(video_path):
        print("✅ Already safe format")
    else:
        video_path = try_reencode_video(video_path)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

print(f"\n📹 Video: {total_frames} frames, {w}x{h}, {fps:.2f} FPS")
print(f"\n🚀 Config: GPUs={'Multi:'+str(num_gpus) if use_multi_gpu else 'Single' if use_cuda else 'CPU'}, AMP={use_amp}, NVDEC={use_nvdec}, NVENC={use_nvenc}, Batch={batch_size}")

# ============================================================
#                   PROCESSING
# ============================================================

if use_multi_gpu and num_gpus > 1:
    precache_model(model_type)
    torch.cuda.empty_cache()
    
    print(f"\n⚡ Multi-GPU: {num_gpus} GPUs")
    
    frames_per_gpu = total_frames // num_gpus
    frame_ranges = [(i * frames_per_gpu, total_frames if i == num_gpus - 1 else (i + 1) * frames_per_gpu) for i in range(num_gpus)]
    for i, (s, e) in enumerate(frame_ranges):
        print(f"   GPU {i}: {s} → {e-1} ({e-s} frames)")
    
    temp_dir = os.path.dirname(out_path) or "."
    os.makedirs(temp_dir, exist_ok=True)
    temp_prefix = f"_temp_{os.getpid()}_"
    
    worker_script = os.path.join(temp_dir, f"{temp_prefix}worker.py")
    create_worker_script(worker_script)
    
    chunk_paths, config_paths, progress_files, log_files = [], [], [], []
    
    for i in range(num_gpus):
        chunk_paths.append(os.path.join(temp_dir, f"{temp_prefix}chunk_{i}.mp4"))
        config_paths.append(os.path.join(temp_dir, f"{temp_prefix}config_{i}.json"))
        progress_files.append(os.path.join(temp_dir, f"{temp_prefix}progress_{i}.txt"))
        log_files.append(os.path.join(temp_dir, f"{temp_prefix}log_{i}.txt"))
        
        with open(progress_files[i], 'w') as f: f.write("0")
        with open(log_files[i], 'w') as f: f.write("")
        
        config = {
            "gpu_id": i, "model_type": model_type,
            "video_path": os.path.abspath(video_path),
            "output_chunk_path": os.path.abspath(chunk_paths[i]),
            "start_frame": frame_ranges[i][0], "end_frame": frame_ranges[i][1],
            "fps": fps, "w": w, "h": h, "max_shift": max_shift,
            "output_choice": output_choice, "use_amp": use_amp,
            "progress_file": os.path.abspath(progress_files[i]),
            "log_file": os.path.abspath(log_files[i]),
            "batch_size": batch_size, "use_nvdec": use_nvdec,
            "use_nvenc": use_nvenc, "use_cuda_remap": use_cuda_remap
        }
        with open(config_paths[i], 'w') as f: json.dump(config, f)
    
    processes = []
    print(f"\n🚀 Launching workers...")
    for i in range(num_gpus):
        p = subprocess.Popen(
            [sys.executable, worker_script, config_paths[i]],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
            env={**os.environ, 'PYTHONUNBUFFERED': '1'}
        )
        processes.append(p)
        print(f"   GPU {i}: PID {p.pid}")
    
    print(f"\nProcessing {total_frames} frames [Batch: {batch_size}]...")
    pbar = tqdm(total=total_frames, desc="Processing", unit="frame", mininterval=1, dynamic_ncols=True, file=sys.stdout)
    last_total = 0
    start_time = time.time()
    
    try:
        while True:
            all_done = all(p.poll() is not None for p in processes)
            current_total = 0
            for pf in progress_files:
                try:
                    with open(pf, 'r') as f:
                        val = f.read().strip()
                        if val: current_total += int(val)
                except: pass
            
            if current_total > last_total:
                pbar.update(current_total - last_total)
                last_total = current_total
            
            if all_done: break
            time.sleep(0.2)
        
        # Final update
        current_total = sum(int(open(pf).read().strip() or "0") for pf in progress_files if os.path.exists(pf))
        if current_total > last_total:
            pbar.update(current_total - last_total)
    finally:
        pbar.close()
    
    elapsed = time.time() - start_time
    
    # Check errors
    has_errors = False
    for i in range(num_gpus):
        err_file = progress_files[i] + ".error"
        if os.path.exists(err_file):
            has_errors = True
            print(f"\n❌ GPU {i} Error:")
            with open(err_file) as f: print(f"   {f.read()[:800]}")
        
        try:
            frames = int(open(progress_files[i]).read().strip() or "0")
        except:
            frames = 0
        
        if frames == 0:
            has_errors = True
            print(f"\n📋 GPU {i} Log:")
            try:
                with open(log_files[i]) as f: print(f"   {f.read()[:1000]}")
            except: pass
            stderr = processes[i].stderr.read()
            if stderr: print(f"   Stderr: {stderr[:500]}")
    
    missing = [p for p in chunk_paths if not os.path.exists(p) or os.path.getsize(p) == 0]
    
    if missing:
        print(f"\n❌ Missing chunks: {missing}")
        print("\n💡 Try: Disable NVDEC/NVENC, reduce batch size")
    else:
        print(f"\n📦 Merging...")
        if merge_videos_ffmpeg(chunk_paths, out_path):
            for path in chunk_paths + config_paths + progress_files + log_files:
                for f in [path, path + ".done", path + ".error"]:
                    if os.path.exists(f): os.remove(f)
            if os.path.exists(worker_script): os.remove(worker_script)
            
            avg_fps = last_total / elapsed if elapsed > 0 else 0
            print(f"\n✅ Done! {last_total} frames, {elapsed:.1f}s, {avg_fps:.2f} FPS")
            print(f"   Saved: {out_path}")
        else:
            print("\n⚠️ Merge failed. Chunks kept.")

else:
    frames = single_gpu_process(model_type, video_path, out_path, max_shift, output_choice, use_amp, use_cuda, batch_size, use_nvenc, use_cuda_remap)
    print(f"\n✅ Done. {frames} frames saved: {out_path}")