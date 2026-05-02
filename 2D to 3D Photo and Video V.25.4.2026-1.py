# 2D to 3D Photo and Video V.25.4.2026-1
'''
Made by: Ayan Khan
Version: 25.4.2026-1 (MiDas 3.1 Support)
'''

print("Loading...")
import time
script_start_time = time.time()

# === AGGRESSIVE TIMM FIX ===
import subprocess
import sys
import os
import gc
import shutil

def fix_timm():
    """Nuclear option: completely purge and reinstall timm"""
    to_delete = [m for m in sys.modules if m.startswith('timm')]
    for m in to_delete:
        del sys.modules[m]
    try:
        import site
        for site_dir in site.getsitepackages():
            timm_path = os.path.join(site_dir, 'timm')
            if os.path.exists(timm_path):
                shutil.rmtree(timm_path)
                print(f"🗑️ Removed {timm_path}")
            for item in os.listdir(site_dir):
                if item.startswith('timm') and ('.dist-info' in item or '.egg-info' in item):
                    shutil.rmtree(os.path.join(site_dir, item))
    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")
    print("📦 Installing timm 0.6.13...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "--force-reinstall", "--no-cache-dir", "timm==0.6.13"
    ])
    import timm
    print(f"✅ timm {timm.__version__} ready")
    return timm

need_fix = False
try:
    import timm
    if tuple(map(int, timm.__version__.split('.')[:2])) >= (1, 0):
        print(f"⚠️ Found timm {timm.__version__}, need to downgrade...")
        need_fix = True
except:
    need_fix = True

if need_fix:
    timm = fix_timm()

import json
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

def check_cuda_focus_support():
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            a = cv2.cuda_GpuMat()
            b = cv2.cuda_GpuMat()
            dummy_img = np.random.randint(0, 255, (20, 20), dtype=np.uint8)
            dummy_tmpl = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
            a.upload(dummy_img.astype(np.float32))
            b.upload(dummy_tmpl.astype(np.float32))
            cv2.cuda.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED)
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
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

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

# ---------- Focus Estimator Classes ----------
class CUDAFocusEstimator:
    def __init__(self, w, h, max_disp=300):
        self.max_disp = max_disp
        self.w = w
        self.h = h
        self._left_gpu  = cv2.cuda_GpuMat()
        self._right_gpu = cv2.cuda_GpuMat()
        self._out_l_gpu = cv2.cuda_GpuMat()
        self._out_r_gpu = cv2.cuda_GpuMat()

    def estimate(self, left_gray: np.ndarray, right_gray: np.ndarray) -> int:
        h, w = left_gray.shape
        band_h = h // 3
        y1 = h // 2 - band_h // 2
        y2 = h // 2 + band_h // 2

        left_band  = left_gray[y1:y2, :].astype(np.float32)
        right_band = right_gray[y1:y2, :].astype(np.float32)

        pad = self.max_disp
        right_padded = cv2.copyMakeBorder(right_band, 0, 0, pad, pad,
                                          borderType=cv2.BORDER_REPLICATE)

        self._left_gpu.upload(left_band)
        self._right_gpu.upload(right_padded)

        result_gpu = cv2.cuda.matchTemplate(self._right_gpu, self._left_gpu,
                                            cv2.TM_CCOEFF_NORMED)
        result_cpu = result_gpu.download()

        _, _, _, max_loc = cv2.minMaxLoc(result_cpu)
        shift = max_loc[0] - pad
        return clamp(shift, -self.max_disp, self.max_disp)

    def apply_shift(self, left_view: np.ndarray, right_view: np.ndarray,
                    focus_px: int):
        h, w = left_view.shape[:2]
        half_sep = int(focus_px) // 2

        M_L = np.float32([[1, 0, +half_sep], [0, 1, 0]])
        M_R = np.float32([[1, 0, -half_sep], [0, 1, 0]])

        self._left_gpu.upload(left_view)
        self._right_gpu.upload(right_view)

        cv2.cuda.warpAffine(self._left_gpu,  M_L, (w, h),
                            dst=self._out_l_gpu,
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
        cv2.cuda.warpAffine(self._right_gpu, M_R, (w, h),
                            dst=self._out_r_gpu,
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)

        return self._out_l_gpu.download(), self._out_r_gpu.download()

class CPUFocusEstimator:
    def __init__(self, max_disp=300):
        self.max_disp = max_disp

    def estimate(self, left_gray: np.ndarray, right_gray: np.ndarray) -> int:
        h, w = left_gray.shape
        band_h = h // 3
        y1 = h // 2 - band_h // 2
        y2 = h // 2 + band_h // 2
        left_band  = left_gray[y1:y2, :]
        right_band = right_gray[y1:y2, :]

        pad = self.max_disp
        right_padded = cv2.copyMakeBorder(right_band, 0, 0, pad, pad,
                                          borderType=cv2.BORDER_REPLICATE)
        res = cv2.matchTemplate(right_padded, left_band, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(res)
        shift = max_loc[0] - pad
        return clamp(shift, -self.max_disp, self.max_disp)

    def apply_shift(self, left_view: np.ndarray, right_view: np.ndarray,
                    focus_px: int):
        h, w = left_view.shape[:2]
        half_sep = int(focus_px) // 2
        M_L = np.float32([[1, 0, +half_sep], [0, 1, 0]])
        M_R = np.float32([[1, 0, -half_sep], [0, 1, 0]])
        left_shifted  = cv2.warpAffine(left_view,  M_L, (w, h),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REPLICATE)
        right_shifted = cv2.warpAffine(right_view, M_R, (w, h),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REPLICATE)
        return left_shifted, right_shifted

def build_focus_estimator(use_cuda_focus: bool, w: int, h: int, max_disp: int = 300):
    if use_cuda_focus:
        try:
            est = CUDAFocusEstimator(w, h, max_disp)
            dummy = np.zeros((h, w), dtype=np.uint8)
            est.estimate(dummy, dummy)
            return est
        except Exception as e:
            print(f"⚠️ CUDA focus estimator failed ({e}), using CPU")
    return CPUFocusEstimator(max_disp)

# ---------- Anaglyph builder ----------
def build_anaglyph(left_s: np.ndarray, right_s: np.ndarray,
                   anaglyph_mode: int) -> np.ndarray:
    out = np.zeros_like(left_s)
    if anaglyph_mode == 1:
        out[:, :, 2] = left_s[:, :, 2]
        out[:, :, 1] = right_s[:, :, 1]
        out[:, :, 0] = right_s[:, :, 0]
    elif anaglyph_mode == 2:
        gray_left = cv2.cvtColor(left_s, cv2.COLOR_BGR2GRAY)
        out[:, :, 2] = gray_left
        out[:, :, 1] = right_s[:, :, 1]
        out[:, :, 0] = right_s[:, :, 0]
    else:
        gray_left  = cv2.cvtColor(left_s,  cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_s, cv2.COLOR_BGR2GRAY)
        out[:, :, 2] = gray_left
        out[:, :, 1] = gray_right
        out[:, :, 0] = gray_right
    return out

# ---------- Remapper Classes ----------
class CUDARemapper:
    def __init__(self, w, h, max_shift):
        grid_x_np, grid_y_np = np.meshgrid(
            np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        self.grid_x_gpu = cv2.cuda_GpuMat()
        self.grid_y_gpu = cv2.cuda_GpuMat()
        self.grid_x_gpu.upload(grid_x_np)
        self.grid_y_gpu.upload(grid_y_np)
        self.frame_gpu = cv2.cuda_GpuMat()
        self.map_x_gpu = cv2.cuda_GpuMat()
        self.left_gpu  = cv2.cuda_GpuMat()
        self.right_gpu = cv2.cuda_GpuMat()
        self.grid_x_np = grid_x_np
        self.grid_y_np = grid_y_np
        self.max_shift = max_shift

    def remap(self, frame_bgr, depth_norm):
        shift_map   = depth_norm * float(self.max_shift)
        map_x_left  = (self.grid_x_np - shift_map).astype(np.float32)
        map_x_right = (self.grid_x_np + shift_map).astype(np.float32)
        self.frame_gpu.upload(frame_bgr)
        self.map_x_gpu.upload(map_x_left)
        cv2.cuda.remap(self.frame_gpu, self.map_x_gpu, self.grid_y_gpu,
                       cv2.INTER_LINEAR, cv2.BORDER_REPLICATE, dst=self.left_gpu)
        left_view = self.left_gpu.download()
        self.map_x_gpu.upload(map_x_right)
        cv2.cuda.remap(self.frame_gpu, self.map_x_gpu, self.grid_y_gpu,
                       cv2.INTER_LINEAR, cv2.BORDER_REPLICATE, dst=self.right_gpu)
        right_view = self.right_gpu.download()
        return left_view, right_view

class CPURemapper:
    def __init__(self, w, h, max_shift):
        self.grid_x, self.grid_y = np.meshgrid(
            np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        self.max_shift = max_shift

    def remap(self, frame_bgr, depth_norm):
        shift_map = depth_norm * float(self.max_shift)
        left  = cv2.remap(frame_bgr, self.grid_x - shift_map, self.grid_y,
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        right = cv2.remap(frame_bgr, self.grid_x + shift_map, self.grid_y,
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return left, right

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
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL, text=True, bufsize=1)
        pbar = tqdm(total=duration, desc="FFmpeg Pre-encode", unit="sec",
                    mininterval=1, dynamic_ncols=True, file=sys.stderr) if duration else None
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
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
               "-i", list_path, "-c", "copy", output_path]
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove(list_path)
        return result.returncode == 0
    except Exception as e:
        print(f"FFmpeg merge failed: {e}")
        return False

def precache_model(model_type: str):
    print(f"\n📥 Pre-caching {model_type} model...")
    try:
        _ = torch.hub.load("intel-isl/MiDaS", model_type, verbose=True)
    except RuntimeError as e:
        if "relative_position_index" in str(e) or "Unexpected key" in str(e):
            print("⚠️ Fixing BEiT state_dict compatibility...")
            _ = torch.hub.load("intel-isl/MiDaS", model_type, verbose=True, pretrained=False)
            checkpoint_dir = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints")
            for fname in os.listdir(checkpoint_dir):
                if model_type.lower().replace("_", "") in fname.lower().replace("_", "") or \
                   any(k in fname for k in ["beit", "swin", "next_vit", "levit"] if k in model_type.lower()):
                    ckpt_path = os.path.join(checkpoint_dir, fname)
                    state_dict = torch.load(ckpt_path, map_location="cpu")
                    _.load_state_dict(state_dict, strict=False)
                    break
        else:
            raise
    _ = torch.hub.load("intel-isl/MiDaS", "transforms", verbose=False)
    del _
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("✅ Model cached successfully")

def probe_streams(video_path: str) -> list:
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "stream=index,codec_type,codec_name:stream_tags=language,title,handler_name",
            "-of", "json", video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return []
        data = json.loads(result.stdout)
        streams = []
        audio_idx = sub_idx = 0
        for stream in data.get("streams", []):
            codec_type = stream.get("codec_type", "")
            tags = stream.get("tags", {})
            if codec_type == "audio":
                streams.append({
                    "type": "audio", "stream_index": stream.get("index"),
                    "relative_index": audio_idx,
                    "codec": stream.get("codec_name", "aac"),
                    "lang": tags.get("language", "und"),
                    "title": tags.get("title", tags.get("handler_name", "")),
                })
                audio_idx += 1
            elif codec_type == "subtitle":
                streams.append({
                    "type": "subtitle", "stream_index": stream.get("index"),
                    "relative_index": sub_idx,
                    "codec": stream.get("codec_name", "srt"),
                    "lang": tags.get("language", "und"),
                    "title": tags.get("title", tags.get("handler_name", "")),
                })
                sub_idx += 1
        return streams
    except Exception as e:
        print(f"⚠️ Stream probe failed: {e}")
        return []

def extract_streams(video_path: str, streams: list, temp_dir: str, temp_prefix: str, start_time: float = 0.0, duration: float = None) -> list:
    if not streams:
        return []
    extracted = []
    print(f"\n📤 Extracting {len(streams)} audio/subtitle track(s) [Start: {start_time:.2f}s | Dur: {duration:.2f}s]...")
    for stream in streams:
        stype = stream["type"]
        ridx  = stream["relative_index"]
        codec = stream["codec"]
        if stype == "audio":
            ext = "mka"
        else:
            ext_map = {"subrip": "srt", "srt": "srt", "ass": "ass", "ssa": "ssa",
                       "mov_text": "srt", "webvtt": "vtt", "dvd_subtitle": "sub"}
            ext = ext_map.get(codec, "mks")
        temp_file = os.path.join(temp_dir, f"{temp_prefix}{stype}_{ridx}.{ext}")
        map_spec  = f"0:a:{ridx}" if stype == "audio" else f"0:s:{ridx}"
        cmd = ["ffmpeg", "-y", "-i", video_path]
        if start_time > 0:
            cmd.extend(["-ss", str(start_time)])
        if duration and duration > 0:
            cmd.extend(["-t", str(duration)])
        cmd.extend(["-map", map_spec, "-c", "copy", temp_file])
        try:
            result = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL, timeout=120)
            if result.returncode == 0 and os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
                extracted.append({**stream, "temp_file": temp_file})
                title_display = stream["title"] if stream["title"] else f"{stype.capitalize()} {ridx+1}"
                print(f"   ✅ {title_display} [{stream['lang']}] ({codec})")
            else:
                print(f"   ⚠️ Failed: {stype} track {ridx}")
        except Exception as e:
            print(f"   ⚠️ Error extracting {stype} {ridx}: {e}")
    return extracted

def get_video_duration(video_path: str) -> float:
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return float(result.stdout.strip())
    except:
        return 0.0

def remux_with_streams(video_path: str, extracted_streams: list, output_path: str,
                       original_duration: float = None) -> bool:
    if not extracted_streams:
        if video_path != output_path:
            shutil.move(video_path, output_path)
        return True
    print(f"\n📥 Remuxing {len(extracted_streams)} track(s) into output...")
    temp_output = output_path + ".remux_temp.mp4"
    cmd = ["ffmpeg", "-y", "-i", video_path]
    for stream in extracted_streams:
        cmd.extend(["-i", stream["temp_file"]])
    cmd.extend(["-map", "0:v"])
    for i, stream in enumerate(extracted_streams):
        input_idx = i + 1
        if stream["type"] == "audio":
            cmd.extend(["-map", f"{input_idx}:a:0"])
        else:
            cmd.extend(["-map", f"{input_idx}:s:0"])
    
    cmd.extend(["-c", "copy", "-shortest"])
    
    audio_out_idx = sub_out_idx = 0
    for stream in extracted_streams:
        if stream["type"] == "audio":
            if stream["title"]:
                cmd.extend([f"-metadata:s:a:{audio_out_idx}", f"title={stream['title']}"])
            if stream["lang"]:
                cmd.extend([f"-metadata:s:a:{audio_out_idx}", f"language={stream['lang']}"])
            audio_out_idx += 1
        else:
            if stream["title"]:
                cmd.extend([f"-metadata:s:s:{sub_out_idx}", f"title={stream['title']}"])
            if stream["lang"]:
                cmd.extend([f"-metadata:s:s:{sub_out_idx}", f"language={stream['lang']}"])
            sub_out_idx += 1
    cmd.extend(["-map_chapters", "0", "-map_metadata", "0", temp_output])
    try:
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                text=True, timeout=300)
        if result.returncode == 0 and os.path.exists(temp_output):
            if original_duration and original_duration > 0:
                new_duration = get_video_duration(temp_output)
                if abs(new_duration - original_duration) > 1.0:
                    print(f"   ⚠️ Duration mismatch: {original_duration:.2f}s → {new_duration:.2f}s")
            if os.path.exists(video_path) and video_path != output_path:
                os.remove(video_path)
            shutil.move(temp_output, output_path)
            print(f"   ✅ Remux complete")
            return True
        else:
            print(f"   ❌ Remux failed: {result.stderr[:200] if result.stderr else 'Unknown error'}")
            if video_path != output_path:
                shutil.move(video_path, output_path)
            return False
    except Exception as e:
        print(f"   ❌ Remux error: {e}")
        if video_path != output_path and os.path.exists(video_path):
            shutil.move(video_path, output_path)
        return False

def cleanup_temp_streams(extracted_streams: list):
    for stream in extracted_streams:
        temp_file = stream.get("temp_file", "")
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass

def load_midas_model(model_type: str, device=None):
    try:
        model = torch.hub.load("intel-isl/MiDaS", model_type, verbose=False)
    except RuntimeError as e:
        if "Unexpected key" in str(e):
            print("⚠️ Applying BEiT compatibility fix...")
            model = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=False, verbose=False)
            checkpoint_dir = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints")
            for fname in os.listdir(checkpoint_dir):
                if "beit" in fname.lower() or "swin" in fname.lower() or "next_vit" in fname.lower():
                    ckpt_path = os.path.join(checkpoint_dir, fname)
                    state_dict = torch.load(ckpt_path, map_location="cpu")
                    model.load_state_dict(state_dict, strict=False)
                    break
        else:
            raise
    if device:
        model = model.to(device)
    return model.eval()

def get_midas_transform(model_type: str, midas_transforms):
    if "512" in model_type:
        return midas_transforms.beit512_transform
    elif "LeViT" in model_type or "224" in model_type:
        return midas_transforms.levit_transform
    elif "small" in model_type.lower() or "256" in model_type or "T_256" in model_type:
        return midas_transforms.small_transform
    else:
        return midas_transforms.dpt_transform

# ============================================================
#                     WORKER SCRIPT
# ============================================================
def create_worker_script(script_path: str):
    worker_code = '''#!/usr/bin/env python3
import os, sys, json, time, subprocess, traceback, gc

try:
    import timm
    if tuple(map(int, timm.__version__.split(".")[:2])) >= (1, 0):
        for m in list(sys.modules.keys()):
            if m.startswith("timm"):
                del sys.modules[m]
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "timm"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "timm==0.6.13"])
        import timm
except:
    pass

config_path = sys.argv[1]
with open(config_path, "r") as f:
    config = json.load(f)

gpu_id = config["gpu_id"]
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
os.environ["PYTHONUNBUFFERED"] = "1"

import torch
import torch.nn.functional as F
import cv2
import numpy as np

model_type          = config["model_type"]
video_path          = config["video_path"]
output_chunk_path   = config["output_chunk_path"]
start_frame         = config["start_frame"]
end_frame           = config["end_frame"]
fps                 = config["fps"]
w, h                = config["w"], config["h"]
max_shift           = config["max_shift"]
output_choice       = config["output_choice"]
anaglyph_mode       = config.get("anaglyph_mode", 1)
use_amp             = config["use_amp"]
progress_file       = config["progress_file"]
batch_size          = config.get("batch_size", 9)
use_nvdec           = config.get("use_nvdec", False)
use_nvenc           = config.get("use_nvenc", False)
use_cuda_remap      = config.get("use_cuda_remap", False)
use_cuda_focus      = config.get("use_cuda_focus", False)
focus_cache_interval = config.get("focus_cache_interval", batch_size)
log_file            = config.get("log_file", progress_file + ".log")
script_start_time   = config.get("script_start_time", time.time())
max_runtime         = config.get("max_runtime", 43200)

def log(msg):
    with open(log_file, "a") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\\n")

log(f"Worker started: GPU {gpu_id}, batch={batch_size}, nvdec={use_nvdec}, "
    f"nvenc={use_nvenc}, cuda_focus={use_cuda_focus}, focus_interval={focus_cache_interval}")

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

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

class CUDAFocusEstimator:
    def __init__(self, w, h, max_disp=300):
        self.max_disp = max_disp
        self._left_gpu  = cv2.cuda_GpuMat()
        self._right_gpu = cv2.cuda_GpuMat()
        self._out_l_gpu = cv2.cuda_GpuMat()
        self._out_r_gpu = cv2.cuda_GpuMat()

    def estimate(self, left_gray, right_gray):
        h, w = left_gray.shape
        band_h = h // 3
        y1 = h // 2 - band_h // 2
        y2 = h // 2 + band_h // 2
        left_band  = left_gray[y1:y2, :].astype(np.float32)
        right_band = right_gray[y1:y2, :].astype(np.float32)
        pad = self.max_disp
        right_padded = cv2.copyMakeBorder(right_band, 0, 0, pad, pad,
                                          borderType=cv2.BORDER_REPLICATE)
        self._left_gpu.upload(left_band)
        self._right_gpu.upload(right_padded)
        result_gpu = cv2.cuda.matchTemplate(self._right_gpu, self._left_gpu,
                                            cv2.TM_CCOEFF_NORMED)
        result_cpu = result_gpu.download()
        _, _, _, max_loc = cv2.minMaxLoc(result_cpu)
        return clamp(max_loc[0] - pad, -self.max_disp, self.max_disp)

    def apply_shift(self, left_view, right_view, focus_px):
        h, w = left_view.shape[:2]
        half_sep = int(focus_px) // 2
        M_L = np.float32([[1, 0, +half_sep], [0, 1, 0]])
        M_R = np.float32([[1, 0, -half_sep], [0, 1, 0]])
        self._left_gpu.upload(left_view)
        self._right_gpu.upload(right_view)
        cv2.cuda.warpAffine(self._left_gpu,  M_L, (w, h), dst=self._out_l_gpu,
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        cv2.cuda.warpAffine(self._right_gpu, M_R, (w, h), dst=self._out_r_gpu,
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return self._out_l_gpu.download(), self._out_r_gpu.download()

class CPUFocusEstimator:
    def __init__(self, max_disp=300):
        self.max_disp = max_disp

    def estimate(self, left_gray, right_gray):
        h, w = left_gray.shape
        band_h = h // 3
        y1 = h // 2 - band_h // 2
        y2 = h // 2 + band_h // 2
        left_band  = left_gray[y1:y2, :]
        right_band = right_gray[y1:y2, :]
        pad = self.max_disp
        right_padded = cv2.copyMakeBorder(right_band, 0, 0, pad, pad,
                                          borderType=cv2.BORDER_REPLICATE)
        res = cv2.matchTemplate(right_padded, left_band, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(res)
        return clamp(max_loc[0] - pad, -self.max_disp, self.max_disp)

    def apply_shift(self, left_view, right_view, focus_px):
        h, w = left_view.shape[:2]
        half_sep = int(focus_px) // 2
        M_L = np.float32([[1, 0, +half_sep], [0, 1, 0]])
        M_R = np.float32([[1, 0, -half_sep], [0, 1, 0]])
        left_s  = cv2.warpAffine(left_view,  M_L, (w, h),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        right_s = cv2.warpAffine(right_view, M_R, (w, h),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return left_s, right_s

def build_focus_estimator(use_cuda_focus, w, h, max_disp=300):
    if use_cuda_focus:
        try:
            est = CUDAFocusEstimator(w, h, max_disp)
            dummy = np.zeros((h, w), dtype=np.uint8)
            est.estimate(dummy, dummy)
            return est
        except Exception as e:
            log(f"CUDA focus failed: {e}, using CPU")
    return CPUFocusEstimator(max_disp)

def build_anaglyph(left_s, right_s, anaglyph_mode):
    out = np.zeros_like(left_s)
    if anaglyph_mode == 1:
        out[:, :, 2] = left_s[:, :, 2]
        out[:, :, 1] = right_s[:, :, 1]
        out[:, :, 0] = right_s[:, :, 0]
    elif anaglyph_mode == 2:
        gray_left = cv2.cvtColor(left_s, cv2.COLOR_BGR2GRAY)
        out[:, :, 2] = gray_left
        out[:, :, 1] = right_s[:, :, 1]
        out[:, :, 0] = right_s[:, :, 0]
    else:
        gray_left  = cv2.cvtColor(left_s,  cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_s, cv2.COLOR_BGR2GRAY)
        out[:, :, 2] = gray_left
        out[:, :, 1] = gray_right
        out[:, :, 0] = gray_right
    return out

class NVDECReader:
    def __init__(self, video_path, start_frame, end_frame, w, h, fps):
        self.w, self.h = w, h
        self.frame_size = w * h * 3
        self.frames_to_read = end_frame - start_frame
        self.frames_read = 0
        start_time = start_frame / fps
        cmd = [
            "ffmpeg", "-hwaccel", "cuda",
            "-ss", str(start_time), "-i", video_path,
            "-frames:v", str(self.frames_to_read),
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-vsync", "0", "pipe:1"
        ]
        self.process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            bufsize=self.frame_size * 10,
            env={**os.environ, "PYTHONUNBUFFERED": "1"}
        )
        test_data = self.process.stdout.read(1)
        if not test_data:
            raise RuntimeError("NVDEC failed to start")
        self._first_byte  = test_data
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
        self.w, self.h = w, h
        self.frame_size = w * h * 3
        self.frames_to_read = end_frame - start_frame
        self.frames_read = 0
        start_time = start_frame / fps
        cmd = [
            "ffmpeg", "-ss", str(start_time), "-i", video_path,
            "-frames:v", str(self.frames_to_read),
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-vsync", "0", "pipe:1"
        ]
        self.process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            bufsize=self.frame_size * 10,
            env={**os.environ, "PYTHONUNBUFFERED": "1"}
        )

    def read(self):
        if self.frames_read >= self.frames_to_read:
            return False, None
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

class NVENCWriter:
    def __init__(self, output_path, w, h, fps):
        self.failed = False
        cmd = [
            "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:0",
            "-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr",
            "-cq", "20", "-b:v", "0", "-pix_fmt", "yuv420p", output_path
        ]
        self.process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            bufsize=w * h * 3 * 10,
            env={**os.environ, "PYTHONUNBUFFERED": "1"}
        )

    def write(self, frame):
        if self.failed:
            return
        try:
            self.process.stdin.write(frame.tobytes())
        except BrokenPipeError:
            self.failed = True

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
            raise RuntimeError("Failed to create writer")

    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()

class CUDARemapper:
    def __init__(self, w, h, max_shift):
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        self.grid_x_gpu = cv2.cuda_GpuMat(); self.grid_x_gpu.upload(gx)
        self.grid_y_gpu = cv2.cuda_GpuMat(); self.grid_y_gpu.upload(gy)
        self.frame_gpu  = cv2.cuda_GpuMat()
        self.map_x_gpu  = cv2.cuda_GpuMat()
        self.left_gpu   = cv2.cuda_GpuMat()
        self.right_gpu  = cv2.cuda_GpuMat()
        self.grid_x_np = gx; self.grid_y_np = gy
        self.max_shift  = max_shift

    def remap(self, frame_bgr, depth_norm):
        shift_map   = depth_norm * float(self.max_shift)
        self.frame_gpu.upload(frame_bgr)
        self.map_x_gpu.upload((self.grid_x_np - shift_map).astype(np.float32))
        cv2.cuda.remap(self.frame_gpu, self.map_x_gpu, self.grid_y_gpu,
                       cv2.INTER_LINEAR, cv2.BORDER_REPLICATE, dst=self.left_gpu)
        left_view = self.left_gpu.download()
        self.map_x_gpu.upload((self.grid_x_np + shift_map).astype(np.float32))
        cv2.cuda.remap(self.frame_gpu, self.map_x_gpu, self.grid_y_gpu,
                       cv2.INTER_LINEAR, cv2.BORDER_REPLICATE, dst=self.right_gpu)
        return left_view, self.right_gpu.download()

class CPURemapper:
    def __init__(self, w, h, max_shift):
        self.grid_x, self.grid_y = np.meshgrid(
            np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        self.max_shift = max_shift

    def remap(self, frame_bgr, depth_norm):
        shift_map = depth_norm * float(self.max_shift)
        left  = cv2.remap(frame_bgr, self.grid_x - shift_map, self.grid_y,
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        right = cv2.remap(frame_bgr, self.grid_x + shift_map, self.grid_y,
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return left, right

def main():
    try:
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True
        log(f"CUDA: {torch.cuda.is_available()}")

        log("Loading model...")
        try:
            midas = torch.hub.load("intel-isl/MiDaS", model_type, verbose=False).to(device).eval()
        except RuntimeError as e:
            if "Unexpected key" in str(e):
                log("BEiT fix...")
                midas = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=False, verbose=False)
                ckpt_dir = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints")
                for fname in os.listdir(ckpt_dir):
                    if any(k in fname.lower() for k in ["beit", "swin", "next_vit"]):
                        sd = torch.load(os.path.join(ckpt_dir, fname), map_location="cpu")
                        midas.load_state_dict(sd, strict=False)
                        break
                midas = midas.to(device).eval()
            else:
                raise
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", verbose=False)
        if   "512"  in model_type:                                      transform = midas_transforms.beit512_transform
        elif "LeViT" in model_type or "224" in model_type:              transform = midas_transforms.levit_transform
        elif "small" in model_type.lower() or "256" in model_type or "T_256" in model_type:
                                                                        transform = midas_transforms.small_transform
        else:                                                           transform = midas_transforms.dpt_transform
        log("Model loaded")

        reader = None
        if use_nvdec:
            try:
                reader = NVDECReader(video_path, start_frame, end_frame, w, h, fps)
                log("NVDEC OK")
            except Exception as e:
                log(f"NVDEC failed: {e}")
        if reader is None:
            reader = StandardReader(video_path, start_frame, end_frame, w, h, fps)
            log("StandardReader")

        out_w, out_h = (w, h) if output_choice == 1 else (2 * w, h)
        writer = None
        if use_nvenc:
            try:
                writer = NVENCWriter(output_chunk_path, out_w, out_h, fps)
                log("NVENC OK")
            except Exception as e:
                log(f"NVENC failed: {e}")
        if writer is None:
            writer = StandardWriter(output_chunk_path, out_w, out_h, fps)
            log("StandardWriter")

        remapper = None
        if use_cuda_remap:
            try:
                remapper = CUDARemapper(w, h, max_shift)
                log("CUDA Remap OK")
            except Exception as e:
                log(f"CUDA Remap failed: {e}")
        if remapper is None:
            remapper = CPURemapper(w, h, max_shift)
            log("CPU Remap")

        focus_est = build_focus_estimator(use_cuda_focus, w, h)
        log(f"Focus estimator: {'CUDA' if isinstance(focus_est, CUDAFocusEstimator) else 'CPU'}")

        frames_done          = 0
        last_progress_write  = time.time()
        total_frames         = end_frame - start_frame
        cached_focus_px      = [None]
        frames_since_focus   = 0

        timed_out = False
        log(f"Processing {total_frames} frames, focus_interval={focus_cache_interval}...")

        while frames_done < total_frames:
            # Check Timeout
            if time.time() - script_start_time >= max_runtime:
                log("Max runtime reached. Saving progress and exiting gracefully.")
                timed_out = True
                break

            batch_frames = []
            for _ in range(batch_size):
                ret, frame = reader.read()
                if not ret or frame is None:
                    break
                batch_frames.append(frame)
            if not batch_frames:
                break

            input_tensors = [transform(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in batch_frames]
            batch_tensor  = torch.cat(input_tensors, dim=0).to(device)

            with torch.no_grad():
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        predictions = midas(batch_tensor)
                else:
                    predictions = midas(batch_tensor)
                predictions = F.interpolate(
                    predictions.unsqueeze(1), size=(h, w),
                    mode="bicubic", align_corners=False).squeeze(1)
                depths = predictions.cpu().float().numpy()

            for i in range(len(batch_frames)):
                depth = depths[i]
                if not np.isfinite(depth).all():
                    depth = np.nan_to_num(depth, nan=0.5, posinf=1.0, neginf=0.0)
                depth_norm = normalize_depth_safe(depth)
                left_view, right_view = remapper.remap(batch_frames[i], depth_norm)

                if output_choice == 1:
                    if frames_since_focus >= focus_cache_interval:
                        cached_focus_px[0] = None
                        frames_since_focus = 0

                    gray_l = cv2.cvtColor(left_view,  cv2.COLOR_BGR2GRAY)
                    gray_r = cv2.cvtColor(right_view, cv2.COLOR_BGR2GRAY)
                    if cached_focus_px[0] is None:
                        cached_focus_px[0] = focus_est.estimate(gray_l, gray_r)

                    left_s, right_s = focus_est.apply_shift(
                        left_view, right_view, cached_focus_px[0])
                    output_frame = build_anaglyph(left_s, right_s, anaglyph_mode)
                    frames_since_focus += 1
                else:
                    output_frame = np.hstack((left_view, right_view))

                if isinstance(writer, NVENCWriter) and writer.failed:
                    log("NVENC crashed mid-run → StandardWriter fallback")
                    try: writer.release()
                    except: pass
                    writer = StandardWriter(output_chunk_path, out_w, out_h, fps)

                writer.write(output_frame)
                frames_done += 1

            if time.time() - last_progress_write >= 0.2:
                with open(progress_file, "w") as pf:
                    pf.write(str(frames_done))
                last_progress_write = time.time()

            try: del batch_frames, input_tensors, batch_tensor, predictions, depths
            except NameError: pass
            gc.collect()
            torch.cuda.empty_cache()

        with open(progress_file, "w") as pf:
            pf.write(str(frames_done))
        reader.release()
        writer.release()
        
        with open(progress_file + ".done", "w") as pf:
            if timed_out:
                pf.write(f"TIMEOUT:{frames_done}")
            else:
                pf.write(f"OK:{frames_done}")
        log(f"Done: {frames_done} frames")

    except Exception as e:
        err = f"{e}\\n{traceback.format_exc()}"
        log(f"ERROR: {err}")
        with open(progress_file + ".error", "w") as pf:
            pf.write(err)
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    with open(script_path, 'w') as f:
        f.write(worker_code)

# ============================================================
#               SINGLE GPU PROCESSING
# ============================================================
def single_gpu_process(model_type, video_path, out_path, max_shift, output_choice,
                       anaglyph_mode, use_amp, use_cuda, batch_size=9,
                       use_nvdec=False, use_nvenc=False,
                       use_cuda_remap=False, use_cuda_focus=False,
                       focus_cache_interval=None, start_frame=0, end_frame=None, max_runtime=43200):
    if focus_cache_interval is None:
        focus_cache_interval = batch_size

    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    print("\nInitializing model...")
    midas = load_midas_model(model_type, device)
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = get_midas_transform(model_type, midas_transforms)

    cap_probe = cv2.VideoCapture(video_path)
    fps   = cap_probe.get(cv2.CAP_PROP_FPS) or 30.0
    video_length = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w     = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_probe.release()

    if end_frame is None or end_frame > video_length:
        end_frame = video_length
    total = end_frame - start_frame
    start_time = start_frame / fps if fps > 0 else 0

    out_w, out_h = (w, h) if output_choice == 1 else (2 * w, h)

    reader_process = None
    cap = None
    if use_nvdec and use_cuda:
        try:
            frame_size = w * h * 3
            cmd = [
                'ffmpeg', '-hwaccel', 'cuda',
                '-ss', str(start_time), '-i', video_path,
                '-frames:v', str(total),
                '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-vsync', '0', 'pipe:1'
            ]
            reader_process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                bufsize=frame_size * 10,
                env={**os.environ, 'PYTHONUNBUFFERED': '1'}
            )
            test = reader_process.stdout.read(1)
            if not test:
                raise RuntimeError("NVDEC produced no output")
            reader_process._first_byte  = test
            reader_process._frame_size  = frame_size
            reader_process._first_frame = True
            print("✅ NVDEC enabled")
        except Exception as e:
            print(f"⚠️ NVDEC failed ({e}), using cv2.VideoCapture")
            if reader_process:
                try: reader_process.kill()
                except: pass
            reader_process = None
    if reader_process is None:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    def read_frame():
        if reader_process is not None:
            fsize = reader_process._frame_size
            if reader_process._first_frame:
                raw = reader_process._first_byte + reader_process.stdout.read(fsize - 1)
                reader_process._first_frame = False
            else:
                raw = reader_process.stdout.read(fsize)
            if len(raw) != fsize:
                return False, None
            return True, np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3)).copy()
        else:
            return cap.read()

    def release_reader():
        if reader_process is not None:
            try: reader_process.terminate(); reader_process.wait(timeout=5)
            except: reader_process.kill()
        if cap is not None:
            cap.release()

    writer_process = None
    writer = None
    if use_nvenc:
        try:
            cmd = [
                'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                '-s', f'{out_w}x{out_h}', '-r', str(fps), '-i', 'pipe:0',
                '-c:v', 'h264_nvenc', '-preset', 'p4', '-rc', 'vbr',
                '-cq', '20', '-b:v', '0', '-pix_fmt', 'yuv420p', out_path
            ]
            writer_process = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, bufsize=out_w * out_h * 3 * 10,
                env={**os.environ, 'PYTHONUNBUFFERED': '1'}
            )
            print("✅ NVENC enabled")
        except:
            writer_process = None
    if writer_process is None:
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))

    remapper = None
    if use_cuda_remap and use_cuda:
        try:
            remapper = CUDARemapper(w, h, max_shift)
            print("✅ CUDA Remap enabled")
        except Exception as e:
            print(f"⚠️ CUDA Remap failed ({e}), using CPU")
    if remapper is None:
        remapper = CPURemapper(w, h, max_shift)

    focus_est = None
    if output_choice == 1:
        focus_est = build_focus_estimator(use_cuda_focus and use_cuda, w, h)
        kind = "CUDA" if isinstance(focus_est, CUDAFocusEstimator) else "CPU"
        print(f"✅ Focus estimator: {kind}  (cache interval: every {focus_cache_interval} frames)")

        anaglyph_mode_names = {1: "Full Color", 2: "Half Color", 3: "Gray"}
        print(f"🎨 Anaglyph Mode: {anaglyph_mode_names.get(anaglyph_mode, 'Full Color')}")

    print(f"\nProcessing {total} frames on {device} [Batch: {batch_size}]...")
    pbar = tqdm(total=total, desc="Processing", unit="frame",
                mininterval=1, dynamic_ncols=True, file=sys.stderr)
    frames_written     = 0
    cached_focus_px    = [None]
    frames_since_focus = 0

    try:
        while frames_written < total:
            # Check Timeout
            if time.time() - script_start_time >= max_runtime:
                print("\n⏳ Max runtime reached. Saving progress and exiting gracefully.")
                break

            batch_frames = []
            for _ in range(batch_size):
                if frames_written + len(batch_frames) >= total:
                    break
                ret, frame = read_frame()
                if not ret:
                    break
                batch_frames.append(frame)
            if not batch_frames:
                break

            input_tensors = [transform(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in batch_frames]
            batch_tensor  = torch.cat(input_tensors, dim=0).to(device)

            with torch.no_grad():
                if use_cuda and use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        predictions = midas(batch_tensor)
                else:
                    predictions = midas(batch_tensor)
                predictions = F.interpolate(
                    predictions.unsqueeze(1), size=(h, w),
                    mode="bicubic", align_corners=False).squeeze(1)
                depths = predictions.cpu().float().numpy()

            for i, frame in enumerate(batch_frames):
                depth = depths[i]
                if not np.isfinite(depth).all():
                    depth = np.nan_to_num(depth, nan=0.5, posinf=1.0, neginf=0.0)
                depth_norm = normalize_depth_safe(depth)

                left, right = remapper.remap(frame, depth_norm)

                if output_choice == 1:
                    if frames_since_focus >= focus_cache_interval:
                        cached_focus_px[0] = None
                        frames_since_focus = 0

                    if cached_focus_px[0] is None:
                        gray_l = cv2.cvtColor(left,  cv2.COLOR_BGR2GRAY)
                        gray_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
                        cached_focus_px[0] = focus_est.estimate(gray_l, gray_r)

                    left_s, right_s = focus_est.apply_shift(left, right, cached_focus_px[0])
                    out_frame = build_anaglyph(left_s, right_s, anaglyph_mode)
                    frames_since_focus += 1
                else:
                    out_frame = np.hstack((left, right))

                if writer_process:
                    try:
                        writer_process.stdin.write(out_frame.tobytes())
                    except BrokenPipeError:
                        print("\n⚠️ NVENC crashed, falling back to standard writer...")
                        try: writer_process.stdin.close()
                        except: pass
                        writer_process.kill()
                        writer_process = None
                        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                                 fps, (out_w, out_h))
                        writer.write(out_frame)
                else:
                    writer.write(out_frame)

                frames_written += 1
                pbar.update(1)
            
            try: del batch_frames, input_tensors, batch_tensor, predictions, depths
            except NameError: pass
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        release_reader()
        if writer_process:
            try: writer_process.stdin.close()
            except BrokenPipeError: pass
            try: writer_process.wait(timeout=10)
            except: writer_process.kill()
        elif writer:
            writer.release()
        pbar.close()

    return frames_written

# ============================================================
#                        MAIN PROGRAM
# ============================================================
MODELS = {
    1:  ("DPT_Large",          "dpt_large_384.pt",        384),
    2:  ("DPT_Hybrid",         "dpt_hybrid_384.pt",       384),
    3:  ("MiDaS_small",        "midas_v21_small_256.pt",  256),
    4:  ("DPT_BEiT_L_512",     "dpt_beit_large_512.pt",   512),
    5:  ("DPT_BEiT_L_384",     "dpt_beit_large_384.pt",   384),
    6:  ("DPT_BEiT_B_384",     "dpt_beit_base_384.pt",    384),
    7:  ("DPT_SwinV2_L_384",   "dpt_swin2_large_384.pt",  384),
    8:  ("DPT_SwinV2_B_384",   "dpt_swin2_base_384.pt",   384),
    9:  ("DPT_SwinV2_T_256",   "dpt_swin2_tiny_256.pt",   256),
    10: ("DPT_Swin_L_384",     "dpt_swin_large_384.pt",   384),
    11: ("DPT_Next_ViT_L_384", "dpt_next_vit_large_384.pt", 384),
    12: ("DPT_LeViT_224",      "dpt_levit_224.pt",        224),
}

print("\n📦 Choose Depth Model:")
print("─── MiDaS 3.0 (mostly works with timm 1.0.25) ───")
print("  1. DPT_Large (384)")
print("  2. DPT_Hybrid (384)")
print("  3. MiDaS_small (256) - Fastest")
print("─── MiDaS 3.1 (Better Accuracy) ───")
print("  4. DPT_BEiT_L_512 ⭐ Best Quality (recommended timm 0.6)")
print("  5. DPT_BEiT_L_384 (recommended timm 0.6)")
print("  6. DPT_BEiT_B_384 (recommended timm 0.6)")
print("  7. DPT_SwinV2_L_384 (recommended timm 0.6)")
print("  8. DPT_SwinV2_B_384 (recommended timm 0.6)")
print("  9. DPT_SwinV2_T_256 (usually works with timm 1.0.25 but recommended timm 0.6)")
print(" 10. DPT_Swin_L_384 (usually works with timm 1.0.25 but recommended timm 0.6)")
print(" 11. DPT_Next_ViT_L_384 (recommended timm 0.6)")
print(" 12. DPT_LeViT_224 - Fast")

while True:
    try:
        choice = int(input(">>> ") or "4")
        if choice in MODELS:
            model_type, model_name, model_resolution = MODELS[choice]
            break
        print("Enter 1–12!\n")
    except ValueError:
        print("Enter a number 1–12!\n")

while True:
    try:
        answer = int(input("Model source:\n1. Already Downloaded\n2. Download Automatically/Use Cached\n>>> "))
        if answer == 1:
            while True:
                try:
                    model_path = input("Enter full model path: ").strip().strip('"').strip("'")
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

num_gpus = torch.cuda.device_count()
if num_gpus > 0:
    print(f"\n🔍 Found {num_gpus} GPU(s):")
    for i in range(num_gpus):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("\n🔍 No GPUs found")

HAS_NVENC        = check_nvenc_support()
HAS_NVDEC        = check_nvdec_support()
HAS_CUDA_REMAP   = check_cuda_remap_support()
HAS_CUDA_FOCUS   = check_cuda_focus_support()

print(f"\n🔧 Acceleration Support:")
print(f"   NVDEC: {'✅' if HAS_NVDEC else '❌'}  "
      f"NVENC: {'✅' if HAS_NVENC else '❌'}  "
      f"CUDA Remap: {'✅' if HAS_CUDA_REMAP else '❌'}  "
      f"CUDA Focus: {'✅' if HAS_CUDA_FOCUS else '❌'}")

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

use_amp = use_nvdec = use_nvenc = use_cuda_remap = use_cuda_focus = False
batch_size = 9
focus_cache_interval = batch_size

if use_cuda:
    use_amp = 1
    print("\n⚡ Acceleration Options:")
    if HAS_NVDEC: use_nvdec = 1
    if HAS_NVENC: use_nvenc = 1
    if HAS_CUDA_REMAP: use_cuda_remap = 1
    if HAS_CUDA_FOCUS: use_cuda_focus = 1
    batch_size = 9
    focus_cache_interval = batch_size

while True:
    video_path = input("Enter video input path: ").strip().strip('"').strip("'")
    if os.path.isfile(video_path): break
    print("Not found!")

while True:
    out_path = input("Enter video output path: ").strip().strip('"').strip("'")
    if out_path:
        if not out_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            out_path += ".mp4"
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        break
    print("Cannot be empty!")

print()
while True:
    try:
        duration_type = int(input("Choose duration:\n 1. Full Video\n 2. Custom\n>>> ") or "1")
        if duration_type in [1, 2]:
            break
        print("ERROR: Enter 1 or 2!\n")
    except ValueError:
        print("ERROR: Enter 1 or 2!\n")

if duration_type == 1:
    start_frame = 0
    end_frame = None
elif duration_type == 2:
    while True:
        try:
            start_frame = int(input("Start Frame: "))
            end_frame = int(input("End Frame: "))
            if end_frame is None or end_frame > start_frame:
                break
            else:
                print("End frame must be higher than start frame!\n")
        except ValueError:
            print("Enter integer values!\n")

recode        = 1
output_choice = 1

anaglyph_mode = 1
if output_choice == 1:
    while True:
        try:
            anaglyph_mode = 2
            if anaglyph_mode in [1, 2, 3]:
                break
            print("Enter 1–3!\n")
        except ValueError:
            print("Enter a number 1–3!\n")

max_shift = 20

while True:
    try:
        max_runtime = int(input("Enter max runtime in seconds: "))
        break
    except ValueError:
        print("Enter an integer max runtime!\n")

if recode:
    if is_safe_format(video_path):
        print("✅ Already safe format")
    else:
        video_path = try_reencode_video(video_path)

cap = cv2.VideoCapture(video_path)
fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

if end_frame is None or end_frame > video_length:
    end_frame = video_length

total_frames = end_frame - start_frame

original_duration  = total_frames / fps
detected_streams   = probe_streams(video_path)
temp_stream_dir    = os.path.dirname(out_path) or "."
temp_stream_prefix = f"_temp_streams_{os.getpid()}_"

if detected_streams:
    print(f"\n🔍 Found {len([s for s in detected_streams if s['type']=='audio'])} audio, "
          f"{len([s for s in detected_streams if s['type']=='subtitle'])} subtitle track(s)")
else:
    print("\n🔍 No audio/subtitle tracks found")

print(f"\n📹 Video: {total_frames} frames, {w}x{h}, {fps:.2f} FPS")

anaglyph_mode_names = {1: "Full Color", 2: "Half Color", 3: "Gray"}
output_desc = f"Anaglyph ({anaglyph_mode_names.get(anaglyph_mode, 'Full Color')})" if output_choice == 1 else "SBS Stereo"
print(f"🎨 Output: {output_desc}")

print(f"\n🚀 Config: GPUs={'Multi:'+str(num_gpus) if use_multi_gpu else 'Single' if use_cuda else 'CPU'}, "
      f"AMP={use_amp}, NVDEC={use_nvdec}, NVENC={use_nvenc}, "
      f"CUDARemap={use_cuda_remap}, CUDAFocus={use_cuda_focus}, "
      f"FocusInterval={focus_cache_interval}, Batch={batch_size}")

# ============================================================
#                       PROCESSING
# ============================================================
if use_multi_gpu and num_gpus > 1:
    precache_model(model_type)
    torch.cuda.empty_cache()

    print(f"\n⚡ Multi-GPU: {num_gpus} GPUs")
    frames_per_gpu = total_frames // num_gpus
    frame_ranges = [
        (start_frame + i * frames_per_gpu,
         start_frame + total_frames if i == num_gpus - 1 else start_frame + (i + 1) * frames_per_gpu)
        for i in range(num_gpus)
    ]
    for i, (s, e) in enumerate(frame_ranges):
        print(f"   GPU {i}: {s} → {e-1} ({e-s} frames)")

    temp_dir    = os.path.dirname(out_path) or "."
    temp_prefix = f"_temp_{os.getpid()}_"
    os.makedirs(temp_dir, exist_ok=True)

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
            "output_choice": output_choice, "anaglyph_mode": anaglyph_mode,
            "use_amp": use_amp,
            "progress_file": os.path.abspath(progress_files[i]),
            "log_file": os.path.abspath(log_files[i]),
            "batch_size": batch_size,
            "use_nvdec": use_nvdec, "use_nvenc": use_nvenc,
            "use_cuda_remap": use_cuda_remap,
            "use_cuda_focus": use_cuda_focus,
            "focus_cache_interval": focus_cache_interval,
            "script_start_time": script_start_time,
            "max_runtime": max_runtime
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
    pbar       = tqdm(total=total_frames, desc="Processing", unit="frame",
                      mininterval=1, dynamic_ncols=True, file=sys.stderr)
    last_total = 0

    try:
        while True:
            all_done      = all(p.poll() is not None for p in processes)
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

        current_total = sum(
            int(open(pf).read().strip() or "0")
            for pf in progress_files if os.path.exists(pf)
        )
        if current_total > last_total:
            pbar.update(current_total - last_total)
    finally:
        pbar.close()

    elapsed = time.time() - script_start_time

    # Check for specific Timeout flags written by the workers
    timed_out = False
    for i in range(num_gpus):
        done_file = progress_files[i] + ".done"
        if os.path.exists(done_file):
            if "TIMEOUT" in open(done_file).read():
                timed_out = True

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
        print("\n💡 Try: Disable NVDEC/NVENC/CUDA Focus, reduce batch size")
    else:
        if timed_out:
            print("\n⏳ Max runtime reached. Saving individual synchronized chunks...")
            base_out, ext_out = os.path.splitext(out_path)
            for i in range(num_gpus):
                try:
                    frames_done_chunk = int(open(progress_files[i]).read().strip() or "0")
                except:
                    frames_done_chunk = 0
                
                if frames_done_chunk > 0:
                    chunk_duration = frames_done_chunk / fps
                    chunk_start_time = frame_ranges[i][0] / fps
                    part_out_path = f"{base_out}_Part{i+1}{ext_out}"
                    
                    print(f"\n📦 Packaging Part {i+1} ({frames_done_chunk} frames) -> {part_out_path}")
                    c_streams = extract_streams(video_path, detected_streams, temp_stream_dir, f"part{i}_", chunk_start_time, chunk_duration)
                    remux_with_streams(chunk_paths[i], c_streams, part_out_path, chunk_duration)
                    cleanup_temp_streams(c_streams)

            for path in chunk_paths + config_paths + progress_files + log_files:
                for f in [path, path + ".done", path + ".error"]:
                    if os.path.exists(f): os.remove(f)
            if os.path.exists(worker_script): os.remove(worker_script)
            
            avg_fps = last_total / elapsed if elapsed > 0 else 0
            print(f"\n✅ Timeout Save Complete! {last_total} frames processed, {elapsed:.1f}s, {avg_fps:.2f} FPS")

        else:
            print(f"\n📦 Merging full completed video...")
            temp_merged = out_path + ".video_only.mp4" if detected_streams else out_path
            if merge_videos_ffmpeg(chunk_paths, temp_merged):
                for path in chunk_paths + config_paths + progress_files + log_files:
                    for f in [path, path + ".done", path + ".error"]:
                        if os.path.exists(f): os.remove(f)
                if os.path.exists(worker_script): os.remove(worker_script)
                
                if detected_streams:
                    extracted_streams = extract_streams(video_path, detected_streams, temp_stream_dir, "full_", start_frame / fps, total_frames / fps)
                    remux_with_streams(temp_merged, extracted_streams, out_path, original_duration)
                    cleanup_temp_streams(extracted_streams)
                
                avg_fps = last_total / elapsed if elapsed > 0 else 0
                print(f"\n✅ Done! {last_total} frames, {elapsed:.1f}s, {avg_fps:.2f} FPS")
                print(f"   Saved: {out_path}")
            else:
                print("\n⚠️ Merge failed. Chunks kept.")

else:
    temp_video_out = out_path + ".video_only.mp4" if detected_streams else out_path
    
    start_proc_time = time.time()
    frames = single_gpu_process(
        model_type, video_path, temp_video_out, max_shift, output_choice,
        anaglyph_mode, use_amp, use_cuda, batch_size,
        use_nvdec, use_nvenc, use_cuda_remap, use_cuda_focus, focus_cache_interval,
        start_frame, end_frame, max_runtime
    )
    elapsed = time.time() - start_proc_time

    if frames > 0:
        actual_duration = frames / fps
        if detected_streams:
            extracted_streams = extract_streams(video_path, detected_streams, temp_stream_dir, "single_", start_frame / fps, actual_duration)
            remux_with_streams(temp_video_out, extracted_streams, out_path, actual_duration)
            cleanup_temp_streams(extracted_streams)
        
        avg_fps = frames / elapsed if elapsed > 0 else 0
        print(f"\n✅ Done. {frames} frames processed, {elapsed:.1f}s, {avg_fps:.2f} FPS")
        print(f"   Saved: {out_path}")
    else:
        print("\n❌ No frames were processed.")