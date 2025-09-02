2D to 3D Photo and Videos

This Python program converts 2D videos into 3D formats using depth estimation models. It supports both Red/Cyan Anaglyph and Side-by-Side (SBS) stereo outputs.

Features

Converts 2D videos to 3D using MiDaS depth estimation models

Supports multiple model types: DPT_Large, DPT_Hybrid, MiDaS_small

Offers two output formats: Red/Cyan Anaglyph and SBS Stereo

Optional FFmpeg pre-encoding for reliable video decoding

CUDA and mixed precision support for faster processing

Requirements

Python 3.7+ (Recommended: 3.13.5)

PyTorch

OpenCV

NumPy

FFmpeg (optional, for video re-encoding)

Installation

Clone this repository:

git clone https://github.com/ayankhanakaak/2D-to-3D-Photo-and-Videos.git
cd 2D-to-3D-Photo-and-Videos

Install dependencies:

pip install torch torchvision opencv-python numpy

(Optional) Install FFmpeg for video re-encoding:

FFmpeg Installation Guide

Usage

Run the script and follow the prompts:

python "2D to 3D Photo and Video.py"

You will be prompted to:

Choose a depth estimation model

Select model source (download automatically or you provide path)

Choose device (CPU or CUDA)

Enable mixed precision (optional)

Provide input video path

Choose output format (Anaglyph or SBS)

Set maximum pixel shift

Output

The processed video will be saved in the same directory as the input video.

Output filenames will be auto-generated to avoid overwriting.

License

This project is licensed under the GPL-3.0 License.

Author

Made by: Ayan KhanVersion: 2.9.2025-1
