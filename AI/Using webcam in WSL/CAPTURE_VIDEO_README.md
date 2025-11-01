# Webcam Video Capture Script Documentation

## Overview
This script captures a short video from your webcam and saves it to a standardized folder for easy use in AI workflows (TensorFlow, TorchVision, etc.).

- **Script:** `capturevideo.py`
- **Output:** `AI/Using Webcam in WSL/Video/captured_video.avi`
- **Platform:** Windows (Python 3.11 recommended)

---

## Prerequisites
- Python 3.11 (recommended for OpenCV/NumPy stability)
- pip (comes with Python)
- OpenCV and NumPy installed in a virtual environment
- Webcam connected and accessible

---

## Setup Instructions

### 1. Install Python 3.11
Download and install from: https://www.python.org/downloads/windows/
- Check "Add Python to PATH" during installation.

### 2. Create and Activate a Virtual Environment
Open a terminal in your project root:
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```
If you get an execution policy error, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Then activate again.

### 3. Install Dependencies
```powershell
pip install --upgrade pip
pip install opencv-python numpy
```

---

## Running the Script

From the project root, run:
```powershell
python "Using Webcam in WSL/capturevideo.py"
```

- The script will print status messages and try webcam indices 0, 1, and 2.
- The captured video will be saved to `AI/Using Webcam in WSL/Video/captured_video.avi`.

---

## Troubleshooting
- If you see warnings about NumPy and MINGW-W64, downgrade to Python 3.11.
- If the script prints "No video captured..." or "Could not open webcam...", ensure your webcam is not in use by another app.
- If the output folder does not exist, the script will create it automatically.

---

## Using the Output with AI Frameworks
- The output video is a standard AVI file (XVID codec), ready for use with TensorFlow, TorchVision, or other computer vision tools.
- For batch dataset creation, run the script multiple times and rename/move files as needed.

---

## License
MIT or project default.
