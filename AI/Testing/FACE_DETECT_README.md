# Face Detection Script Documentation

## Overview
This script uses OpenCV's Haar Cascade classifier to detect faces in images and draw bounding boxes around them. Perfect for computer vision learning and AI dataset preparation.

- **Script:** `face_detect.py`
- **Input:** Images from webcam capture or any supported image format
- **Output:** `Testing/faces_output.jpg` with detected faces highlighted
- **Platform:** Windows (Python 3.11 recommended)

---

## Prerequisites
- Python 3.11 (recommended for OpenCV/NumPy stability)
- OpenCV installed in a virtual environment
- An image file to process (can be captured using the webcam scripts)

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
pip install opencv-python
```

---

## Running the Script

### Step 1: Capture an Image (Optional)
If you need an image to test with, capture one using your webcam:
```powershell
python ".\Using Webcam in WSL\captureimage.py"
```

### Step 2: Run Face Detection
From the project root:
```powershell
python ".\Testing\face_detect.py"
```

The script will:
- Automatically search for images in standard locations
- Detect faces using Haar Cascade classifier
- Draw blue rectangles around detected faces
- Save the result to `Testing/faces_output.jpg`
- Print the number of faces found

---

## How It Works
1. **Image Loading**: The script searches for images in multiple locations:
   - `Using Webcam in WSL/Image/captured_image.jpg` (from webcam capture)
   - `Testing/captured_image.jpg` (alternative location)
   
2. **Face Detection**: Uses OpenCV's pre-trained Haar Cascade classifier (`haarcascade_frontalface_default.xml`)

3. **Visualization**: Draws blue rectangles around detected faces

4. **Output**: Saves the annotated image for review

---

## Troubleshooting
- **"No image found" error**: Run the webcam capture script first or place an image in one of the expected locations
- **Script crashes**: Ensure your virtual environment is activated and OpenCV is installed
- **No faces detected**: Try adjusting lighting or face angle; Haar cascades work best with frontal faces
- **Import errors**: Reinstall OpenCV with `pip install opencv-python`

---

## Advanced Usage
- **Custom Images**: Place your own images in the expected paths or modify the `image_paths` list in the script
- **Detection Parameters**: Adjust `detectMultiScale` parameters for better results:
  - Scale factor (default 1.1): How much the image size is reduced at each scale
  - Min neighbors (default 4): How many neighbors each candidate rectangle should have to retain it

---

## Using with AI Frameworks
- The output image shows bounding box coordinates that can be extracted for training data
- Detected face coordinates are available in the `faces` variable for programmatic use
- Perfect for creating labeled datasets for facial recognition or emotion detection models

---

## License
MIT or project default.
