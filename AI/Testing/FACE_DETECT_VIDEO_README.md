# Video Face Detection Script Documentation

## Overview
This script processes video files to detect faces frame-by-frame using OpenCV's Haar Cascade classifier. It creates an annotated output video with bounding boxes around detected faces, perfect for computer vision learning and AI dataset preparation.

- **Script:** `face_detect_video.py`
- **Input:** Video files from webcam capture or any supported video format
- **Output:** `Testing/faces_video_output.avi` with detected faces highlighted
- **Platform:** Windows (Python 3.11 recommended)

---

## Prerequisites
- Python 3.11 (recommended for OpenCV/NumPy stability)
- OpenCV installed in a virtual environment
- A video file to process (can be captured using the webcam video script)

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

### Step 1: Capture a Video (Optional)
If you need a video to test with, capture one using your webcam:
```powershell
python ".\Using Webcam in WSL\capturevideo.py"
```

### Step 2: Run Video Face Detection
From the project root:
```powershell
python ".\Testing\face_detect_video.py"
```

The script will:
- Automatically search for videos in standard locations
- Process each frame for face detection
- Draw blue rectangles around detected faces
- Maintain original video properties (resolution, frame rate)
- Show progress updates during processing
- Save the annotated video to `Testing/faces_video_output.avi`

---

## How It Works

### 1. Video Loading
The script searches for videos in multiple locations:
- `Using Webcam in WSL/Video/captured_video.avi` (from webcam capture)
- `Video/captured_video.avi` (alternative location)
- `Testing/captured_video.avi` (another alternative)

### 2. Frame Processing
- Reads video properties (resolution, FPS, frame count)
- Processes each frame individually for face detection
- Uses OpenCV's pre-trained Haar Cascade classifier
- Maintains video quality and timing

### 3. Face Detection
- Converts each frame to grayscale for detection
- Uses `detectMultiScale` with optimized parameters
- Draws blue rectangles around detected faces
- Tracks total faces detected across all frames

### 4. Output Generation
- Preserves original video codec (XVID)
- Maintains original resolution and frame rate
- Saves annotated video for review

---

## Sample Output
```
Video face detection script started.
Loading video from: Using Webcam in WSL/Video/captured_video.avi
Video properties: 640x480, 20 FPS, 100 frames
Processing video frames...
Progress: 30.0% (30/100 frames)
Progress: 60.0% (60/100 frames)
Progress: 90.0% (90/100 frames)

Video processing complete!
Processed 100 frames
Total faces detected: 45
Output saved as: Testing/faces_video_output.avi
You can now play the output video to see face detection results.
```

---

## Troubleshooting

### Common Issues
- **"No video found" error**: Run the webcam video capture script first or place a video in one of the expected locations
- **"Could not open video file" error**: Ensure the video file isn't corrupted and is in a supported format
- **Script crashes during processing**: Check available disk space and memory
- **No faces detected**: Haar cascades work best with frontal faces; try better lighting or positioning

### Performance Tips
- **Large videos**: Processing time depends on video length and resolution
- **Memory usage**: For very long videos, consider processing in chunks
- **Speed optimization**: Reduce video resolution before processing for faster results

---

## Advanced Usage

### Custom Video Paths
Modify the `video_paths` list in the script to add your own video locations:
```python
video_paths = [
    'your/custom/path/video.avi',
    'Using Webcam in WSL/Video/captured_video.avi',
    # ... existing paths
]
```

### Detection Parameters
Adjust face detection sensitivity by modifying `detectMultiScale` parameters:
```python
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
# Parameters: scaleFactor, minNeighbors
# Lower scaleFactor = more thorough but slower
# Higher minNeighbors = fewer false positives
```

### Output Customization
- **Different codec**: Change `fourcc = cv2.VideoWriter_fourcc(*'XVID')` to other formats
- **Rectangle color**: Modify `(255, 0, 0)` to change bounding box color (BGR format)
- **Rectangle thickness**: Change the last parameter in `cv2.rectangle()` for thicker/thinner boxes

---

## Using with AI Frameworks

### Dataset Creation
- Extract frames with face coordinates for training datasets
- Use detected face regions for facial recognition model training
- Create labeled video datasets for emotion detection or behavior analysis

### Integration Examples
- **TensorFlow**: Convert video processing to tf.data pipeline
- **PyTorch**: Use detected regions for custom dataset classes
- **MediaPipe**: Combine with other computer vision tasks

### Data Export
The `faces` variable contains bounding box coordinates `(x, y, width, height)` that can be:
- Exported to JSON/CSV for dataset annotation
- Used for cropping face regions
- Converted to other annotation formats (YOLO, COCO, etc.)

---

## Performance Benchmarks
- **640x480 @ 20 FPS**: ~2-5 seconds per second of video (depending on face count)
- **Memory usage**: ~50-100MB for typical videos
- **CPU usage**: Single-threaded; benefits from faster CPU

---

## License
MIT or project default.
