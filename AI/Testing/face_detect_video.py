import cv2
import os

print("Video face detection script started.")

# Ensure output directory exists
os.makedirs('Testing', exist_ok=True)

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Try to load a video (check multiple possible locations)
video_paths = [
    'Using Webcam in WSL/Video/captured_video.avi',  # From your webcam capture
    'Video/captured_video.avi',  # Alternative location
    'Testing/captured_video.avi'  # Another alternative
]

input_video_path = None
for path in video_paths:
    if os.path.exists(path):
        input_video_path = path
        print(f"Loading video from: {path}")
        break

if input_video_path is None:
    print("Error: No video found. Please run the webcam video capture script first:")
    print("python \"Using Webcam in WSL\\capturevideo.py\"")
    exit(1)

# Open the video file
cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file {input_video_path}")
    exit(1)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_path = 'Testing/faces_video_output.avi'
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_count = 0
faces_detected_total = 0

print("Processing video frames...")

while True:
    # Read frame from video
    ret, frame = cap.read()
    
    if not ret:
        break
    
    frame_count += 1
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) > 0:
        faces_detected_total += len(faces)
    
    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Write the frame to output video
    out.write(frame)
    
    # Print progress every 30 frames
    if frame_count % 30 == 0:
        progress = (frame_count / total_frames) * 100
        print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")

# Release everything
cap.release()
out.release()

print(f"\nVideo processing complete!")
print(f"Processed {frame_count} frames")
print(f"Total faces detected: {faces_detected_total}")
print(f"Output saved as: {output_path}")
print("You can now play the output video to see face detection results.")