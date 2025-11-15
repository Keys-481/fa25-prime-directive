import cv2
import os
import warnings

print("Object detection video script started.")

# Ensure output directories exist
os.makedirs('Image Recognition Testing/Video', exist_ok=True)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def capture_video():
    """Capture 5-second video from webcam and save to Image Recognition Testing/Video"""
    print("Starting 5-second video capture...")
    
    def try_capture(index):
        print(f"Trying webcam at index {index}...")
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            print(f"Could not open webcam at index {index}.")
            return False
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_path = 'Image Recognition Testing/Video/captured_video.avi'
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))
        frame_count = 0
        for _ in range(100):  # Capture 100 frames (~5 seconds at 20fps)
            ret, frame = cap.read()
            if ret:
                out.write(frame)
                frame_count += 1
            else:
                print("Failed to capture frame.")
                break
        cap.release()
        out.release()
        if frame_count > 0:
            print(f"Video saved as {output_path} ({frame_count} frames)")
            return output_path
        else:
            print("No video captured. Check your webcam and try again.")
            return None

    # Try to capture video from webcam
    video_path = None
    for idx in range(3):
        video_path = try_capture(idx)
        if video_path:
            break
    
    if not video_path:
        print("No working webcam found on indices 0, 1, or 2.")
        return None
    
    return video_path

def detect_faces_in_video(input_video_path):
    """Process video for face detection and save result to same folder"""
    print(f"Processing video for face detection: {input_video_path}")
    
    # Load the face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Open the video file
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {input_video_path}")
        return False

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")

    # Define the codec and create VideoWriter object for face detection output
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_path = 'Image Recognition Testing/Video/faces_detected_video.avi'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    faces_detected_total = 0

    print("Processing video frames for face detection...")

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

    print(f"\nFace detection processing complete!")
    print(f"Processed {frame_count} frames")
    print(f"Total faces detected: {faces_detected_total}")
    print(f"Face detection output saved as: {output_path}")
    return True

# Main execution
try:
    print("Step 1: Capturing 5-second video...")
    # Always capture a fresh video
    input_video_path = capture_video()
    
    if input_video_path is None:
        print("Failed to capture video. Exiting.")
        exit(1)
    
    print("Step 2: Processing video for face detection...")
    # Process the captured video for face detection
    success = detect_faces_in_video(input_video_path)
    
    if success:
        print("\nScript completed successfully!")
        print("Files created:")
        print("- Image Recognition Testing/Video/captured_video.avi (original)")
        print("- Image Recognition Testing/Video/faces_detected_video.avi (with face detection)")
    else:
        print("Script failed during face detection processing.")

except Exception as e:
    print(f"Script crashed with exception: {e}")
    
print("Object detection video script finished.")