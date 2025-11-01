


import warnings
import os
import cv2

print("Script started.")
try:
    # Ensure the correct output directory exists
    output_dir = os.path.join("Using Webcam in WSL", "Video")
    os.makedirs(output_dir, exist_ok=True)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    def try_capture(index):
        print(f"Trying webcam at index {index}...")
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            print(f"Could not open webcam at index {index}.")
            return False
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_path = os.path.join(output_dir, 'captured_video.avi')
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
            return True
        else:
            print("No video captured. Check your webcam and try again.")
            return False

    found = False
    for idx in range(3):
        if try_capture(idx):
            found = True
            break
    if not found:
        print("No working webcam found on indices 0, 1, or 2.")
    print("Script finished.")
except Exception as e:
    print(f"Script crashed with exception: {e}")
