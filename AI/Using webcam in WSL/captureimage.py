

import warnings
import os
import cv2

print("Script started.")
try:
    # Ensure the correct output directory exists
    output_dir = os.path.join("Using Webcam in WSL", "Image")
    os.makedirs(output_dir, exist_ok=True)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    cap = cv2.VideoCapture(0)  # 0 is usually the default webcam
    ret, frame = cap.read()
    output_path = os.path.join(output_dir, 'captured_image.jpg')
    if ret:
        cv2.imwrite(output_path, frame)
        print(f"Image saved as {output_path}")
    else:
        print("Failed to capture image.")
    cap.release()
    print("Script finished.")
except Exception as e:
    print(f"Script crashed with exception: {e}")
