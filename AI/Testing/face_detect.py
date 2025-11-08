import cv2
import os

print("Face detection script started.")

# Ensure output directory exists
os.makedirs('Testing', exist_ok=True)

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Try to load an image (check multiple possible locations)
image_paths = [
    'Using Webcam in WSL/Image/captured_image.jpg',  # From your webcam capture
    'Testing/captured_image.jpg',  # Alternative location
    'path_to_image.jpg'  # Original (will fail)
]

img = None
for path in image_paths:
    if os.path.exists(path):
        img = cv2.imread(path)
        print(f"Loading image from: {path}")
        break

if img is None:
    print("Error: No image found. Please run the webcam capture script first:")
    print("python \"Using Webcam in WSL\\captureimage.py\"")
    exit(1)

# Convert to grayscale and detect faces
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

print(f"Found {len(faces)} face(s)")

# Draw rectangles around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Save the result
cv2.imwrite('Testing/faces_output.jpg', img)
print('Face detection complete. Result saved as Testing/faces_output.jpg')