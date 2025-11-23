import cv2
img = cv2.imread("Image Specific Testing\image\path_to_image.jpeg")  # Use relative path from current folder
print("Loaded:", img is not None)
if img is not None:
    cv2.imwrite("Image Specific Testing\image\test_out.jpg", img)
else:
    print("Image not found or failed to load.")