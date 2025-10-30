import cv2
img = cv2.imread('Testing/path_to_image.jpeg')
if img is not None:
    cv2.imwrite('Testing/output.jpg', img)
    print('Image saved as Testing/output.jpg')
else:
    print('Error: Image not found or path is incorrect.')