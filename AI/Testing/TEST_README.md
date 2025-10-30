# Test Script for OpenCV in WSL

This README provides instructions for running `test.py`, a simple OpenCV test script, in a WSL (Windows Subsystem for Linux) environment.

## Requirements

- Python 3.8 or newer (recommended: Python 3.12)
- WSL (Windows Subsystem for Linux)
- pip (Python package manager)
- OpenCV (`opencv-python`)
- NumPy

## Setup

1. **Open your WSL terminal.**
2. **Navigate to the project directory:**
	 ```bash
	 cd /mnt/c/Users/{your-user-name}/path-to-file/Git/fa25-prime-directive/AI
	 ```
3. **(Recommended) Create and activate a virtual environment:**
	 ```bash
	 python3 -m venv venv
	 source venv/bin/activate
	 ```
4. **Install requirements:**
	 ```bash
	 pip install --break-system-packages opencv-python numpy
	 ```

## Running test.py

1. Place your input image in the `Testing` folder and update the filename in `test.py` if needed.
2. Run the script:
	 ```bash
	 python3 Testing/test.py
	 ```
3. The output image will be saved as `Testing/output.jpg`.

## No Compilation Needed

Python scripts do not require compilation. Simply run them as shown above.

## Troubleshooting

- **ModuleNotFoundError:**
	- Make sure you have installed all requirements in your (activated) virtual environment.
- **Permission or environment errors:**
	- If you see `externally-managed-environment` errors, use the `--break-system-packages` flag as shown above.
- **cv2.imshow or display errors:**
	- WSL does not support GUI windows by default. This script saves output to a file instead of displaying it.
- **Image not found:**
	- Ensure the input image path in `test.py` is correct and the file exists.

## Example test.py

```python
import cv2
img = cv2.imread('Testing/path_to_image.jpeg')
if img is not None:
		cv2.imwrite('Testing/output.jpg', img)
		print('Image saved as Testing/output.jpg')
else:
		print('Error: Image not found or path is incorrect.')
```

---

For more advanced OpenCV usage, see the main project Instructions.txt or the [OpenCV documentation](https://docs.opencv.org/).
