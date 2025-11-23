Here's how to restore a working Python virtual environment for all scripts in your project:

1. Decide Python Version

For maximum compatibility with OpenCV and numpy, use Python 3.10 or 3.11 (not 3.14, not 2.7 unless absolutely required).

2. Create a New Virtual Environment
Open PowerShell in your project root (AI folder):

# If you want Python 3.11 (recommended for OpenCV/numpy)
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1

3. Install Required Packages

pip install --upgrade pip
pip install opencv-python numpy

If you get "externally-managed-environment" errors:

pip install --break-system-packages opencv-python numpy

4. (Optional) Install other packages

pip install scikit-learn
pip install tensorflow torch torchvision

5. Test Your Environment
Run this to check all versions:

python -c "import sys; print('Python:', sys.version); import numpy; print('NumPy:', numpy.__version__); import cv2; print('OpenCV:', cv2.__version__)"

You should see all versions printed and no crash.

6. Run Your Scripts
Activate the environment each time:

.venv\Scripts\Activate.ps1

Then run any script, e.g.:

python "Image Specific Testing\test.py"
python "Image Recognition Testing\dataset_matcher_clean.py"
python "Testing\face_detect.py"

If you need Python 2.7:

Use py -2 -m virtualenv .venv27 and activate with .venv27\Scripts\Activate.ps1
Install only compatible versions: pip install numpy==1.16.6 opencv-python==4.2.0.32