# Precise Image Detection

## How to Compile and Run

1. **Install Python (Recommended: 3.10 or 3.11):**
   - Download from https://www.python.org/
   - Add Python to your PATH during installation.

2. **Install Required Packages:**
   - Open a terminal in this folder.
   - Run:
     ```powershell
     pip install opencv-python numpy scikit-learn
     ```

3. **Run the Script:**
   - In the terminal, execute:
     ```powershell
     python "Precise_image_detection.py"
     ```
   - Follow the on-screen prompts to select the matching algorithm and operation mode.

## Code Details and Usage

- **Purpose:**
  - This script analyzes images from a dataset, extracts features, and matches new images (e.g., from a webcam) against dataset categories for recognition.

- **Dataset Structure:**
  - The dataset should be in the `Data_Set` folder, organized as:
    - `Data_Set/<category>/<imageN>/[original + 15 variants]`

- **Main Features:**
  - Loads all categories and image subfolders, processing every image (original + variants).
  - Extracts features using color histograms, texture, gradients, and frequency domain analysis.
  - Supports three algorithms:
    - **Original:** Color-heavy, best for digital images.
    - **Enhanced:** Optimized for B&W/printed images.
    - **Auto:** Combines both for general use.
  - Can capture images from a webcam and match them against the dataset.
  - Saves matching results to `matching_results.txt` in this folder.

- **Outputs:**
  - Captured images (e.g., `captured_for_matching_1.jpg`) and logs (`matching_results.txt`) are saved in the same folder as the script.

## Known Flaws and Accuracy Limitations

- **Feature Overlap:**
  - If different categories share common visual elements (e.g., clouds in both "cloud" and "pier" images), the matching may confuse them.
  - The model may associate background features (like sky or water) with a category, reducing specificity.

- **Dataset Quality:**
  - Accuracy depends on having diverse, well-curated images for each category.
  - Overlapping or ambiguous images can lead to incorrect matches.

- **Feature Extraction:**
  - The current approach uses general image features, which may not capture unique category details.
  - For improved accuracy, consider refining feature extraction to focus on distinctive elements or use deep learning models.

- **No Training:**
  - This script does not train a classifier; it matches based on feature similarity only.

## Recommendations

- Curate your dataset to minimize overlap of common features between categories.
- Add more diverse images to each category.
- Consider advanced feature extraction or machine learning for better results.

---
For further improvements or troubleshooting, contact the project maintainer or refer to the code comments.