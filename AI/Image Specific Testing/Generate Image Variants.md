# Image Variant Generator Documentation

## Overview
`generate_image_variants.py` is a Python script designed to automate the creation of multiple augmented variants for each image in a dataset. This process is essential for improving the robustness and generalization of AI models, especially in image recognition tasks.

## Purpose
- Enhance dataset diversity by generating variants of each image.
- Simulate real-world conditions (lighting, blur, noise, etc.) to make AI training more effective.
- Organize output for easy integration into machine learning pipelines.

## Dataset Structure
- The script expects the dataset to be located in the `Data_Set` folder.
- Each category is a subfolder within `Data_Set` (e.g., `Data_Set/cat`, `Data_Set/dog`).
- Each category contains image subfolders (e.g., `image1`, `image2`, ...), each holding one original image and its variants.

Example:
```
Data_Set/
  ├── category1/
  │     ├── image1/
  │     │     ├── original.jpg
  │     │     ├── var00000001.jpeg
  │     │     ├── ...
  │     ├── image2/
  │     │     ├── original.jpg
  │     │     ├── var00000001.jpeg
  │     │     ├── ...
  ├── category2/
      ├── image1/
      ├── ...
```

## How It Works
- Traverses every category and image subfolder in `Data_Set`.
- For each original image, generates 15 variants using different augmentation techniques.
- Saves each variant as `varXXXXXXXX.jpeg` (zero-padded index) in the same image subfolder.
- Creates a `variant_debug_log.txt` in each image subfolder, listing the type of each variant and its filename.

## Augmentation Techniques
The following variants are generated for each image:
1. Darker (70% brightness)
2. Brighter (130% brightness)
3. Higher contrast
4. Lower contrast
5. Grayscale (B&W)
6. Sepia tone
7. Slight blur
8. Random noise
9. Rotated 5 degrees
10. Rotated -5 degrees
11. Horizontal flip
12. Scaled down and up (simulates lower resolution)
13. Color shift
14. Desaturated (less color)
15. JPEG compression artifacts

## Output
- **Variant Images:** Saved as `varXXXXXXXX.jpeg` in the same folder as the original image.
- **Debug Log:** `variant_debug_log.txt` lists each variant's filename and description for traceability.
- **Console Output:** Prints progress and details of each variant as it is generated and saved.

## Usage Instructions
1. **Install Python (Recommended: 3.10 or 3.11):**
   - Download from https://www.python.org/
   - Add Python to your PATH during installation.

2. **Install Required Packages:**
   - Open a terminal in this folder.
   - Run:
     ```powershell
     pip install opencv-python numpy
     ```

3. **Run the Script:**
   - In the terminal, execute:
     ```powershell
     python "generate_image_variants.py"
     ```
   - The script will automatically process all images in `Data_Set`.

## Extending the Script
- To add more variants, define new augmentation methods in the `ImageVariantGenerator` class and update the `generate_all_variants` method.
- You can adjust the number of variants or their parameters for your specific use case.
- The script is modular and can be integrated into larger data processing pipelines.

## Limitations
- Only processes images in the expected folder structure.
- Assumes each image subfolder contains one original image (not named with `var` prefix).
- Does not handle non-image files or corrupted images robustly (prints warnings).
- Augmentation parameters are fixed but can be customized in the code.

## Best Practices
- Curate your dataset to ensure high-quality original images.
- Review generated variants for realism and relevance to your AI task.
- Use the debug log to audit and verify the augmentation process.

## Troubleshooting
- If images are not being processed, check the folder structure and file naming conventions.
- Ensure all required Python packages are installed and compatible with your Python version.
- Review console output and debug logs for warnings or errors.

---
For further customization or support, refer to the code comments or contact the project maintainer.