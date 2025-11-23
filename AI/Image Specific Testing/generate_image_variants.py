"""
Image Variant Generator
Creates multiple variants of each image in the dataset to improve AI recognition.

Variants include:
- Different sizes (scaled up/down)
- Different lighting (brightness, contrast adjustments)
- Different colorization (grayscale, sepia, color shifts)
- Different quality (compression, blur)
- Different rotations and flips
- Different noise levels
- Different cropping
"""

import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import os
from pathlib import Path


class ImageVariantGenerator:
    def __init__(self, dataset_path="Data_Set"):
        self.dataset_path = dataset_path
        self.variants_per_image = 15  # Generate 15 variants per original image
        
    def generate_brightness_variant(self, img, factor):
        """Adjust brightness by a factor"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = hsv[:, :, 2] * factor
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def generate_contrast_variant(self, img, factor):
        """Adjust contrast by a factor"""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        mean = np.mean(lab[:, :, 0])
        lab[:, :, 0] = mean + (lab[:, :, 0] - mean) * factor
        lab[:, :, 0] = np.clip(lab[:, :, 0], 0, 255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    def generate_grayscale_variant(self, img):
        """Convert to grayscale and back to BGR"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    def generate_sepia_variant(self, img):
        """Apply sepia tone filter"""
        kernel = np.array([[0.272, 0.534, 0.131],
                          [0.349, 0.686, 0.168],
                          [0.393, 0.769, 0.189]])
        sepia = cv2.transform(img, kernel)
        return np.clip(sepia, 0, 255).astype(np.uint8)
    
    def generate_blur_variant(self, img, kernel_size=5):
        """Apply Gaussian blur"""
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    def generate_noise_variant(self, img, noise_level=25):
        """Add random noise to image"""
        noise = np.random.normal(0, noise_level, img.shape).astype(np.float32)
        noisy = img.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    def generate_rotation_variant(self, img, angle):
        """Rotate image by angle"""
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    def generate_flip_variant(self, img, flip_code):
        """Flip image (0=vertical, 1=horizontal, -1=both)"""
        return cv2.flip(img, flip_code)
    
    def generate_scale_variant(self, img, scale_factor):
        """Scale image by a factor"""
        h, w = img.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Resize back to original dimensions
        return cv2.resize(scaled, (w, h), interpolation=cv2.INTER_LINEAR)
    
    def generate_color_shift_variant(self, img, hue_shift=10):
        """Shift colors in HSV space"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def generate_saturation_variant(self, img, factor):
        """Adjust color saturation"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * factor
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def generate_compression_variant(self, img, quality=50):
        """Simulate JPEG compression artifacts"""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', img, encode_param)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    
    def generate_crop_variant(self, img, crop_percent=0.1):
        """Crop and resize back to original size"""
        h, w = img.shape[:2]
        crop_h, crop_w = int(h * crop_percent), int(w * crop_percent)
        
        cropped = img[crop_h:h-crop_h, crop_w:w-crop_w]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    
    def generate_all_variants(self, img):
        """Generate all variants of an image, with descriptions"""
        variants = []
        variant_descriptions = [
            "Darker (70% brightness)",
            "Brighter (130% brightness)",
            "Higher contrast",
            "Lower contrast",
            "Grayscale (B&W)",
            "Sepia tone",
            "Slight blur",
            "Random noise",
            "Rotated 5 degrees",
            "Rotated -5 degrees",
            "Horizontal flip",
            "Scaled down and up (simulate lower resolution)",
            "Color shift",
            "Desaturated (less color)",
            "JPEG compression artifacts"
        ]
        variants.append(self.generate_brightness_variant(img, 0.7))
        variants.append(self.generate_brightness_variant(img, 1.3))
        variants.append(self.generate_contrast_variant(img, 1.3))
        variants.append(self.generate_contrast_variant(img, 0.7))
        variants.append(self.generate_grayscale_variant(img))
        variants.append(self.generate_sepia_variant(img))
        variants.append(self.generate_blur_variant(img, 5))
        variants.append(self.generate_noise_variant(img, 15))
        variants.append(self.generate_rotation_variant(img, 5))
        variants.append(self.generate_rotation_variant(img, -5))
        variants.append(self.generate_flip_variant(img, 1))
        variants.append(self.generate_scale_variant(img, 0.7))
        variants.append(self.generate_color_shift_variant(img, 15))
        variants.append(self.generate_saturation_variant(img, 0.5))
        variants.append(self.generate_compression_variant(img, 60))
        return variants, variant_descriptions
    
    def process_dataset(self):
        """Process entire dataset and generate variants, with debug logging"""
        print("=== Image Variant Generator ===")
        print("Dataset path: {}".format(self.dataset_path))
        print("Variants per image: {}".format(self.variants_per_image))
        print()
        if not os.path.exists(self.dataset_path):
            print("Error: Dataset path '{}' not found!".format(self.dataset_path))
            return
        categories = [d for d in os.listdir(self.dataset_path) 
                     if os.path.isdir(os.path.join(self.dataset_path, d))]
        print("Found {} categories".format(len(categories)))
        print()
        total_originals = 0
        total_variants = 0
        for category in categories:
            print("Processing category: {}".format(category))
            category_path = os.path.join(self.dataset_path, category)
            image_folders = [d for d in os.listdir(category_path)
                           if os.path.isdir(os.path.join(category_path, d)) and d.startswith('image')]
            image_folders.sort(key=lambda x: int(x.replace('image', '')))
            for image_folder in image_folders:
                image_folder_path = os.path.join(category_path, image_folder)
                image_files = [f for f in os.listdir(image_folder_path)
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
                             and not f.startswith('var')]
                if not image_files:
                    print("  Warning: No original image found in {}".format(image_folder))
                    continue
                original_image_name = image_files[0]
                original_image_path = os.path.join(image_folder_path, original_image_name)
                existing_variants = [f for f in os.listdir(image_folder_path)
                                   if f.startswith('var') and f.lower().endswith(('.jpg', '.jpeg'))]
                if len(existing_variants) >= self.variants_per_image:
                    print("  {}: Variants already exist ({}), skipping".format(image_folder, len(existing_variants)))
                    continue
                img = cv2.imread(original_image_path)
                if img is None:
                    print("  Error: Could not read {}".format(original_image_path))
                    continue
                print("  {}: Generating {} variants...".format(image_folder, self.variants_per_image))
                variants, variant_descriptions = self.generate_all_variants(img)
                debug_log_path = os.path.join(image_folder_path, "variant_debug_log.txt")
                with open(debug_log_path, "w") as debug_log:
                    debug_log.write("Variant Debug Log for {}\n".format(image_folder))
                    debug_log.write("Original image: {}\n\n".format(original_image_name))
                    for i, (variant, desc) in enumerate(zip(variants, variant_descriptions), 1):
                        variant_name = "var{:08d}.jpeg".format(i)
                        variant_path = os.path.join(image_folder_path, variant_name)
                        success = cv2.imwrite(variant_path, variant)
                        log_line = "Saved {}: {}\n".format(variant_name, desc)
                        debug_log.write(log_line)
                        if success:
                            print("    Saved {}: {}".format(variant_name, desc))
                            total_variants += 1
                        else:
                            print("    Warning: Failed to save {}".format(variant_name))
                            debug_log.write("    Warning: Failed to save {}\n".format(variant_name))
                total_originals += 1
            print("  Completed {}: {} images processed".format(category, len(image_folders)))
            print()
        print("=== GENERATION COMPLETE ===")
        print("Total original images: {}".format(total_originals))
        print("Total variants created: {}".format(total_variants))
        print("Total images in dataset: {}".format(total_originals + total_variants))
        print()
        print("Dataset is now ready for improved image recognition!")
    
    def generate_for_single_image(self, image_path):
        """Generate variants for a single image (for testing)"""
        print("Generating variants for: {}".format(image_path))
        
        if not os.path.exists(image_path):
            print("Error: Image not found!")
            return
        
        img = cv2.imread(image_path)
        if img is None:
            print("Error: Could not read image!")
            return
        
        variants = self.generate_all_variants(img)
        
        # Save variants in same directory
        image_dir = os.path.dirname(image_path)
        image_name = Path(image_path).stem
        
        for i, variant in enumerate(variants, 1):
            variant_name = "{}_var{:08d}.jpeg".format(image_name, i)
            variant_path = os.path.join(image_dir, variant_name)
            cv2.imwrite(variant_path, variant)
            print("  Saved: {}".format(variant_name))
        
        print("\nGenerated {} variants".format(len(variants)))


def main():
    print("=== Image Variant Generator ===")
    print()
    print("This tool generates multiple variants of each image in your dataset.")
    print("Variants include different:")
    print("  - Brightness levels (darker/brighter)")
    print("  - Contrast levels")
    print("  - Colorization (grayscale, sepia, color shifts)")
    print("  - Quality levels (blur, noise, compression)")
    print("  - Orientations (rotations, flips)")
    print("  - Saturation levels")
    print()
    print("This will help the AI recognize images under various real-world conditions.")
    print()
    print("Processing entire dataset automatically...")
    print()
    
    generator = ImageVariantGenerator()
    generator.process_dataset()


if __name__ == "__main__":
    main()
