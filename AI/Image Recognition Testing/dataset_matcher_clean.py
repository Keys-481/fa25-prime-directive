import cv2
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", category=RuntimeWarning)

class DatasetImageMatcher:
    def __init__(self, dataset_path="../Data_Set", algorithm="auto"):
        self.dataset_path = dataset_path
        self.categories = []
        self.dataset_features = {}
        self.algorithm = algorithm  # "original", "enhanced", or "auto"
        print(f"Dataset Image Matcher initialized with {algorithm} algorithm.")
        self.load_dataset()
    
    def load_dataset(self):
        """Load and analyze the dataset structure"""
        print(f"Loading dataset from: {self.dataset_path}")
        
        if not os.path.exists(self.dataset_path):
            print(f"Error: Dataset path '{self.dataset_path}' not found!")
            return
        
        # Get all category folders
        for item in os.listdir(self.dataset_path):
            item_path = os.path.join(self.dataset_path, item)
            if os.path.isdir(item_path):
                self.categories.append(item)
        
        print(f"Found {len(self.categories)} categories:")
        for i, category in enumerate(self.categories, 1):
            category_path = os.path.join(self.dataset_path, category)
            image_count = len([f for f in os.listdir(category_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            print(f"  {i}. {category}: {image_count} images")
        
        # Extract features from all dataset images
        self.extract_dataset_features()
    
    def extract_features_original(self, image_path):
        """Extract features using original color-heavy method"""
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            # Resize for consistency
            img_resized = cv2.resize(img, (256, 256))
            
            # Convert to different color spaces
            gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
            
            # Original feature extraction methods:
            
            # 1. Color histogram features
            hist_b = cv2.calcHist([img_resized], [0], None, [32], [0, 256])
            hist_g = cv2.calcHist([img_resized], [1], None, [32], [0, 256])
            hist_r = cv2.calcHist([img_resized], [2], None, [32], [0, 256])
            hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
            
            color_features = np.concatenate([
                hist_b.flatten(), hist_g.flatten(), hist_r.flatten(),
                hist_h.flatten(), hist_s.flatten(), hist_v.flatten()
            ])
            
            # 2. Basic texture features
            # Edges using Canny
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.mean(edges) / 255.0
            
            # Basic gradient features
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            texture_features = np.array([
                edge_density,
                np.mean(gradient_magnitude),
                np.std(gradient_magnitude),
                np.mean(gray),
                np.std(gray)
            ])
            
            # Combine all features
            features = np.concatenate([color_features, texture_features])
            
            # Normalize features
            features = features / (np.linalg.norm(features) + 1e-7)
            
            return features
            
        except Exception as e:
            print(f"Error extracting original features from {image_path}: {e}")
            return None
    
    def extract_features(self, image_path):
        """Extract features using enhanced B&W-optimized method"""
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            # Resize for consistency
            img_resized = cv2.resize(img, (256, 256))
            
            # Convert to grayscale for B&W optimization
            gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            
            # Enhanced feature extraction for B&W printed images:
            
            # 1. Frequency domain features (FFT)
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            
            # Sample frequency features
            h, w = magnitude_spectrum.shape
            center_h, center_w = h//2, w//2
            
            # Low frequency (overall structure)
            low_freq = magnitude_spectrum[center_h-16:center_h+16, center_w-16:center_w+16]
            low_freq_features = [np.mean(low_freq), np.std(low_freq)]
            
            # High frequency (details/texture)
            high_freq_mask = np.ones_like(magnitude_spectrum)
            high_freq_mask[center_h-32:center_h+32, center_w-32:center_w+32] = 0
            high_freq = magnitude_spectrum * high_freq_mask
            high_freq_features = [np.mean(high_freq), np.std(high_freq)]
            
            # 2. Enhanced gradient analysis
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            gradient_direction = np.arctan2(grad_y, grad_x)
            
            # Gradient histogram
            grad_hist, _ = np.histogram(gradient_magnitude.flatten(), bins=32, range=(0, 255))
            grad_hist = grad_hist.astype(float)
            grad_hist = grad_hist / (np.sum(grad_hist) + 1e-7)
            
            # Direction histogram  
            dir_hist, _ = np.histogram(gradient_direction.flatten(), bins=16, range=(-np.pi, np.pi))
            dir_hist = dir_hist.astype(float)
            dir_hist = dir_hist / (np.sum(dir_hist) + 1e-7)
            
            # 3. Local Binary Pattern-like features
            kernel_size = 3
            pattern_features = []
            for i in range(0, gray.shape[0]-kernel_size, kernel_size*2):
                for j in range(0, gray.shape[1]-kernel_size, kernel_size*2):
                    patch = gray[i:i+kernel_size, j:j+kernel_size]
                    center = patch[1, 1]
                    pattern = (patch > center).astype(int)
                    pattern_features.append(np.mean(pattern))
            
            pattern_features = pattern_features[:64]  # Limit size
            if len(pattern_features) < 64:
                pattern_features.extend([0] * (64 - len(pattern_features)))
            
            # 4. Edge-based features optimized for printed material
            # Multiple edge detection methods
            edges_canny = cv2.Canny(gray, 50, 150)
            edges_laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            edges_laplacian = np.abs(edges_laplacian)
            
            edge_features = [
                np.mean(edges_canny) / 255.0,
                np.std(edges_canny) / 255.0,
                np.mean(edges_laplacian) / 255.0,
                np.std(edges_laplacian) / 255.0,
            ]
            
            # 5. Minimal color features (in case some color info is present)
            hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
            
            # Just basic color statistics
            color_stats = [
                np.mean(hsv[:,:,1]),  # Saturation mean
                np.std(hsv[:,:,1]),   # Saturation std
                np.mean(hsv[:,:,2]),  # Value mean
                np.std(hsv[:,:,2]),   # Value std
            ]
            
            # Combine all enhanced features with adaptive weighting
            freq_features = np.array(low_freq_features + high_freq_features)
            gradient_features = np.concatenate([grad_hist, dir_hist])
            pattern_features = np.array(pattern_features)
            edge_features = np.array(edge_features)
            color_features = np.array(color_stats)
            
            # Adaptive weighting: emphasize texture/pattern over color for B&W
            freq_weight = 3.0
            gradient_weight = 2.5
            pattern_weight = 2.0
            edge_weight = 2.0
            color_weight = 0.5
            
            features = np.concatenate([
                freq_features * freq_weight,
                gradient_features * gradient_weight,
                pattern_features * pattern_weight,
                edge_features * edge_weight,
                color_features * color_weight
            ])
            
            # Normalize features
            features = features / (np.linalg.norm(features) + 1e-7)
            
            return features
            
        except Exception as e:
            print(f"Error extracting enhanced features from {image_path}: {e}")
            return None
    
    def extract_dataset_features(self):
        """Extract features from all images in the dataset"""
        print(f"Extracting features from dataset images using {self.algorithm} algorithm...")
        
        for category in self.categories:
            category_path = os.path.join(self.dataset_path, category)
            self.dataset_features[category] = []
            
            image_files = [f for f in os.listdir(category_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            print(f"Processing category '{category}' ({len(image_files)} images)...")
            
            for img_file in image_files:
                img_path = os.path.join(category_path, img_file)
                
                # Choose feature extraction method based on algorithm setting
                if self.algorithm == "original":
                    features = self.extract_features_original(img_path)
                elif self.algorithm == "enhanced":
                    features = self.extract_features(img_path)
                else:  # auto mode
                    # Use both and combine the results
                    features_orig = self.extract_features_original(img_path)
                    features_enhanced = self.extract_features(img_path)
                    if features_orig is not None and features_enhanced is not None:
                        features = np.concatenate([features_orig, features_enhanced])
                    else:
                        features = features_orig if features_orig is not None else features_enhanced
                
                if features is not None:
                    self.dataset_features[category].append({
                        'filename': img_file,
                        'path': img_path,
                        'features': features
                    })
        
        total_images = sum(len(self.dataset_features[cat]) for cat in self.categories)
        print(f"Feature extraction complete! Processed {total_images} images.")
    
    def capture_and_match(self):
        """Capture image from webcam and match against dataset"""
        print("\n=== AUTOMATIC IMAGE CAPTURE & MATCHING ===")
        print("No camera window needed - automatic capture!")
        print()
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            # Try different camera indices
            for i in range(1, 4):
                print(f"Trying camera index {i}...")
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    print(f"Success! Using camera index {i}")
                    break
            else:
                print("No working camera found")
                return
        
        capture_count = 0
        
        while True:
            print(f"\n--- Capture Session {capture_count + 1} ---")
            print("Position your printed image in front of the camera...")
            
            # Give user time to position image
            for countdown in range(5, 0, -1):
                print(f"Capturing in {countdown} seconds...")
                import time
                time.sleep(1)
            
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                continue
            
            capture_count += 1
            capture_path = f'captured_for_matching_{capture_count}.jpg'
            
            # Save captured frame
            success = cv2.imwrite(capture_path, frame)
            if success:
                print(f"Image captured successfully: {capture_path}")
                print(f"Image size: {frame.shape[1]}x{frame.shape[0]} pixels")
                
                # Match against dataset
                print("Processing match...")
                self.match_image(capture_path)
            else:
                print("Failed to save image")
            
            # Ask if user wants to capture another
            print("\nCapture another image? (y/n): ", end='')
            user_input = input().strip().lower()
            if user_input != 'y' and user_input != 'yes':
                break
        
        cap.release()
        print(f"All done! Total images captured: {capture_count}")
    
    def test_camera(self):
        """Test camera functionality"""
        print("Testing camera...")
        
        # Try to find working camera
        for i in range(4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print(f"Camera found at index {i}")
                    cap.release()
                    return True
                cap.release()
        
        print("No camera found")
        return False
        
    def quick_camera_test(self):
        """Quick test of camera capture without display"""
        print("Quick camera test - capturing test image...")
        
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cv2.imwrite('test_capture.jpg', frame)
                print(f"Camera test successful!")
                print(f"Test image saved as: test_capture.jpg")
                print(f"Image size: {frame.shape[1]}x{frame.shape[0]} pixels")
                cap.release()
                return True
            else:
                print("Failed to capture test image")
        
        cap.release()
        return False
    
    def match_image(self, image_path):
        """Match a single image against the dataset"""
        print(f"\nMatching image: {image_path}")
        print(f"Using {self.algorithm} algorithm for feature extraction...")
        
        # Extract features from the input image using chosen algorithm
        if self.algorithm == "original":
            input_features = self.extract_features_original(image_path)
        elif self.algorithm == "enhanced":
            input_features = self.extract_features(image_path)
        else:  # auto mode
            # Use both and combine the results
            features_orig = self.extract_features_original(image_path)
            features_enhanced = self.extract_features(image_path)
            if features_orig is not None and features_enhanced is not None:
                input_features = np.concatenate([features_orig, features_enhanced])
            else:
                input_features = features_orig if features_orig is not None else features_enhanced
        
        if input_features is None:
            print("Error: Could not extract features from input image")
            return
        
        # Compare against all dataset images
        matches = []
        
        for category in self.categories:
            category_matches = []
            
            for img_data in self.dataset_features[category]:
                # Calculate cosine similarity
                similarity = cosine_similarity(
                    input_features.reshape(1, -1),
                    img_data['features'].reshape(1, -1)
                )[0][0]
                
                category_matches.append({
                    'category': category,
                    'filename': img_data['filename'],
                    'similarity': similarity
                })
            
            # Get best match in this category
            if category_matches:
                best_category_match = max(category_matches, key=lambda x: x['similarity'])
                matches.append(best_category_match)
        
        # Sort all matches by similarity
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        print(f"\n=== MATCHING RESULTS ({self.algorithm.upper()} ALGORITHM) ===")
        print("Top matches:")
        
        for i, match in enumerate(matches[:5]):  # Show top 5 matches
            percentage = match['similarity'] * 100
            print(f"  {i+1}. {match['category']}: {percentage:.1f}%")
            print(f"     Best file: {match['filename']}")
        
        # Determine conclusion
        best_match = matches[0] if matches else None
        if best_match and best_match['similarity'] >= 0.5:
            conclusion = f"BEST MATCH: {best_match['category']} ({best_match['similarity']*100:.1f}%)"
        else:
            conclusion = "No confident match found (all matches below 50%)"
        
        print(f"\nConclusion: {conclusion}")
        
        # Save results to file
        self.save_results(image_path, matches, conclusion)
        
        return matches
    
    def save_results(self, image_path, matches, conclusion):
        """Save matching results to a file"""
        try:
            results_file = "matching_results.txt"
            
            with open(results_file, 'a', encoding='utf-8') as f:
                f.write(f"\n=== MATCH RESULT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                f.write(f"Image: {image_path}\n")
                f.write(f"Algorithm: {self.algorithm}\n")
                f.write(f"Conclusion: {conclusion}\n")
                f.write("Top matches:\n")
                
                for i, match in enumerate(matches[:5]):
                    percentage = match['similarity'] * 100
                    f.write(f"  {i+1}. {match['category']}: {percentage:.1f}% ({match['filename']})\n")
                
                f.write("-" * 50 + "\n")
            
            print(f"\nResults saved to: {results_file}")
        except Exception as e:
            print(f"Error saving results: {e}")


def main():
    print("=== Dataset Image Matcher ===")
    print("This tool captures images and matches them against your dataset categories.")
    print()
    
    # Algorithm selection
    print("Choose matching algorithm:")
    print("1. Original (color-heavy) - best for color digital images")
    print("2. Enhanced (B&W optimized) - best for printed B&W images")
    print("3. Auto (combines both) - works for both types")
    
    algo_choice = input("Enter algorithm choice (1, 2, or 3): ").strip()
    
    if algo_choice == '1':
        algorithm = "original"
    elif algo_choice == '2':
        algorithm = "enhanced"
    else:
        algorithm = "auto"
    
    # Initialize matcher with chosen algorithm
    matcher = DatasetImageMatcher(algorithm=algorithm)
    
    if not matcher.categories:
        print("No dataset categories found. Please check your Data_Set folder.")
        return
    
    print("\nChoose an option:")
    print("1. Auto-capture and match (no camera window)")
    print("2. Match existing image file")
    print("3. Test camera only")
    
    choice = input("Enter choice (1, 2, or 3): ").strip()
    
    if choice == '1':
        matcher.capture_and_match()
    elif choice == '2':
        image_path = input("Enter path to image file: ").strip()
        if os.path.exists(image_path):
            matcher.match_image(image_path)
        else:
            print(f"Error: File '{image_path}' not found!")
    elif choice == '3':
        matcher.quick_camera_test()
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()