import os
import shutil
import random
from pathlib import Path
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import math

class DatasetProcessor:
    def __init__(self, source_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        # Create output directories
        self.train_dir = self.output_dir / 'train'
        self.val_dir = self.output_dir / 'val'
        self.test_dir = self.output_dir / 'test'
        
        # Define class mapping
        self.class_mapping = {
            # Ripeness category
            'green': ('ripeness', 'green'),
            'yellow': ('ripeness', 'yellow'),
            'yellow_green': ('ripeness', 'yellow_green'),
            # Bruises category
            'bruised': ('bruises', 'bruised'),
            'not_bruised': ('bruises', 'not_bruised')
        }
        
    def create_directories(self):
        """Create the hierarchical directory structure for train/val/test splits"""
        for split_dir in [self.train_dir, self.val_dir, self.test_dir]:
            split_dir.mkdir(parents=True, exist_ok=True)
            
            # Create ripeness subdirectories
            ripeness_dir = split_dir / 'ripeness'
            ripeness_dir.mkdir(exist_ok=True)
            (ripeness_dir / 'green').mkdir(exist_ok=True)
            (ripeness_dir / 'yellow').mkdir(exist_ok=True)
            (ripeness_dir / 'yellow_green').mkdir(exist_ok=True)
            
            # Create bruises subdirectories
            bruises_dir = split_dir / 'bruises'
            bruises_dir.mkdir(exist_ok=True)
            (bruises_dir / 'bruised').mkdir(exist_ok=True)
            (bruises_dir / 'not_bruised').mkdir(exist_ok=True)
    
    def get_image_files(self, directory):
        """Get all image files from a directory"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(directory.glob(f'*{ext}'))
            image_files.extend(directory.glob(f'*{ext.upper()}'))
        
        return image_files
    
    def split_dataset(self):
        """Split the dataset into train/val/test sets with new hierarchy"""
        print("Splitting dataset into hierarchical structure...")
        
        # Process each original class directory
        for original_class in self.class_mapping.keys():
            source_class_dir = self.source_dir / original_class
            
            if not source_class_dir.exists():
                print(f"Warning: Directory {original_class} not found, skipping...")
                continue
                
            category, new_class_name = self.class_mapping[original_class]
            print(f"Processing {original_class} -> {category}/{new_class_name}")
            
            # Get all images in this class
            images = self.get_image_files(source_class_dir)
            
            if len(images) == 0:
                print(f"No images found in {original_class}")
                continue
            
            # Split into train and temp (val + test)
            train_images, temp_images = train_test_split(
                images, 
                test_size=(self.val_ratio + self.test_ratio),
                random_state=42
            )
            
            # Split temp into val and test
            val_images, test_images = train_test_split(
                temp_images,
                test_size=self.test_ratio / (self.val_ratio + self.test_ratio),
                random_state=42
            )
            
            print(f"  Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")
            
            # Copy images to respective hierarchical directories
            train_dest = self.train_dir / category / new_class_name
            val_dest = self.val_dir / category / new_class_name
            test_dest = self.test_dir / category / new_class_name
            
            self._copy_images(train_images, train_dest)
            self._copy_images(val_images, val_dest)
            self._copy_images(test_images, test_dest)
    
    def _copy_images(self, image_list, destination_dir):
        """Copy images to destination directory"""
        destination_dir.mkdir(parents=True, exist_ok=True)
        for img_path in image_list:
            dest_path = destination_dir / img_path.name
            shutil.copy2(img_path, dest_path)
    
    def apply_rotation(self, image, angle):
        """Apply rotation augmentation"""
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
        return rotated
    
    def apply_flip(self, image, flip_code):
        """Apply flipping augmentation
        flip_code: 0 = vertical flip, 1 = horizontal flip, -1 = both
        """
        return cv2.flip(image, flip_code)
    
    def apply_gaussian_blur(self, image, kernel_size=(5, 5), sigma=1.0):
        """Apply Gaussian blur augmentation"""
        return cv2.GaussianBlur(image, kernel_size, sigma)
    
    def apply_brightness_adjustment(self, image, alpha=1.0, beta=0):
        """Apply brightness/contrast adjustment
        alpha: contrast control (1.0-3.0)
        beta: brightness control (0-100)
        """
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    def apply_noise(self, image, noise_type='gaussian'):
        """Apply noise to image"""
        if noise_type == 'gaussian':
            mean = 0
            var = 0.1
            sigma = var ** 0.5
            gaussian = np.random.normal(mean, sigma, image.shape)
            gaussian = gaussian.reshape(image.shape).astype('uint8')
            noisy = cv2.add(image, gaussian)
            return noisy
        return image
    
    def apply_crop_and_resize(self, image, crop_ratio=0.9):
        """Apply random crop and resize back to original size"""
        h, w = image.shape[:2]
        new_h, new_w = int(h * crop_ratio), int(w * crop_ratio)
        
        # Random crop position
        start_h = random.randint(0, h - new_h)
        start_w = random.randint(0, w - new_w)
        
        cropped = image[start_h:start_h + new_h, start_w:start_w + new_w]
        resized = cv2.resize(cropped, (w, h))
        return resized
    
    def apply_perspective_transform(self, image, max_warp=0.05):
        """Apply minor perspective transform"""
        h, w = image.shape[:2]

        # Small random displacements
        dx = int(w * max_warp)
        dy = int(h * max_warp)

        src_pts = np.float32([
            [0, 0],
            [w - 1, 0],
            [0, h - 1],
            [w - 1, h - 1]
        ])
        dst_pts = np.float32([
            [random.randint(0, dx), random.randint(0, dy)],
            [w - 1 - random.randint(0, dx), random.randint(0, dy)],
            [random.randint(0, dx), h - 1 - random.randint(0, dy)],
            [w - 1 - random.randint(0, dx), h - 1 - random.randint(0, dy)]
        ])

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, M, (w, h))
        return warped

    def get_augmentation_combinations(self, category=None):
        """Get augmentation combinations, customized per category"""
        base_augmentations = [
            ('rot15', lambda img: self.apply_rotation(img, 15)),
            ('rot-15', lambda img: self.apply_rotation(img, -15)),
            ('rot30', lambda img: self.apply_rotation(img, 30)),
            ('rot-30', lambda img: self.apply_rotation(img, -30)),
            ('rot45', lambda img: self.apply_rotation(img, 45)),
            ('rot-45', lambda img: self.apply_rotation(img, -45)),
            ('rot10', lambda img: self.apply_rotation(img, 10)),
            ('rot-10', lambda img: self.apply_rotation(img, -10)),
            ('hflip', lambda img: self.apply_flip(img, 1)),
            ('vflip', lambda img: self.apply_flip(img, 0)),
            ('bothflip', lambda img: self.apply_flip(img, -1)),
            ('noise', lambda img: self.apply_noise(img)),
            ('crop_90', lambda img: self.apply_crop_and_resize(img, 0.9)),
            ('crop_85', lambda img: self.apply_crop_and_resize(img, 0.85)),
            ('crop_80', lambda img: self.apply_crop_and_resize(img, 0.8)),
            ('perspective', lambda img: self.apply_perspective_transform(img)),  # ✅ added for both
        ]

        # Ripeness-specific: add gaussian blur (no brightness/contrast anymore)
        if category == 'ripeness':
            base_augmentations.extend([
                ('blur_light', lambda img: self.apply_gaussian_blur(img, (3, 3), 0.8)),
                ('blur_medium', lambda img: self.apply_gaussian_blur(img, (5, 5), 1.2)),
                ('blur_heavy', lambda img: self.apply_gaussian_blur(img, (7, 7), 1.5)),
            ])
        else:
            # Non-ripeness (e.g., bruises) keeps brightness/contrast
            base_augmentations.extend([
                ('bright_up', lambda img: self.apply_brightness_adjustment(img, 1.0, 20)),
                ('bright_down', lambda img: self.apply_brightness_adjustment(img, 1.0, -20)),
                ('contrast_up', lambda img: self.apply_brightness_adjustment(img, 1.3, 0)),
                ('contrast_down', lambda img: self.apply_brightness_adjustment(img, 0.7, 0)),
            ])

        # Generate combos
        combinations = []
        combinations.extend(base_augmentations)

        # Double combos
        for i, (name1, func1) in enumerate(base_augmentations):
            for j, (name2, func2) in enumerate(base_augmentations):
                if i < j:
                    combo_name = f"{name1}_{name2}"
                    combo_func = lambda img, f1=func1, f2=func2: f2(f1(img))
                    combinations.append((combo_name, combo_func))

        # Triple combos (limited)
        selected_base = base_augmentations[:8]
        for i, (name1, func1) in enumerate(selected_base):
            for j, (name2, func2) in enumerate(selected_base):
                for k, (name3, func3) in enumerate(selected_base):
                    if i < j < k:
                        combo_name = f"{name1}_{name2}_{name3}"
                        combo_func = lambda img, f1=func1, f2=func2, f3=func3: f3(f2(f1(img)))
                        combinations.append((combo_name, combo_func))

        return combinations


    def augment_training_data_massive(self, target_additional_images=10000):
        """Apply massive augmentation separately for ripeness and bruises categories"""
        print(f"Applying massive augmentation to generate {target_additional_images} additional images...")

        total_augmented = 0
        for category in ['ripeness', 'bruises']:
            print(f"\n--- Augmenting {category.upper()} category ---")

            category_dir = self.train_dir / category
            if not category_dir.exists():
                continue

            class_dirs = [d for d in category_dir.iterdir() if d.is_dir()]
            original_train_count = sum(len(self.get_image_files(d)) for d in class_dirs)

            if original_train_count == 0:
                print(f"No images in {category}, skipping...")
                continue

            augmentations_per_image = math.ceil(target_additional_images / original_train_count)
            print(f"{category}: {original_train_count} original images, {augmentations_per_image} augmentations per image")

            all_combinations = self.get_augmentation_combinations(category=category)

            for class_dir in class_dirs:
                class_name = class_dir.name
                images = self.get_image_files(class_dir)
                class_augmented = 0

                for img_path in images:
                    image = cv2.imread(str(img_path))
                    if image is None:
                        continue

                    base_name = img_path.stem
                    extension = img_path.suffix

                    if augmentations_per_image <= len(all_combinations):
                        selected_combinations = random.sample(all_combinations, augmentations_per_image)
                    else:
                        selected_combinations = all_combinations.copy()
                        remaining = augmentations_per_image - len(all_combinations)
                        selected_combinations.extend(random.choices(all_combinations, k=remaining))

                    for aug_name, aug_func in selected_combinations:
                        try:
                            augmented_img = aug_func(image)

                            aug_filename = f"{base_name}_{aug_name}{extension}"
                            aug_path = class_dir / aug_filename
                            counter = 1
                            while aug_path.exists():
                                aug_filename = f"{base_name}_{aug_name}_{counter}{extension}"
                                aug_path = class_dir / aug_filename
                                counter += 1

                            cv2.imwrite(str(aug_path), augmented_img)
                            class_augmented += 1
                            total_augmented += 1

                        except Exception as e:
                            print(f"    Error applying {aug_name} to {img_path}: {e}")
                            continue

                print(f"    Added {class_augmented} augmented images to {category}/{class_name}")

        print(f"\nTotal augmented images created: {total_augmented}")
        print(f"Target was: {target_additional_images}")

    def augment_training_data(self, augmentation_factor=2):
        """Apply standard augmentation separately for ripeness and bruises categories"""
        print("Applying standard augmentation to training data...")

        for category in ['ripeness', 'bruises']:
            print(f"\n--- Augmenting {category.upper()} category ---")
            category_dir = self.train_dir / category
            if not category_dir.exists():
                continue

            class_dirs = [d for d in category_dir.iterdir() if d.is_dir()]
            for class_dir in class_dirs:
                class_name = class_dir.name
                print(f"  Augmenting class: {category}/{class_name}")

                images = self.get_image_files(class_dir)
                total_augmented = 0

                for img_path in images:
                    image = cv2.imread(str(img_path))
                    if image is None:
                        continue

                    base_name = img_path.stem
                    extension = img_path.suffix

                    augmentations = [
                        ('rot15', lambda img: self.apply_rotation(img, 15)),
                        ('rot-15', lambda img: self.apply_rotation(img, -15)),
                        ('rot30', lambda img: self.apply_rotation(img, 30)),
                        ('rot-30', lambda img: self.apply_rotation(img, -30)),
                        ('hflip', lambda img: self.apply_flip(img, 1)),
                        ('vflip', lambda img: self.apply_flip(img, 0)),
                        ('perspective', lambda img: self.apply_perspective_transform(img)),  # ✅ both categories
                    ]

                    # Ripeness gets blur too
                    if category == 'ripeness':
                        augmentations.extend([
                            ('blur_light', lambda img: self.apply_gaussian_blur(img, (3, 3), 0.8)),
                            ('blur_medium', lambda img: self.apply_gaussian_blur(img, (5, 5), 1.2)),
                            ('blur_heavy', lambda img: self.apply_gaussian_blur(img, (7, 7), 1.5)),
                        ])

                    selected_augs = random.sample(augmentations, min(augmentation_factor, len(augmentations)))
                    for aug_name, aug_func in selected_augs:
                        augmented_img = aug_func(image)
                        aug_filename = f"{base_name}_{aug_name}{extension}"
                        aug_path = class_dir / aug_filename
                        cv2.imwrite(str(aug_path), augmented_img)
                        total_augmented += 1

                print(f"    Added {total_augmented} augmented images")

    
    def get_dataset_statistics(self):
        """Print dataset statistics with hierarchical structure"""
        print("\nDataset Statistics:")
        print("=" * 60)
        
        total_train = 0
        total_val = 0
        total_test = 0
        
        # Process each category
        for category in ['ripeness', 'bruises']:
            print(f"\n{category.upper()} Category:")
            print("-" * 40)
            
            category_train = 0
            category_val = 0
            category_test = 0
            
            # Collect all class names across train/val/test
            class_names = set()
            for split_dir in [self.train_dir, self.val_dir, self.test_dir]:
                cat_dir = split_dir / category
                if cat_dir.exists():
                    class_names.update([d.name for d in cat_dir.iterdir() if d.is_dir()])
            
            for class_name in sorted(class_names):
                train_count = len(self.get_image_files(self.train_dir / category / class_name))
                val_count   = len(self.get_image_files(self.val_dir   / category / class_name))
                test_count  = len(self.get_image_files(self.test_dir  / category / class_name))
                
                category_train += train_count
                category_val   += val_count
                category_test  += test_count
                
                print(f"  {class_name:12} - Train: {train_count:4}, Val: {val_count:4}, Test: {test_count:4}")
            
            print(f"  {'Subtotal':12} - Train: {category_train:4}, Val: {category_val:4}, Test: {category_test:4}")
            
            total_train += category_train
            total_val   += category_val
            total_test  += category_test
        
        print("\n" + "=" * 60)
        print(f"{'TOTAL':12} - Train: {total_train:4}, Val: {total_val:4}, Test: {total_test:4}")
        
        total = total_train + total_val + total_test
        if total > 0:
            print(f"{'Ratios':12} - Train: {total_train/total:.1%}, Val: {total_val/total:.1%}, Test: {total_test/total:.1%}")

    
    def print_directory_structure(self):
        """Print the expected output directory structure"""
        print("\nOutput Directory Structure:")
        print("=" * 40)
        structure = """
dataset_split/
├── train/
│   ├── ripeness/
│   │   ├── green/
│   │   ├── yellow/
│   │   └── yellow_green/
│   └── bruises/
│       ├── bruised/
│       └── not_bruised/
├── val/
│   ├── ripeness/
│   │   ├── green/
│   │   ├── yellow/
│   │   └── yellow_green/
│   └── bruises/
│       ├── bruised/
│       └── not_bruised/
└── test/
    ├── ripeness/
    │   ├── green/
    │   ├── yellow/
    │   └── yellow_green/
    └── bruises/
        ├── bruised/
        └── not_bruised/
        """
        print(structure)

def main():
    parser = argparse.ArgumentParser(description='Split dataset into hierarchical structure and apply augmentation')
    parser.add_argument('--source', default='.', help='Source directory containing class folders')
    parser.add_argument('--output', default='./dataset_split', help='Output directory for split dataset')
    parser.add_argument('--augment-factor', type=int, default=2, help='Number of augmentations per training image (standard method)')
    parser.add_argument('--massive-augment', type=int, default=0, help='Target number of additional images for massive augmentation (e.g., 10000)')
    parser.add_argument('--no-augment', action='store_true', help='Skip augmentation step')
    parser.add_argument('--show-structure', action='store_true', help='Show expected output structure and exit')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = DatasetProcessor(
        source_dir=args.source,
        output_dir=args.output,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    if args.show_structure:
        processor.print_directory_structure()
        return
    
    print("Class Mapping:")
    print("-" * 30)
    for original, (category, new_name) in processor.class_mapping.items():
        print(f"{original:12} -> {category}/{new_name}")
    
    # Create directory structure
    processor.create_directories()
    
    # Split dataset
    processor.split_dataset()
    
    # Apply augmentation to training data (unless disabled)
    if not args.no_augment:
        if args.massive_augment > 0:
            processor.augment_training_data_massive(target_additional_images=args.massive_augment)
        else:
            processor.augment_training_data(augmentation_factor=args.augment_factor)
    
    # Print statistics
    processor.get_dataset_statistics()
    
    print(f"\nDataset processing complete! Output saved to: {args.output}")
    processor.print_directory_structure()

if __name__ == "__main__":
    main()