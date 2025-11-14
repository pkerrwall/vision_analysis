import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt

class Augmentation:
    """
    Data augmentation pipeline specifically designed for microscopy images
    """
    
    def __init__(self, image_size=(512, 512)):
        self.image_size = image_size
        self.train_transform = A.Compose([
            # Resize first to ensure consistent size
            A.Resize(self.image_size[0], self.image_size[1]),
            
            # Geometric transformations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=45, p=0.3),
            
            # Elastic deformation (good for biological structures)
            A.ElasticTransform(
                alpha=1, sigma=50, alpha_affine=50, 
                interpolation=cv2.INTER_LINEAR, 
                border_mode=cv2.BORDER_REFLECT_101, 
                p=0.3
            ),
            
            # Intensity augmentations (important for microscopy)
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.2, 
                p=0.5
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            
            # Noise augmentations
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.MultiplicativeNoise(multiplier=[0.9, 1.1], p=0.2),
            
            # Blur augmentations
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.MotionBlur(blur_limit=3, p=0.2),
            
            # Normalize to [0,1] range
            A.Normalize(mean=0.0, std=1.0),
            
            # Convert to tensor
            ToTensorV2()
        ])
        
        self.val_transform = A.Compose([
            A.Resize(self.image_size[0], self.image_size[1]),
            A.Normalize(mean=0.0, std=1.0),
            ToTensorV2()
        ])
    
    def resize_and_save_originals(self, cropped_dir, binary_dir):
        """
        Resize all original images to consistent size and overwrite them
        
        Args:
            cropped_dir: Directory containing cropped images
            binary_dir: Directory containing binary masks
        """
        # Find all cropped images
        image_files = glob.glob(os.path.join(cropped_dir, '*.cropped.tif'))
        
        print(f"Resizing {len(image_files)} original images to {self.image_size}")
        
        for img_path in image_files:
            # Get corresponding mask path
            base_name = os.path.basename(img_path).replace('.cropped.tif', '')
            mask_path = os.path.join(binary_dir, f"{base_name}.binary.tif")
            
            if not os.path.exists(mask_path):
                print(f"Warning: No mask found for {img_path}")
                continue
            
            # Load image and mask
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None or mask is None:
                print(f"Error loading {img_path} or {mask_path}")
                continue
            
            # Print original size
            print(f"Original size for {base_name}: {image.shape}")
            
            # Resize both image and mask
            resized_image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
            resized_mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)  # Use NEAREST for masks
            
            # Ensure mask is binary
            resized_mask = (resized_mask > 127).astype(np.uint8) * 255
            
            # Overwrite original files with resized versions
            cv2.imwrite(img_path, resized_image)
            cv2.imwrite(mask_path, resized_mask)
            
            print(f"Resized and saved: {base_name}")
        
        print("All original images resized!")
    
    def augment_dataset(self, cropped_dir, binary_dir, num_augmentations=5, resize_originals=True):
        """
        Generate augmented versions of your dataset and save them in the same directories
        
        Args:
            cropped_dir: Directory containing cropped images (will also save augmented cropped images here)
            binary_dir: Directory containing binary masks (will also save augmented masks here)
            num_augmentations: Number of augmented versions per original image
            resize_originals: Whether to resize original images first
        """
        # First, resize all original images if requested
        if resize_originals:
            self.resize_and_save_originals(cropped_dir, binary_dir)
        
        # Find all cropped images (now resized)
        image_files = glob.glob(os.path.join(cropped_dir, '*.cropped.tif'))
        
        print(f"Found {len(image_files)} images for augmentation")
        
        for img_path in image_files:
            # Skip augmented files (only process originals)
            if '_aug_' in os.path.basename(img_path):
                continue
                
            # Get corresponding mask path
            base_name = os.path.basename(img_path).replace('.cropped.tif', '')
            mask_path = os.path.join(binary_dir, f"{base_name}.binary.tif")
            
            if not os.path.exists(mask_path):
                print(f"Warning: No mask found for {img_path}")
                continue
            
            # Load image and mask (already resized)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None or mask is None:
                print(f"Error loading {img_path} or {mask_path}")
                continue
            
            # Generate augmentations
            for i in range(num_augmentations):
                # Apply same augmentation to image and mask
                augmented = self.apply_augmentation(image, mask)
                
                if augmented is not None:
                    aug_image, aug_mask = augmented
                    
                    # Save augmented cropped image in cropped directory
                    aug_cropped_path = os.path.join(cropped_dir, f"{base_name}_aug_{i}.cropped.tif")
                    cv2.imwrite(aug_cropped_path, aug_image)
                    
                    # Save augmented mask in binary directory
                    aug_binary_path = os.path.join(binary_dir, f"{base_name}_aug_{i}.binary.tif")
                    cv2.imwrite(aug_binary_path, aug_mask)
                    
                    print(f"Created augmentation {i+1} for {base_name}")
        
        print(f"Augmentation complete!")
        print(f"All images are now {self.image_size} pixels")
        print(f"Augmented images saved in: {cropped_dir}")
        print(f"Augmented masks saved in: {binary_dir}")
    
    def apply_augmentation(self, image, mask):
        """
        Apply augmentation to image-mask pair (assumes images are already resized)
        """
        try:
            # Ensure binary mask
            mask = (mask > 127).astype(np.uint8) * 255
            
            # Create augmentation transform (no resize needed since already done)
            aug_transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Rotate(limit=45, p=0.3),
                A.ElasticTransform(
                    alpha=1, sigma=50, alpha_affine=50, 
                    interpolation=cv2.INTER_LINEAR, 
                    border_mode=cv2.BORDER_REFLECT_101, 
                    p=0.3
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, 
                    contrast_limit=0.2, 
                    p=0.5
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            ])
            
            # Apply augmentation
            augmented = aug_transform(image=image, mask=mask)
            
            return augmented['image'], augmented['mask']
            
        except Exception as e:
            print(f"Error in augmentation: {e}")
            return None
    
    def check_image_sizes(self, cropped_dir, binary_dir):
        """
        Check and print sizes of all images in the dataset
        """
        image_files = glob.glob(os.path.join(cropped_dir, '*.cropped.tif'))
        
        print(f"Checking sizes of {len(image_files)} images:")
        sizes = []
        
        for img_path in image_files:
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                sizes.append(image.shape)
                print(f"{os.path.basename(img_path)}: {image.shape}")
        
        unique_sizes = list(set(sizes))
        print(f"\nUnique sizes found: {unique_sizes}")
        
        if len(unique_sizes) == 1:
            print("✓ All images have the same size!")
        else:
            print("⚠ Images have different sizes - need resizing!")
        
        return unique_sizes
    
    def visualize_augmentations(self, image_path, mask_path, num_samples=4):
        """
        Visualize original and augmented versions
        """
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        fig, axes = plt.subplots(2, num_samples + 1, figsize=(20, 8))
        
        # Show original
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title(f'Original Image\n{image.shape}')
        axes[0, 0].axis('off')
        
        axes[1, 0].imshow(mask, cmap='gray')
        axes[1, 0].set_title(f'Original Mask\n{mask.shape}')
        axes[1, 0].axis('off')
        
        # Show augmentations
        for i in range(num_samples):
            augmented = self.apply_augmentation(image, mask)
            if augmented is not None:
                aug_image, aug_mask = augmented
                
                axes[0, i + 1].imshow(aug_image, cmap='gray')
                axes[0, i + 1].set_title(f'Augmented {i+1}\n{aug_image.shape}')
                axes[0, i + 1].axis('off')
                
                axes[1, i + 1].imshow(aug_mask, cmap='gray')
                axes[1, i + 1].set_title(f'Aug Mask {i+1}\n{aug_mask.shape}')
                axes[1, i + 1].axis('off')
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    """
    Example usage
    """
    # Initialize augmentation with desired model input size
    # Common sizes: (256, 256), (512, 512), (1024, 1024)
    augmenter = Augmentation(image_size=(512, 512))
    
    # Set your paths
    cropped_directory = "/home/bhunn1/vision_analysis/src/data/cropped"
    binary_directory = "/home/bhunn1/vision_analysis/src/data/binary"

    # First, check current image sizes
    print("Checking current image sizes...")
    augmenter.check_image_sizes(cropped_directory, binary_directory)
    
    # Generate augmented dataset (will resize originals first)
    augmenter.augment_dataset(
        cropped_dir=cropped_directory,
        binary_dir=binary_directory,
        num_augmentations=8,  # Creates 8 augmented versions per original
        resize_originals=True  # Set to False if you've already resized
    )
    
    # Check sizes again after processing
    print("\nChecking sizes after augmentation...")
    augmenter.check_image_sizes(cropped_directory, binary_directory)
    
    # Visualize some augmentations (optional)
    sample_images = glob.glob(os.path.join(cropped_directory, '*.cropped.tif'))
    if sample_images:
        # Find an original (non-augmented) image
        orig_images = [img for img in sample_images if '_aug_' not in img]
        if orig_images:
            sample_image = orig_images[0]
            base_name = os.path.basename(sample_image).replace('.cropped.tif', '')
            sample_mask = os.path.join(binary_directory, f"{base_name}.binary.tif")
            
            if os.path.exists(sample_mask):
                print("Visualizing augmentations...")
                augmenter.visualize_augmentations(sample_image, sample_mask)